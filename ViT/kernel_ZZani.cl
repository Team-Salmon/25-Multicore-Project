__kernel void linear(
    __global const float* input,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M,
    const int K,
    const int N) {

    int batch_idx = get_global_id(0);
    int out_idx = get_global_id(1);

    if (out_idx >= N || batch_idx >= M) return;

    float sum = 0.0f;
    int input_offset = batch_idx * K;

    int weight_offset = out_idx * K;

    for (int k = 0; k < K; k += 4) {
        float4 in_vec = vload4(0, &input[input_offset + k]);
        float4 w_vec = vload4(0, &weights[weight_offset + k]);

        sum += dot(in_vec, w_vec);
    }

    output[batch_idx * N + out_idx] = sum + bias[out_idx];
}

inline float gelu(float x) {
    const float INV_SQRT_2 = 0.70710678f;

    const float p = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;

    float scaled_x = x * INV_SQRT_2;
    float abs_x = fabs(scaled_x);
    float sign_val = (scaled_x >= 0.0f) ? 1.0f : -1.0f;
    float t = native_recip(1.0f + p * abs_x);
    float y = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1;
    float erf = sign_val * (1.0f - y * t * native_exp(-abs_x * abs_x));
    return 0.5f * x * (1.0f + erf);
}

/*
inline float gelu(float x) {
    return 0.5f * x * (1.0f + erf(x * 0.70710678f));
}
*/

__kernel void linear_gelu(
    __global const float* input,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M,
    const int K,
    const int N) {

    int batch_idx = get_global_id(0);
    int out_idx = get_global_id(1);

    if (out_idx >= N || batch_idx >= M) return;

    float sum = 0.0f;
    int input_offset = batch_idx * K;

    int weight_offset = out_idx * K;

    for (int k = 0; k < K; k += 4) {
        float4 in_vec = vload4(0, &input[input_offset + k]);
        float4 w_vec = vload4(0, &weights[weight_offset + k]);

        sum += dot(in_vec, w_vec);
    }

    float x = sum + bias[out_idx];
    output[batch_idx * N + out_idx] = gelu(x);
}

__kernel void attn_score(
    __global const float* QKV,
    __global float* scores) {

    int i = get_global_id(0);
    int z = get_global_id(1);

    int batch_idx = z / NUM_HEADS;
    int head_idx = z % NUM_HEADS;

    if (i >= TOKENS || batch_idx >= BATCH_SIZE) return;

    int head_offset = head_idx * HEAD_DIM;

    int token_offset_q = (batch_idx * TOKENS + i) * QKV_DIM; // QKV_DIM = 768 * 3

    int q_start = token_offset_q + head_offset;

    float4 q_reg[16];

#pragma unroll
   for (int k = 0; k < 16; ++k) {
        q_reg[k] = vload4(0, &QKV[q_start + k * 4]);
    }

    // 4. 모든 Key 토큰에 대해 Loop 수행 (Thread Coarsening)
    // 이 스레드 하나가 i행의 모든 열(j=0~196)을 다 계산해버립니다.
    int out_row_offset = (batch_idx * NUM_HEADS + head_idx) * (TOKENS * TOKENS) + (i * TOKENS);
    float scale = 0.125f; // 1/sqrt(64)

    for (int j = 0; j < TOKENS; ++j) {
        
        int token_offset_k = (batch_idx * TOKENS + j) * QKV_DIM;
        int k_start = token_offset_k + EMBED_DIM + head_offset; // K는 Q 다음에 위치

        float sum = 0.0f;

        // 5. 내적 계산 (레지스터에 있는 Q와 Global에 있는 K 연산)
        // 컴파일러가 FMA(Fused Multiply-Add) 최적화를 하도록 유도
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float4 k_vec = vload4(0, &QKV[k_start + k * 4]);
            sum += dot(q_reg[k], k_vec);
        }

        // 결과 저장
        scores[out_row_offset + j] = sum * scale;
    }
}

__kernel void softmax(
    __global float* scores,
    const int size) {

    int row = get_global_id(0);
    int offset = row * size;

    float max_val = scores[offset];
    for (int j = 1; j < size; j++) {
        float val = scores[offset + j];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < size; j++) {
        float exp_val = exp(scores[offset + j] - max_val);
        scores[offset + j] = exp_val;
        sum_exp += exp_val;
    }

    for (int j = 0; j < size; j++) {
        scores[offset + j] /= sum_exp;
    }
}

__kernel void attn_context(
    __global const float* scores,
    __global const float* QKV,
    __global float* attn_out) {

    int i = get_global_id(0);
    int d = get_global_id(1);
    int z = get_global_id(2);

    int batch_idx = z / NUM_HEADS;
    int head_idx = z % NUM_HEADS;

    if (i >= TOKENS || d >= HEAD_DIM || batch_idx >= BATCH_SIZE) return;

    int score_base = (batch_idx * NUM_HEADS + head_idx) * (TOKENS * TOKENS) + i * TOKENS;
    int v_base_offset = 2 * EMBED_DIM + head_idx * HEAD_DIM + d; // V�� 2��°

    float sum = 0.0f;

    for (int j = 0; j < TOKENS; ++j) {
        float s = scores[score_base + j];

        int v_idx = (batch_idx * TOKENS + j) * QKV_DIM + v_base_offset;
        float v = QKV[v_idx];

        sum += s * v;
    }

    int out_idx = (batch_idx * TOKENS + i) * EMBED_DIM + (head_idx * HEAD_DIM + d);
    attn_out[out_idx] = sum;
}

__kernel void patch_embedding(
    __global const float* input,
    __global float* output,
    __global const float* weight,
    __constant float* bias) {

    int oc = get_global_id(0);

    int patch_index = get_global_id(1);
    int batch_index = get_global_id(2);

    int oh = patch_index / OUTPUT_SIZE;
    int ow = patch_index % OUTPUT_SIZE;

    if (oc >= EMBED_DIM || oh >= OUTPUT_SIZE || ow >= OUTPUT_SIZE || batch_index >= BATCH_SIZE) return;

    int batch_offset = batch_index * (OUTPUT_SIZE * OUTPUT_SIZE * EMBED_DIM);

    float sum = bias[oc];

    for (int ic = 0; ic < CHANNELS; ++ic) {
#pragma unroll
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
#pragma unroll
            for (int kw = 0; kw < PATCH_SIZE; ++kw) {
                int ih = oh * PATCH_SIZE + kh;
                int iw = ow * PATCH_SIZE + kw;

                int input_idx = batch_offset + (ic * IMG_SIZE + ih) * IMG_SIZE + iw;
                int kernel_idx = ((oc * CHANNELS + ic) * PATCH_SIZE + kh) * PATCH_SIZE + kw;

                sum += input[input_idx] * weight[kernel_idx];
            }
        }
    }

    batch_offset = batch_index * (OUTPUT_SIZE * OUTPUT_SIZE * EMBED_DIM);
    int out_idx = batch_offset + patch_index * EMBED_DIM + oc;

    output[out_idx] = sum;
}

__kernel void layer_norm(
    __global const float* input,
    __global float* output,
    __global const float* weight,
    __constant float* bias) {

    int t = get_global_id(0);
    if (t >= TOTAL_TOKENS) return;

    int offset = t * EMBED_DIM;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < EMBED_DIM; i++) {
        float val = input[offset + i];
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / EMBED_DIM;
    float var = sum_sq / EMBED_DIM - mean * mean;

    float inv_std = rsqrt(var + 0.000001f);

    for (int i = 0; i < EMBED_DIM; i++) {
        float val = input[offset + i];
        output[offset + i] = (val - mean) * inv_std * weight[i] + bias[i];
    }
}

__kernel void add(
    __global const float* a,
    __global const float* b,
    __global float* output,
    const int size) {

    int i = get_global_id(0);
    if (i >= size) return;

    output[i] = a[i] + b[i];
}

// cls token + position embedding
__kernel void pos_embedding(
    __global const float* patches,
    __global const float* cls_token,
    __global const float* pos_emb,
    __global float* output) {

    int d = get_global_id(0);
    int t = get_global_id(1);
    int b = get_global_id(2);

    if (t >= TOKENS) return;

    int out_idx = b * (TOKENS * EMBED_DIM) + t * EMBED_DIM + d;

    float pos_val = pos_emb[t * EMBED_DIM + d];
    float token_val = (t == 0) ? cls_token[d] : patches[b * (NUM_PATCHES * EMBED_DIM) + (t - 1) * EMBED_DIM + d];

    output[out_idx] = token_val + pos_val;
}

__kernel void extract_cls(
    __global const float* input,
    __global float* output) {

    int b = get_global_id(0);
    int d = get_global_id(1);

    int src_idx = b * (TOKENS * EMBED_DIM) + d;
    int dst_idx = b * EMBED_DIM + d;

    output[dst_idx] = input[src_idx];
}