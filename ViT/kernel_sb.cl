__kernel void linear_layer(
    __global const float* input,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M,
    const int K,
    const int N ) {

    __local float l_input[TILE_SIZE][TILE_SIZE];
    __local float l_weight[TILE_SIZE][TILE_SIZE];
    
    int gr = get_global_id(0); 
    int gc = get_global_id(1);

    int lr = get_local_id(0);
    int lc = get_local_id(1);

    int group_col = get_group_id(1);

    float sum = 0.0f;

    for (int t = 0; t < K; t += TILE_SIZE) {
        int t_input_col = t + lc;

        l_input[lr][lc] = (gr < M && t_input_col < K) ? input[gr * K + t_input_col] : 0.0f;

        int w_gr = group_col * TILE_SIZE + lr;
        int w_gc = t + lc;

        l_weight[lc][lr] = (w_gr < N && w_gc < K) ? weights[w_gr * K + w_gc] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += l_input[lr][k] * l_weight[k][lc];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gr < M && gc < N) {
        output[gr * N + gc] = sum + bias[gc];
    }
}

__kernel void gelu_activation(__global float* data, const int size) {
    int i = get_global_id(0);
    if (i >= size) return;

    float x = data[i];
    data[i] = 0.5f * x * (1.0f + erf(x * 0.70710678f));
}

__kernel void attention_score (
    __global const float* QKV,
    __global float* scores,
    const int head_offset) {

    int i = get_global_id(0); 
    int j = get_global_id(1);
    int b = get_global_id(2);

    if (i >= TOKENS || j >= TOKENS || b >= BATCH_SIZE) return;

    int qkv_batch_offset = b * (TOKENS * QKV_DIM);
    int score_batch_offset = b * (TOKENS * TOKENS);

    float score = 0.0f;
    float scale = 1.0f / sqrt((float)HEAD_DIM);

    int q_base = qkv_batch_offset + i * QKV_DIM + head_offset;
    int k_base = qkv_batch_offset + j * QKV_DIM + EMBED_DIM + head_offset;

    for (int d = 0; d < HEAD_DIM; d++) {
        float q = QKV[q_base + d];
        float k = QKV[k_base + d];

        score += q * k;
    }

    scores[score_batch_offset + i * TOKENS + j] = score * scale;
}

__kernel void softmax (
    __global float* scores,
    const int size ) {

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

__kernel void context (
    __global const float* scores,
    __global const float* QKV,
    __global float* attn_out,
    const int head_offset ) {

    int i = get_global_id(0);
    int d = get_global_id(1);
    int b = get_global_id(2);

    if (i >= TOKENS || d >= HEAD_DIM || b >= BATCH_SIZE) return;

    int batch_score_offset = b * (TOKENS * TOKENS);
    int batch_qkv_offset = b * (TOKENS * QKV_DIM);
    int batch_out_offset = b * (TOKENS * EMBED_DIM);

    float sum = 0.0f;

    for (int j = 0; j < TOKENS; j++) {
        float s = scores[batch_score_offset + i * TOKENS + j];
        float v = QKV[batch_qkv_offset + j * QKV_DIM + (2 * EMBED_DIM) + head_offset + d];

        sum += s * v;
    }

    int out_idx = batch_out_offset + i * EMBED_DIM + head_offset + d;
    attn_out[out_idx] = sum;
}

__kernel void conv2d (
    __global const float* input,
    __global float* output,
    __global const float* weight,
    __global const float* bias ) {

    int oc = get_global_id(0);

    int patch_index = get_global_id(1);
    int batch_index = get_global_id(2);

    int oh = patch_index / OUTPUT_SIZE;
    int ow = patch_index % OUTPUT_SIZE;

    if (oc >= EMBED_DIM || oh >= OUTPUT_SIZE || ow >= OUTPUT_SIZE || batch_index >= BATCH_SIZE) return;

    int batch_offset = batch_index * (OUTPUT_SIZE * OUTPUT_SIZE * EMBED_DIM);

    float sum = bias[oc];

    for (int ic = 0; ic < CHANNELS; ++ic) {
        for (int kh = 0; kh < PATCH_SIZE; ++kh) {
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

__kernel void layer_norm (
    __global const float* input,
    __global float* output,
    __global const float* weight,
    __global const float* bias ) {

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

__kernel void add (
    __global const float* a,
    __global const float* b,
    __global float* output,
    const int size) {

    int i = get_global_id(0);
    if (i >= size) return;

    output[i] = a[i] + b[i];
}

__kernel void prepare_input (
    __global const float* patches,
    __global const float* cls_token,
    __global const float* pos_emb,
    __global float* output ) {

    int d = get_global_id(0);
    int t = get_global_id(1);
    int b = get_global_id(2);

    if (t >= TOKENS) return;

    int out_idx = b * (TOKENS * EMBED_DIM) + t * EMBED_DIM + d;
    
    float pos_val = pos_emb[t * EMBED_DIM + d];
    float token_val = (t == 0) ? cls_token[d] : patches[b * (NUM_PATCHES * EMBED_DIM) + (t - 1) * EMBED_DIM + d];

    output[out_idx] = token_val + pos_val;
}

__kernel void extract_cls (
    __global const float* input,
    __global float* output ) {

    int b = get_global_id(0);
    int d = get_global_id(1);

    int src_idx = b * (TOKENS * EMBED_DIM) + d;
    int dst_idx = b * EMBED_DIM + d;

    output[dst_idx] = input[src_idx];
}