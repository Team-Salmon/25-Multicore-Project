// 호스트 코드 설정: local_work_size = {4, 64}
#define L_SIZE_0 4   // Output Channel Dim (lx)
#define L_SIZE_1 64  // Token Dim (ly)

// 스레드당 처리량 (기존 유지)
#define PER_THREAD_M 4 // 한 스레드가 처리하는 토큰 수
#define PER_THREAD_N 8 // 한 스레드가 처리하는 출력 수

// 타일 크기
#define TILE_K 16 

// 로컬 메모리 크기 계산
// Input: (64 * 4) tokens * 16 K = 256 * 16
// Weights: 16 K * (4 * 8) outputs = 16 * 32
// Bank Conflict 방지를 위해 열 크기에 +1 (Padding)
#define L_STRIDE_K 17 
#define L_STRIDE_N 33

inline float gelu(float x) {
    const float INV_SQRT_2 = 0.70710678f;

    const float p  = 0.3275911f;
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

__kernel void linear(
    __global const float* input,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M,
    const int K,
    const int N) {

    // 로컬 메모리 선언 (Padding 적용)
    // l_input: [Token Index][K Index]
    __local float l_input[L_SIZE_1 * PER_THREAD_M][L_STRIDE_K];
    // l_weights: [K Index][Output Index]
    __local float l_weights[TILE_K][L_STRIDE_N];

    int lx = get_local_id(0); // 0 ~ 3
    int ly = get_local_id(1); // 0 ~ 63
    int local_linear_id = ly * L_SIZE_0 + lx; // 0 ~ 255

    int out_group_idx = get_group_id(0) * L_SIZE_0 + lx;
    int token_group_idx = get_group_id(1) * L_SIZE_1 + ly;

    int out_idx_base = out_group_idx * PER_THREAD_N;
    int token_idx_base = token_group_idx * PER_THREAD_M;

    // Accumulator
    float acc[PER_THREAD_M][PER_THREAD_N]; // 실제로는 스칼라 누적용

    // 초기화
    #pragma unroll
    for (int t = 0; t < PER_THREAD_M; ++t) {
        #pragma unroll
        for (int c = 0; c < PER_THREAD_N; ++c) {
            acc[t][c] = 0.0f;
        }
    }

    // =========================================================
    // Main Loop: TILE_K (16) 단위로 이동
    // =========================================================
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {

        // -----------------------------------------------------
        // 1. Input 로딩 (협력 로딩 최적화)
        // lx(0~3) 4명이 협력하여 K차원(16개)을 4개씩 나눠서 로딩
        // -----------------------------------------------------
        #pragma unroll
        for (int t = 0; t < PER_THREAD_M; ++t) {
            // 현재 스레드가 담당하는 로컬 토큰 인덱스 (0~255)
            int local_token_row = ly * PER_THREAD_M + t;
            // 글로벌 토큰 인덱스
            int global_token_idx = token_idx_base + t;

            // lx (0~3)에 따라 읽어올 K 오프셋 결정 (0, 4, 8, 12)
            int k_offset = lx * 4; 
            
            if (global_token_idx < M && (k_tile + k_offset) < K) {
                // float4 하나만 딱 읽으면 됨! (매우 효율적)
                float4 val = vload4(0, &input[global_token_idx * K + (k_tile + k_offset)]);
                
                // 로컬 메모리에 저장
                l_input[local_token_row][k_offset + 0] = val.x;
                l_input[local_token_row][k_offset + 1] = val.y;
                l_input[local_token_row][k_offset + 2] = val.z;
                l_input[local_token_row][k_offset + 3] = val.w;
            } else {
                l_input[local_token_row][k_offset + 0] = 0.0f;
                l_input[local_token_row][k_offset + 1] = 0.0f;
                l_input[local_token_row][k_offset + 2] = 0.0f;
                l_input[local_token_row][k_offset + 3] = 0.0f;
            }
        }

        // -----------------------------------------------------
        // 2. Weights 로딩 (선형 로딩)
        // 타일 크기: 16(K) * 32(N) = 512 floats
        // 스레드 수: 256명 -> 인당 2 float씩 읽어서 채움
        // -----------------------------------------------------
        int global_out_group_start = get_group_id(0) * L_SIZE_0 * PER_THREAD_N; // 현재 WorkGroup의 시작 Output Index

        // 각 스레드가 2개씩 로딩 (512개 / 256명 = 2개)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int load_idx = local_linear_id * 2 + i; // 0 ~ 511
            
            // 2D 인덱스로 변환 (Row: k, Col: n)
            int w_k = load_idx / (L_SIZE_0 * PER_THREAD_N); // 0 ~ 15
            int w_n = load_idx % (L_SIZE_0 * PER_THREAD_N); // 0 ~ 31

            if ((k_tile + w_k) < K && (global_out_group_start + w_n) < N) {
                float val = weights[(global_out_group_start + w_n) * K + (k_tile + w_k)]; // Transposed? 확인필요
                // *주의*: 원래 코드에서 weights 읽는 패턴은 weights[out * K + k] 였습니다.
                // 즉, Global Memory에서는 Row-Major로 가정하고 읽습니다.
                
                l_weights[w_k][w_n] = val;
            } else {
                l_weights[w_k][w_n] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // -----------------------------------------------------
        // 3. 연산 (Compute)
        // -----------------------------------------------------
        for (int k = 0; k < TILE_K; ++k) {
            
            // Weights 레지스터 캐싱
            float w_reg[PER_THREAD_N];
            int local_n_base = lx * PER_THREAD_N;
            
            #pragma unroll
            for (int c = 0; c < PER_THREAD_N; ++c) {
                w_reg[c] = l_weights[k][local_n_base + c];
            }

            // 계산
            #pragma unroll
            for (int t = 0; t < PER_THREAD_M; ++t) {
                int local_m_idx = ly * PER_THREAD_M + t;
                // Bank Conflict 없이 읽음 (L_STRIDE_K 덕분)
                float in_val = l_input[local_m_idx][k];

                #pragma unroll
                for (int c = 0; c < PER_THREAD_N; ++c) {
                    acc[t][c] = fma(in_val, w_reg[c], acc[t][c]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // =========================================================
    // 결과 저장 (이전과 동일)
    // =========================================================
    if (out_idx_base >= N || token_idx_base >= M) return;

    float4 b0 = vload4(0, &bias[out_idx_base + 0]);
    float4 b1 = vload4(0, &bias[out_idx_base + 4]);

    #pragma unroll
    for (int t = 0; t < PER_THREAD_M; ++t) {
        int current_token_idx = token_idx_base + t;

        if (current_token_idx < M) {
            float4 res0 = (float4)(acc[t][0], acc[t][1], acc[t][2], acc[t][3]) + b0;
            float4 res1 = (float4)(acc[t][4], acc[t][5], acc[t][6], acc[t][7]) + b1;

            __global float* out_ptr = &output[current_token_idx * N + out_idx_base];
            vstore4(res0, 0, out_ptr + 0);
            vstore4(res1, 0, out_ptr + 4);
        }
    }
}

__kernel void linear_gelu(
    __global const float* input,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M,
    const int K,
    const int N) {

    int out_group_idx = get_global_id(0);
    int token_group_idx = get_global_id(1);

    int out_idx_base = out_group_idx * 8;
    int token_idx_base = token_group_idx * 4;

    if (out_idx_base >= N || token_idx_base >= M) return;

    float4 acc[4][8];

#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            acc[t][c] = 0.0f;
        }
    }

    int wt_base = out_idx_base * K;

    for (int k = 0; k < K; k += 4) {
        float4 w[8];

#pragma unroll
        for (int c = 0; c < 8; ++c) {
            w[c] = vload4(0, &weights[wt_base + c * K + k]);
        }

#pragma unroll
        for (int t = 0; t < 4; ++t) {
            int current_token_idx = token_idx_base + t;

            if (current_token_idx < M) {
                float4 in_val = vload4(0, &input[current_token_idx * K + k]);

#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    acc[t][c] = fma(in_val, w[c], acc[t][c]);
                }
            }
        }
    }

    float4 b0 = vload4(0, &bias[out_idx_base + 0]);
    float4 b1 = vload4(0, &bias[out_idx_base + 4]);

#pragma unroll
    for (int t = 0; t < 4; ++t) {
        int current_token_idx = token_idx_base + t;

        if (current_token_idx < M) {
            float sum[8];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                sum[c] = acc[t][c].x + acc[t][c].y + acc[t][c].z + acc[t][c].w;
            }

            float4 temp0 = (float4)(sum[0], sum[1], sum[2], sum[3]) + b0;
            float4 temp1 = (float4)(sum[4], sum[5], sum[6], sum[7]) + b1;

            float4 res0 = (float4)(gelu(temp0.x), gelu(temp0.y), gelu(temp0.z), gelu(temp0.w));
            float4 res1 = (float4)(gelu(temp1.x), gelu(temp1.y), gelu(temp1.z), gelu(temp1.w));

            __global float* out_ptr = &output[current_token_idx * N + out_idx_base];
            vstore4(res0, 0, out_ptr + 0);
            vstore4(res1, 0, out_ptr + 4);
        }
    }
}

__kernel void attn_score(
    __global const float* QKV,
    __global float* scores) {

    int i = get_global_id(0);
    int j = get_global_id(1);
    int z = get_global_id(2);

    int batch_idx = z / NUM_HEADS;
    int head_idx = z % NUM_HEADS;
    int j_start = j * 4;

    if (i >= TOKENS || j_start >= TOKENS || batch_idx >= BATCH_SIZE) return;

    int head_offset = head_idx * HEAD_DIM;
    int q_offset_base = (batch_idx * TOKENS + i) * QKV_DIM + head_offset;
    int k_chunk_base = (batch_idx * TOKENS) * QKV_DIM + EMBED_DIM + head_offset;

    bool has_1 = (j_start + 1 < TOKENS);
    bool has_2 = (j_start + 2 < TOKENS);
    bool has_3 = (j_start + 3 < TOKENS);

    int k_addr_0 = k_chunk_base + (j_start + 0) * QKV_DIM;

    int k_addr_1 = (has_1) ? (k_chunk_base + (j_start + 1) * QKV_DIM) : k_addr_0;
    int k_addr_2 = (has_2) ? (k_chunk_base + (j_start + 2) * QKV_DIM) : k_addr_0;
    int k_addr_3 = (has_3) ? (k_chunk_base + (j_start + 3) * QKV_DIM) : k_addr_0;;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (int d = 0; d < HEAD_DIM; d += 4) {
        float4 q_vec = vload4(0, &QKV[q_offset_base + d]);

        sum0 += dot(q_vec, vload4(0, &QKV[k_addr_0 + d]));

        if (has_1) sum1 += dot(q_vec, vload4(0, &QKV[k_addr_1 + d]));
        if (has_2) sum2 += dot(q_vec, vload4(0, &QKV[k_addr_2 + d]));
        if (has_3) sum3 += dot(q_vec, vload4(0, &QKV[k_addr_3 + d]));
    }

    float scale = 0.125f;
    int out_base = (batch_idx * NUM_HEADS + head_idx) * (TOKENS * TOKENS) + (i * TOKENS + j_start);

    scores[out_base + 0] = sum0 * scale;
    if (has_1) scores[out_base + 1] = sum1 * scale;
    if (has_2) scores[out_base + 2] = sum2 * scale;
    if (has_3) scores[out_base + 3] = sum3 * scale;
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

    int d_start = d * 4;
    int batch_idx = z / NUM_HEADS;
    int head_idx = z % NUM_HEADS;

    if (i >= TOKENS || d >= HEAD_DIM || batch_idx >= BATCH_SIZE) return;

    int score_base = (batch_idx * NUM_HEADS + head_idx) * (TOKENS * TOKENS) + i * TOKENS;
    int v_base_offset = 2 * EMBED_DIM + head_idx * HEAD_DIM + d_start;
    int qkv_batch_base = batch_idx * TOKENS * QKV_DIM;

    float4 sum = (float4)(0.0f);

    for (int j = 0; j < TOKENS; ++j) {
        float s = scores[score_base + j];

        int v_idx = qkv_batch_base + j * QKV_DIM + v_base_offset;
        float4 v = vload4(0, &QKV[v_idx]);

        sum += s * v;
    }

    int out_idx = (batch_idx * TOKENS + i) * EMBED_DIM + (head_idx * HEAD_DIM + d_start);
    vstore4(sum, 0, &attn_out[out_idx]);
}
__kernel void patch_embedding (
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

__kernel void layer_norm (
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

__kernel void add (
    __global const float* a,
    __global const float* b,
    __global float* output,
    const int size) {

    int i = get_global_id(0);
    if (i >= size) return;

    output[i] = a[i] + b[i];
}

// cls token + position embedding
__kernel void pos_embedding (
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