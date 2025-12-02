// kernel_sb.cl 전체 덮어쓰기

inline float4 gelu4(float4 x) {
    const float INV_SQRT_2 = 0.70710678f;
    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;

    float4 scaled_x = x * INV_SQRT_2;
    float4 abs_x = fabs(scaled_x);
    float4 sign_val = copysign((float4)(1.0f), scaled_x);
    float4 t = native_recip(1.0f + p * abs_x);
    float4 y = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1;
    float4 erf = sign_val * (1.0f - y * t * native_exp(-abs_x * abs_x));
    return 0.5f * x * (1.0f + erf);
}

// [핵심 변경] 타일 크기 32에 최적화된 Linear 커널
__kernel void linear_default(
    __global const float* restrict input,
    __global float* restrict output,
    __global const float* restrict weights,
    __global const float* restrict bias,
    const int M, 
    const int K, 
    const int N ) {

    // 타일 크기 32에 맞춘 로컬 메모리
    __local float tile_input[LI_LWS_TOKEN * LI_TPT][LI_STRIDE_IN];
    __local float tile_weights[LI_TILE][LI_STRIDE_WEIGHT];

    int l_out_idx = get_local_id(0);   // 0~3
    int l_token_idx = get_local_id(1); // 0~63
    
    int g_out_base = (get_group_id(0) * LI_LWS_OUT + l_out_idx) * LI_OPT;
    int g_token_base = (get_group_id(1) * LI_LWS_TOKEN + l_token_idx) * LI_TPT;
    
    // Weight 로딩을 위한 인덱스 계산 (Tile 32용)
    int l_flat = l_token_idx * LI_LWS_OUT + l_out_idx; // 0 ~ 255
    // 총 256개 스레드가 32x32(1024개) float를 로딩해야 함 -> 스레드당 4개(float4 1번)
    
    float acc[LI_TPT][LI_OPT];

    // 누적 레지스터 초기화
    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        #pragma unroll
        for (int c = 0; c < LI_OPT; ++c) acc[t][c] = 0.0f;
    }

    for (int k_curr = 0; k_curr < K; k_curr += LI_TILE) {
        
        // 1. Input 로딩 (Global -> Local)
        #pragma unroll
        for (int t = 0; t < LI_TPT; ++t) {
            int l_row = l_token_idx * LI_TPT + t;
            int g_row = g_token_base + t;
            int k_offset = l_out_idx * 4; // 0, 4, 8, 12...

            // 초기화 (패딩 효과)
            float4 val = (float4)(0.0f);
            
            // Tile 32에서는 LI_TPT(4) * 8(out) = 32 이므로 
            // l_out_idx(0~3) * 4 는 0, 4, 8, 12 ... 28 까지 커버 불가능?
            // 주의: LI_TILE이 32가 되면 Input 로딩 로직도 32개를 가져와야 함.
            // 현재 구조: 4개 스레드(l_out_idx)가 가로로 배치됨. 4 * 4(float4) = 16.
            // Tile 32를 채우려면 2번 돌아야 함.
            
            // [LOOP UNROLL 2] Input Loading Expansion for Tile 32
            #pragma unroll
            for(int step = 0; step < 2; step++) { // 16 -> 32 확장을 위해 2번 반복
                int k_off_step = k_offset + (step * 16); 
                
                if (g_row < M && (k_curr + k_off_step) < K) {
                   val = vload4(0, &input[g_row * K + (k_curr + k_off_step)]);
                } else {
                   val = (float4)(0.0f);
                }
                vstore4(val, 0, &tile_input[l_row][k_off_step]);
            }
        }

        // 2. Weight 로딩 (Global -> Local) [Vectorized & Optimized for Tile 32]
        // Weights는 N x K 매트릭스. Tile은 [32(K) x 32(N)] 영역 로딩 필요 (Transposed layout in local)
        // 256 threads loading 32x32(1024) elements. Each thread loads 4 floats.
        // float4로 한 번에 로딩하여 대역폭 효율 극대화
        
        int g_out_group_start = get_group_id(0) * LI_LWS_OUT * LI_OPT;
        
        int load_idx = l_flat * 4; // 각 스레드가 4개씩 책임짐
        int w_r = load_idx & 31;   // % 32 (Tile Size 32)
        int w_c = load_idx >> 5;   // / 32
        
        // Weights: [N][K] layout assuming Row-Major storage where K is inner dimension?
        // 기존 코드: weights[(g_out + w_c) * K + (k_curr + w_r)]
        // w_r이 0~31로 연속적이므로 vload4 가능!
        
        float4 w_val = (float4)(0.0f);
        if ((g_out_group_start + w_c) < N && (k_curr + w_r) < K) {
             // K가 4의 배수라고 가정하고 vload4 사용 (대부분의 ViT 차원은 4의 배수)
             // w_r은 4의 배수로 정렬됨 (l_flat * 4 이므로)
             w_val = vload4(0, &weights[(g_out_group_start + w_c) * K + (k_curr + w_r)]);
        }
        
        // Local Memory에 Transposed 형태로 저장하거나 그대로 저장
        // 기존 로직: tile_weights[w_r][w_c].
        // w_val은 [w_r, w_r+1, w_r+2, w_r+3] 값을 가짐.
        tile_weights[w_r + 0][w_c] = w_val.x;
        tile_weights[w_r + 1][w_c] = w_val.y;
        tile_weights[w_r + 2][w_c] = w_val.z;
        tile_weights[w_r + 3][w_c] = w_val.w;

        barrier(CLK_LOCAL_MEM_FENCE);

        // 3. 계산 (Compute)
        #pragma unroll 32
        for (int k = 0; k < LI_TILE; ++k) {
            float w_cache[LI_OPT];
            int l_col_base = l_out_idx * LI_OPT;

            #pragma unroll
            for (int c = 0; c < LI_OPT; ++c) {
                w_cache[c] = tile_weights[k][l_col_base + c];
            }

            #pragma unroll
            for (int t = 0; t < LI_TPT; ++t) {
                int l_row = l_token_idx * LI_TPT + t;
                float in_val = tile_input[l_row][k];

                #pragma unroll
                for (int c = 0; c < LI_OPT; ++c) {
                    acc[t][c] = fma(in_val, w_cache[c], acc[t][c]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (g_out_base >= N || g_token_base >= M) return;

    // Bias & Store
    float4 b0 = (float4)(0.0f);
    float4 b1 = (float4)(0.0f);
    if (g_out_base + 4 <= N) b0 = vload4(0, &bias[g_out_base + 0]);
    if (g_out_base + 8 <= N) b1 = vload4(0, &bias[g_out_base + 4]);

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int curr_g_token = g_token_base + t;
        if (curr_g_token < M) {
            float4 res0 = (float4)(acc[t][0], acc[t][1], acc[t][2], acc[t][3]) + b0;
            float4 res1 = (float4)(acc[t][4], acc[t][5], acc[t][6], acc[t][7]) + b1;

            __global float* out_ptr = &output[curr_g_token * N + g_out_base];
            if (g_out_base + 4 <= N) vstore4(res0, 0, out_ptr + 0);
            if (g_out_base + 8 <= N) vstore4(res1, 0, out_ptr + 4);
        }
    }
}

// GELU 버전 (위 코드 복사 + gelu4 적용)
__kernel void linear_gelu(
    __global const float* restrict input,
    __global float* restrict output,
    __global const float* restrict weights,
    __global const float* restrict bias,
    const int M, 
    const int K, 
    const int N) {

    __local float tile_input[LI_LWS_TOKEN * LI_TPT][LI_STRIDE_IN];
    __local float tile_weights[LI_TILE][LI_STRIDE_WEIGHT];

    int l_out_idx = get_local_id(0);
    int l_token_idx = get_local_id(1);
    
    int g_out_base = (get_group_id(0) * LI_LWS_OUT + l_out_idx) * LI_OPT;
    int g_token_base = (get_group_id(1) * LI_LWS_TOKEN + l_token_idx) * LI_TPT;
    
    int l_flat = l_token_idx * LI_LWS_OUT + l_out_idx;
    float acc[LI_TPT][LI_OPT];

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        #pragma unroll
        for (int c = 0; c < LI_OPT; ++c) acc[t][c] = 0.0f;
    }

    for (int k_curr = 0; k_curr < K; k_curr += LI_TILE) {
        #pragma unroll
        for (int t = 0; t < LI_TPT; ++t) {
            int l_row = l_token_idx * LI_TPT + t;
            int g_row = g_token_base + t;
            int k_offset = l_out_idx * 4; 

            #pragma unroll
            for(int step = 0; step < 2; step++) {
                int k_off_step = k_offset + (step * 16); 
                float4 val = (float4)(0.0f);
                if (g_row < M && (k_curr + k_off_step) < K) {
                   val = vload4(0, &input[g_row * K + (k_curr + k_off_step)]);
                }
                vstore4(val, 0, &tile_input[l_row][k_off_step]);
            }
        }

        int g_out_group_start = get_group_id(0) * LI_LWS_OUT * LI_OPT;
        int load_idx = l_flat * 4; 
        int w_r = load_idx & 31;   
        int w_c = load_idx >> 5;   
        
        float4 w_val = (float4)(0.0f);
        if ((g_out_group_start + w_c) < N && (k_curr + w_r) < K) {
             w_val = vload4(0, &weights[(g_out_group_start + w_c) * K + (k_curr + w_r)]);
        }
        
        tile_weights[w_r + 0][w_c] = w_val.x;
        tile_weights[w_r + 1][w_c] = w_val.y;
        tile_weights[w_r + 2][w_c] = w_val.z;
        tile_weights[w_r + 3][w_c] = w_val.w;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll 32
        for (int k = 0; k < LI_TILE; ++k) {
            float w_cache[LI_OPT];
            int l_col_base = l_out_idx * LI_OPT;
            #pragma unroll
            for (int c = 0; c < LI_OPT; ++c) w_cache[c] = tile_weights[k][l_col_base + c];
            #pragma unroll
            for (int t = 0; t < LI_TPT; ++t) {
                int l_row = l_token_idx * LI_TPT + t;
                float in_val = tile_input[l_row][k];
                #pragma unroll
                for (int c = 0; c < LI_OPT; ++c) acc[t][c] = fma(in_val, w_cache[c], acc[t][c]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (g_out_base >= N || g_token_base >= M) return;

    float4 b0 = (float4)(0.0f);
    float4 b1 = (float4)(0.0f);
    if (g_out_base + 4 <= N) b0 = vload4(0, &bias[g_out_base + 0]);
    if (g_out_base + 8 <= N) b1 = vload4(0, &bias[g_out_base + 4]);

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int curr_g_token = g_token_base + t;
        if (curr_g_token < M) {
            float4 res0 = (float4)(acc[t][0], acc[t][1], acc[t][2], acc[t][3]) + b0;
            float4 res1 = (float4)(acc[t][4], acc[t][5], acc[t][6], acc[t][7]) + b1;

            // [GELU]
            res0 = gelu4(res0);
            res1 = gelu4(res1);

            __global float* out_ptr = &output[curr_g_token * N + g_out_base];
            if (g_out_base + 4 <= N) vstore4(res0, 0, out_ptr + 0);
            if (g_out_base + 8 <= N) vstore4(res1, 0, out_ptr + 4);
        }
    }
}

// Conv2d도 Tile 32에 맞춰 비트마스킹 수정 (>>4 -> >>5, &15 -> &31)
__kernel void linear_conv2d(
    __global const float* restrict input_img,
    __global float* restrict output,
    __global const float* restrict weights,
    __global const float* restrict bias,
    const int M, 
    const int K, 
    const int N) {

    __local float tile_input[LI_LWS_TOKEN * LI_TPT][LI_STRIDE_IN];
    __local float tile_weights[LI_TILE][LI_STRIDE_WEIGHT];

    int l_out_idx = get_local_id(0);
    int l_token_idx = get_local_id(1);
    
    int g_out_base = (get_group_id(0) * LI_LWS_OUT + l_out_idx) * LI_OPT;
    int g_token_base = (get_group_id(1) * LI_LWS_TOKEN + l_token_idx) * LI_TPT;
    int l_flat = l_token_idx * LI_LWS_OUT + l_out_idx;

    float acc[LI_TPT][LI_OPT];
    int patch_base_addr[LI_TPT];
    
    // Address Calculation
    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int g_patch_idx = g_token_base + t;
        if (g_patch_idx < M) {
            int batch_idx = g_patch_idx / (OUTPUT_SIZE * OUTPUT_SIZE);
            int idx_in_batch = g_patch_idx % (OUTPUT_SIZE * OUTPUT_SIZE);
            int patch_y = idx_in_batch / OUTPUT_SIZE;
            int patch_x = idx_in_batch % OUTPUT_SIZE;
            int global_y_base = (patch_y << 4); 
            int global_x_base = (patch_x << 4);
            
            patch_base_addr[t] = (batch_idx * 3) * (IMG_SIZE * IMG_SIZE) 
                                 + global_y_base * IMG_SIZE + global_x_base;
        }
    }

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        #pragma unroll
        for (int c = 0; c < LI_OPT; ++c) acc[t][c] = 0.0f;
    }

    for (int k_curr = 0; k_curr < K; k_curr += LI_TILE) {
        
        // Input Loading (Patch Logic) - Modified for Tile 32
        #pragma unroll
        for (int t = 0; t < LI_TPT; ++t) {
            int l_row = l_token_idx * LI_TPT + t;
            int g_row = g_token_base + t;
            int k_offset = l_out_idx * 4;

            // Tile 32 Expansion
            #pragma unroll
            for(int step = 0; step < 2; step++) {
                int k_off_step = k_offset + (step * 16); 
                int current_k = k_curr + k_off_step;
                float4 val = (float4)(0.0f);
                
                if (g_row < M && current_k < K) {
                    int ch = current_k / (PATCH_SIZE * PATCH_SIZE);
                    int rem_k = current_k & ((PATCH_SIZE * PATCH_SIZE) - 1);
                    int py = rem_k >> 4;
                    int px = rem_k & (PATCH_SIZE - 1); 

                    int addr = patch_base_addr[t] 
                               + ch * (IMG_SIZE * IMG_SIZE) 
                               + py * IMG_SIZE + px;
                    val = vload4(0, &input_img[addr]);
                }
                vstore4(val, 0, &tile_input[l_row][k_off_step]);
            }
        }

        // Weight Loading (Vectorized)
        int g_out_group_start = get_group_id(0) * LI_LWS_OUT * LI_OPT;
        int load_idx = l_flat * 4;
        int w_r = load_idx & 31; // Tile 32
        int w_c = load_idx >> 5;

        float4 w_val = (float4)(0.0f);
        if ((g_out_group_start + w_c) < N && (k_curr + w_r) < K) {
             w_val = vload4(0, &weights[(g_out_group_start + w_c) * K + (k_curr + w_r)]);
        }
        tile_weights[w_r + 0][w_c] = w_val.x;
        tile_weights[w_r + 1][w_c] = w_val.y;
        tile_weights[w_r + 2][w_c] = w_val.z;
        tile_weights[w_r + 3][w_c] = w_val.w;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute
        #pragma unroll 32
        for (int k = 0; k < LI_TILE; ++k) {
            float w_cache[LI_OPT];
            int l_col_base = l_out_idx * LI_OPT;
            #pragma unroll
            for (int c = 0; c < LI_OPT; ++c) w_cache[c] = tile_weights[k][l_col_base + c];
            #pragma unroll
            for (int t = 0; t < LI_TPT; ++t) {
                int l_row = l_token_idx * LI_TPT + t;
                float in_val = tile_input[l_row][k];
                #pragma unroll
                for (int c = 0; c < LI_OPT; ++c) acc[t][c] = fma(in_val, w_cache[c], acc[t][c]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (g_out_base >= N || g_token_base >= M) return;

    float4 b0 = (float4)(0.0f);
    float4 b1 = (float4)(0.0f);
    if (g_out_base + 4 <= N) b0 = vload4(0, &bias[g_out_base + 0]);
    if (g_out_base + 8 <= N) b1 = vload4(0, &bias[g_out_base + 4]);

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int curr_g_token = g_token_base + t;
        if (curr_g_token < M) {
            float4 res0 = (float4)(acc[t][0], acc[t][1], acc[t][2], acc[t][3]) + b0;
            float4 res1 = (float4)(acc[t][4], acc[t][5], acc[t][6], acc[t][7]) + b1;

            __global float* out_ptr = &output[curr_g_token * N + g_out_base];
            if (g_out_base + 4 <= N) vstore4(res0, 0, out_ptr + 0);
            if (g_out_base + 8 <= N) vstore4(res1, 0, out_ptr + 4);
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