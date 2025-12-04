inline float4 gelu4(float4 x) {
    const float INV_SQRT_2 = 0.70710678f;

    const float p = 0.3275911f;
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

inline void load_weights (
    __global const float* weights,
    __local float tile_weights[LI_TILE][LI_STRIDE_WEIGHT],
    int k_curr, int K, int N,
    int g_out_group_start, int l_flat, int l_token_idx, int l_out_idx ) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int load_idx = l_flat * 2 + i;
        int w_r = load_idx & (LI_TILE - 1);
        int w_c = load_idx >> 4;

        if ((k_curr + w_r) < K && (g_out_group_start + w_c) < N) {
            tile_weights[w_r][w_c] = weights[(g_out_group_start + w_c) * K + (k_curr + w_r)];
        } else {
            tile_weights[w_r][w_c] = 0.0f;
        }
    }
}

inline void gemm (
    __local float tile_input[LI_LWS_TOKEN * LI_TPT][LI_STRIDE_IN],
    __local float tile_weights[LI_TILE][LI_STRIDE_WEIGHT],
    float acc[LI_TPT][LI_OPT],
    int l_token_idx, int l_out_idx) {

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
            for (int c = 0; c < LI_OPT; ++c) {
                acc[t][c] = fma(in_val, w_cache[c], acc[t][c]);
            }
        }
    }
}

inline void load_inputs (
    __global const float* input,
    __local float tile_input[LI_LWS_TOKEN * LI_TPT][LI_STRIDE_IN],
    int k_curr, int K, int M,
    int g_token_base, 
    int l_token_idx, 
    int l_out_idx ) {

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int l_row = l_token_idx * LI_TPT + t;
        int g_row = g_token_base + t;
        int k_offset = l_out_idx * 4;

        float4 val = (float4)0.0f;

        if (g_row < M && (k_curr + k_offset) < K) {
            val = vload4(0, &input[g_row * K + (k_curr + k_offset)]);
        }

        tile_input[l_row][k_offset + 0] = val.x;
        tile_input[l_row][k_offset + 1] = val.y;
        tile_input[l_row][k_offset + 2] = val.z;
        tile_input[l_row][k_offset + 3] = val.w;
    }
}

inline void store_result (
    __global float* output,
    __global const float* bias,
    float acc[LI_TPT][LI_OPT],
    int g_token_base, int g_out_base,
    int M, 
    int N,
    int gelu ) {
    if (g_out_base >= N || g_token_base >= M) return;

    float4 b0 = vload4(0, &bias[g_out_base + 0]);
    float4 b1 = vload4(0, &bias[g_out_base + 4]);

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int curr_g_token = g_token_base + t;
        if (curr_g_token < M) {
            float4 res0 = (float4)(acc[t][0], acc[t][1], acc[t][2], acc[t][3]) + b0;
            float4 res1 = (float4)(acc[t][4], acc[t][5], acc[t][6], acc[t][7]) + b1;

            if (gelu) {
                res0 = gelu4(res0);
                res1 = gelu4(res1);
            }

            __global float* out_ptr = &output[curr_g_token * N + g_out_base];
            vstore4(res0, 0, out_ptr + 0);
            vstore4(res1, 0, out_ptr + 4);
        }
    }
}

__kernel void linear_default(
    __global const float* input,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M, 
    const int K, 
    const int N ) {

    __local float tile_input[LI_LWS_TOKEN * LI_TPT][LI_STRIDE_IN];
    __local float tile_weights[LI_TILE][LI_STRIDE_WEIGHT];

    int l_out_idx = get_local_id(0);
    int l_token_idx = get_local_id(1);
    int g_out_base = (get_group_id(0) * LI_LWS_OUT + l_out_idx) * LI_OPT;
    int g_token_base = (get_group_id(1) * LI_LWS_TOKEN + l_token_idx) * LI_TPT;
    int g_out_group_start = get_group_id(0) * LI_LWS_OUT * LI_OPT;
    int l_flat = l_token_idx * LI_LWS_OUT + l_out_idx;

    float acc[LI_TPT][LI_OPT];
    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t)
        for (int c = 0; c < LI_OPT; ++c) acc[t][c] = 0.0f;

    for (int k_curr = 0; k_curr < K; k_curr += LI_TILE) {
        load_inputs(input, tile_input, k_curr, K, M, g_token_base, l_token_idx, l_out_idx);
        load_weights(weights, tile_weights, k_curr, K, N, g_out_group_start, l_flat, l_token_idx, l_out_idx);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        gemm(tile_input, tile_weights, acc, l_token_idx, l_out_idx);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_result(output, bias, acc, g_token_base, g_out_base, M, N, 0);
}

__kernel void linear_gelu (
    __global const float* input,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M, 
    const int K, 
    const int N ) {

    __local float tile_input[LI_LWS_TOKEN * LI_TPT][LI_STRIDE_IN];
    __local float tile_weights[LI_TILE][LI_STRIDE_WEIGHT];

    int l_out_idx = get_local_id(0);
    int l_token_idx = get_local_id(1);
    int g_out_base = (get_group_id(0) * LI_LWS_OUT + l_out_idx) * LI_OPT;
    int g_token_base = (get_group_id(1) * LI_LWS_TOKEN + l_token_idx) * LI_TPT;
    int g_out_group_start = get_group_id(0) * LI_LWS_OUT * LI_OPT;
    int l_flat = l_token_idx * LI_LWS_OUT + l_out_idx;

    float acc[LI_TPT][LI_OPT];
    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t)
        for (int c = 0; c < LI_OPT; ++c) acc[t][c] = 0.0f;

    for (int k_curr = 0; k_curr < K; k_curr += LI_TILE) {
        load_inputs(input, tile_input, k_curr, K, M, g_token_base, l_token_idx, l_out_idx);
        load_weights(weights, tile_weights, k_curr, K, N, g_out_group_start, l_flat, l_token_idx, l_out_idx);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        gemm(tile_input, tile_weights, acc, l_token_idx, l_out_idx);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_result(output, bias, acc, g_token_base, g_out_base, M, N, 1);
}

__kernel void linear_conv2d(
    __global const float* input_img,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M, 
    const int K, 
    const int N ) {

    __local float tile_input[LI_LWS_TOKEN * LI_TPT][LI_STRIDE_IN];
    __local float tile_weights[LI_TILE][LI_STRIDE_WEIGHT];

    int l_out_idx = get_local_id(0);
    int l_token_idx = get_local_id(1);
    int g_out_base = (get_group_id(0) * LI_LWS_OUT + l_out_idx) * LI_OPT;
    int g_token_base = (get_group_id(1) * LI_LWS_TOKEN + l_token_idx) * LI_TPT;
    int g_out_group_start = get_group_id(0) * LI_LWS_OUT * LI_OPT;
    int l_flat = l_token_idx * LI_LWS_OUT + l_out_idx;

    int patch_base_addr[LI_TPT];
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

            patch_base_addr[t] = (batch_idx * CHANNELS) * (IMG_SIZE * IMG_SIZE)
                + global_y_base * IMG_SIZE + global_x_base;
        }
    }

    float acc[LI_TPT][LI_OPT];
    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t)
        for (int c = 0; c < LI_OPT; ++c) acc[t][c] = 0.0f;

    for (int k_curr = 0; k_curr < K; k_curr += LI_TILE) {
        #pragma unroll
        for (int t = 0; t < LI_TPT; ++t) {
            int l_row = l_token_idx * LI_TPT + t;
            int g_row = g_token_base + t;
            int k_offset = l_out_idx * 4;
            int current_k = k_curr + k_offset;

            float4 val = (float4)(0.0f);

            if (g_row < M && current_k < K) {
                int ch = current_k / (PATCH_SIZE * PATCH_SIZE);
                int rem_k = current_k & ((PATCH_SIZE * PATCH_SIZE) - 1);
                int py = rem_k >> 4;
                int px = rem_k & (PATCH_SIZE - 1);
                int addr = patch_base_addr[t] + ch * (IMG_SIZE * IMG_SIZE) + py * IMG_SIZE + px;
                
                val = vload4(0, &input_img[addr]);
            }

            tile_input[l_row][k_offset + 0] = val.x;
            tile_input[l_row][k_offset + 1] = val.y;
            tile_input[l_row][k_offset + 2] = val.z;
            tile_input[l_row][k_offset + 3] = val.w;
        }

        load_weights(weights, tile_weights, k_curr, K, N, g_out_group_start, l_flat, l_token_idx, l_out_idx);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        gemm(tile_input, tile_weights, acc, l_token_idx, l_out_idx);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_result(output, bias, acc, g_token_base, g_out_base, M, N, 0);
}

inline void load_Q(
    __global const float* QKV,
    __local float tile_input[LI_LWS_TOKEN * LI_TPT][LI_STRIDE_IN],
    int k_curr,
    int batch_head_offset, 
    int g_token_base, 
    int l_token_idx, 
    int l_out_idx ) {

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int l_row = l_token_idx * LI_TPT + t;
        int g_row = g_token_base + t;
        int k_offset = l_out_idx * 4;

        float4 val = (float4)0.0f;
        
        if (g_row < TOKENS && (k_curr + k_offset) < HEAD_DIM) {
            int addr = batch_head_offset + (g_row * QKV_DIM) + (k_curr + k_offset);
            val = vload4(0, &QKV[addr]);
        }

        tile_input[l_row][k_offset + 0] = val.x;
        tile_input[l_row][k_offset + 1] = val.y;
        tile_input[l_row][k_offset + 2] = val.z;
        tile_input[l_row][k_offset + 3] = val.w;
    }
}

inline void load_K (
    __global const float* QKV,
    __local float tile_weights[LI_TILE][LI_STRIDE_WEIGHT],
    int k_curr,
    int batch_head_offset,
    int g_out_group_start, 
    int l_flat, int 
    l_token_idx, 
    int l_out_idx ) {

    int k_start_offset = batch_head_offset + EMBED_DIM; 

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int load_idx = l_flat * 2 + i;
        int w_r = load_idx & (LI_TILE - 1);
        int w_c = load_idx >> 4;

        int target_token = g_out_group_start + w_c;
        int target_dim = k_curr + w_r; 

        if (target_dim < HEAD_DIM && target_token < TOKENS) {
            int addr = k_start_offset + (target_token * QKV_DIM) + target_dim;
            
            tile_weights[w_r][w_c] = QKV[addr];
        } else {
            tile_weights[w_r][w_c] = 0.0f;
        }
    }
}

inline void store_score(
    __global float* scores,
    float acc[LI_TPT][LI_OPT],
    int g_token_base, 
    int g_out_base,
    int batch_head_idx ) {

    if (g_token_base >= TOKENS) return;

    int out_global_offset = batch_head_idx * (TOKENS * TOKENS);
    const float scale = 0.125f;

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int curr_g_token = g_token_base + t;
        
        if (curr_g_token < TOKENS) {
            float vals[8];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                vals[i] = acc[t][i] * scale;
            }

            int row_start = out_global_offset + curr_g_token * TOKENS;
            int col_idx = g_out_base;

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (col_idx + i < TOKENS) {
                    scores[row_start + col_idx + i] = vals[i];
                } else {
                    break;
                }
            }
        }
    }
}

__kernel void attn_score(
    __global const float* QKV,
    __global float* scores ) {
    __local float tile_input[LI_LWS_TOKEN * LI_TPT][LI_STRIDE_IN];
    __local float tile_weights[LI_TILE][LI_STRIDE_WEIGHT];

    int l_out_idx = get_local_id(0);
    int l_token_idx = get_local_id(1);
    
    int g_out_base = (get_group_id(0) * LI_LWS_OUT + l_out_idx) * LI_OPT; // Target Tokens
    int g_token_base = (get_group_id(1) * LI_LWS_TOKEN + l_token_idx) * LI_TPT; // Source Tokens
    
    int batch_head_idx = get_global_id(2); 
    int batch_idx = batch_head_idx / NUM_HEADS;
    int head_idx = batch_head_idx % NUM_HEADS;

    int batch_token_base = batch_idx * TOKENS * QKV_DIM;
    int head_offset = head_idx * HEAD_DIM;
    int cur_bh_offset = batch_token_base + head_offset;

    float acc[LI_TPT][LI_OPT];
    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t)
        for (int c = 0; c < LI_OPT; ++c) acc[t][c] = 0.0f;

    int g_out_group_start = get_group_id(0) * LI_LWS_OUT * LI_OPT;
    int l_flat = l_token_idx * LI_LWS_OUT + l_out_idx;

    for (int k_curr = 0; k_curr < HEAD_DIM; k_curr += LI_TILE) {
        load_Q(QKV, tile_input, k_curr, cur_bh_offset, g_token_base, l_token_idx, l_out_idx);
        load_K(QKV, tile_weights, k_curr, cur_bh_offset, g_out_group_start, l_flat, l_token_idx, l_out_idx);
        
        barrier(CLK_LOCAL_MEM_FENCE);

        gemm(tile_input, tile_weights, acc, l_token_idx, l_out_idx);
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_score(scores, acc, g_token_base, g_out_base, batch_head_idx);
}

__kernel void softmax(
    __global float* input,
    const int cols,
    const int total_rows
) {
    __local float sdata[256];

    int tid = get_local_id(0);
    int bid = get_group_id(0);

    if (bid >= total_rows) return;

    int row_offset = bid * cols;
    __global float* row_ptr = input + row_offset;

    float local_max = -INFINITY;

    for (int i = tid; i < cols; i += 256) {
        float val = row_ptr[i];
        local_max = fmax(local_max, val);
    }
    sdata[tid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = sdata[0];

    float local_sum = 0.0f;

    for (int i = tid; i < cols; i += 256) {
        float val = row_ptr[i];
        float exp_val = native_exp(val - row_max);
        row_ptr[i] = exp_val;
        local_sum += exp_val;
    }
    sdata[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_sum = sdata[0];
    float inv_sum = native_recip(row_sum);

    for (int i = tid; i < cols; i += 256) {
        row_ptr[i] *= inv_sum;
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