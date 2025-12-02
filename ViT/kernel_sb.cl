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

inline int get_patch_index(int g_patch_idx, int k) {
    int batch_idx = g_patch_idx / (OUTPUT_SIZE * OUTPUT_SIZE);
    int idx_in_batch = g_patch_idx % (OUTPUT_SIZE * OUTPUT_SIZE);
    
    int patch_y = idx_in_batch / OUTPUT_SIZE;
    int patch_x = idx_in_batch % OUTPUT_SIZE;

    int ch = k / (PATCH_SIZE * PATCH_SIZE);
    int rem_k = k & ((PATCH_SIZE * PATCH_SIZE) - 1); 
    
    int py = rem_k >> 4;
    int px = rem_k & (PATCH_SIZE - 1);

    int gy = (patch_y << 4) + py;
    int gx = (patch_x << 4) + px;
    
    return (batch_idx * 3 + ch) * (IMG_SIZE * IMG_SIZE) + gy * IMG_SIZE + gx;
}

inline void linear_layer(
    __global const float* input,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M, const int K, const int N,
    __local float* tile_input_ptr,
    __local float* tile_weights_ptr,
    const int PATCH,
    const int GELU ) {

    int l_out_idx = get_local_id(0);
    int l_token_idx = get_local_id(1);
    int l_flat_idx = l_token_idx * LI_LWS_OUT + l_out_idx;

    int g_out_base = (get_group_id(0) * LI_LWS_OUT + l_out_idx) * LI_OPT;
    int g_token_base = (get_group_id(1) * LI_LWS_TOKEN + l_token_idx) * LI_TPT;

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
            int current_k = k_curr + k_offset;

            int l_idx = l_row * LI_STRIDE_IN + k_offset; 

            if (g_row < M && current_k < K) {
                int addr;

                if (PATCH) {
                    addr = get_patch_index(g_row, current_k);
                } else {
                    addr = g_row * K + current_k;
                }
                
                float4 val = vload4(0, &input[addr]);
                
                tile_input_ptr[l_idx + 0] = val.x;
                tile_input_ptr[l_idx + 1] = val.y;
                tile_input_ptr[l_idx + 2] = val.z;
                tile_input_ptr[l_idx + 3] = val.w;
            } else {
                tile_input_ptr[l_idx + 0] = 0.0f;
                tile_input_ptr[l_idx + 1] = 0.0f;
                tile_input_ptr[l_idx + 2] = 0.0f;
                tile_input_ptr[l_idx + 3] = 0.0f;
            }
        }

        int g_out_group_start = get_group_id(0) * LI_LWS_OUT * LI_OPT;
        int tile_width_n = LI_LWS_OUT * LI_OPT;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int load_idx = l_flat_idx * 2 + i;
            int w_row_k = load_idx / tile_width_n;
            int w_col_n = load_idx % tile_width_n;
            
            int l_idx = w_row_k * LI_STRIDE_WEIGHT + w_col_n;

            if ((k_curr + w_row_k) < K && (g_out_group_start + w_col_n) < N) {
                tile_weights_ptr[l_idx] = weights[(g_out_group_start + w_col_n) * K + (k_curr + w_row_k)];
            } else {
                tile_weights_ptr[l_idx] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < LI_TILE; ++k) {
            float w_cache[LI_OPT];
            int l_col_base = l_out_idx * LI_OPT;
            
            #pragma unroll
            for (int c = 0; c < LI_OPT; ++c) {
                // tile_weights[k][l_col_base + c]
                w_cache[c] = tile_weights_ptr[k * LI_STRIDE_WEIGHT + (l_col_base + c)];
            }

            #pragma unroll
            for (int t = 0; t < LI_TPT; ++t) {
                int l_row = l_token_idx * LI_TPT + t;
                // tile_input[l_row][k]
                float in_val = tile_input_ptr[l_row * LI_STRIDE_IN + k];

                #pragma unroll
                for (int c = 0; c < LI_OPT; ++c) {
                    acc[t][c] = fma(in_val, w_cache[c], acc[t][c]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (g_out_base >= N || g_token_base >= M) return;

    float4 b0 = vload4(0, &bias[g_out_base + 0]);
    float4 b1 = vload4(0, &bias[g_out_base + 4]);

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int curr_g_token = g_token_base + t;
        if (curr_g_token < M) {
            float4 res0 = (float4)(acc[t][0], acc[t][1], acc[t][2], acc[t][3]) + b0;
            float4 res1 = (float4)(acc[t][4], acc[t][5], acc[t][6], acc[t][7]) + b1;

            if (GELU) {
                res0 = gelu4(res0);
                res1 = gelu4(res1);
            }

            __global float* out_ptr = &output[curr_g_token * N + g_out_base];
            vstore4(res0, 0, out_ptr + 0);
            vstore4(res1, 0, out_ptr + 4);
        }
    }
}

__kernel void linear_default (
    __global const float* input, 
    __global float* output, 
    __global const float* weights, 
    __global const float* bias,
    const int M, 
    const int K, 
    const int N) {

    __local float l_in[LI_LWS_TOKEN * LI_TPT * LI_STRIDE_IN];
    __local float l_w[LI_TILE * LI_STRIDE_WEIGHT];
    
    linear_layer(input, output, weights, bias, M, K, N, l_in, l_w, 0, 0);
}

__kernel void linear_gelu (
    __global const float* input,
    __global float* output, 
    __global const float* weights, 
    __global const float* bias,
    const int M,
    const int K, 
    const int N) {

    __local float l_in[LI_LWS_TOKEN * LI_TPT * LI_STRIDE_IN];
    __local float l_w[LI_TILE * LI_STRIDE_WEIGHT];
    
    linear_layer(input, output, weights, bias, M, K, N, l_in, l_w, 0, 1);
}

__kernel void linear_conv2d (
    __global const float* input, 
    __global float* output, 
    __global const float* weights, 
    __global const float* bias,
    const int M, 
    const int K, 
    const int N) {

    __local float l_in[LI_LWS_TOKEN * LI_TPT * LI_STRIDE_IN];
    __local float l_w[LI_TILE * LI_STRIDE_WEIGHT];
    
    linear_layer(input, output, weights, bias, M, K, N, l_in, l_w, 1, 0);
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