inline float4 gelu4(float4 x) {
    return 0.5f * x * (1.0f + erf(x / sqrt(2.0f)));
}

inline void load_weights (
    __global const float* weights,
    __local float local_weights[LI_TILE][LI_STRIDE_WEIGHT],
    int k_curr, int K, int N,
    int g_out_group_start, 
    int l_flat, 
    int l_token_idx, 
    int l_out_idx ) {

    int row_in_tile = l_flat >> 2;
    int col_chunk_idx = l_flat & 3;
    int col_in_tile = col_chunk_idx << 2;

    int global_row = g_out_group_start + row_in_tile;
    int global_col = k_curr + col_in_tile;

    float4 val = vload4(0, &weights[global_row * K + global_col]);

    local_weights[col_in_tile + 0][row_in_tile] = val.x;
    local_weights[col_in_tile + 1][row_in_tile] = val.y;
    local_weights[col_in_tile + 2][row_in_tile] = val.z;
    local_weights[col_in_tile + 3][row_in_tile] = val.w;
}

inline void gemm (
    __local float local_input[LI_STRIDE_IN][LI_LWS_TOKEN * LI_TPT + 4],
    __local float local_weights[LI_TILE][LI_STRIDE_WEIGHT],
    float acc[LI_TPT][LI_OPT],
    int l_token_idx, int l_out_idx) {

    float4* acc_vec_ptr;

    for (int k = 0; k < LI_TILE; ++k) {
        __local float* w_ptr = &local_weights[k][l_out_idx * LI_OPT];
        
        float4 w0 = vload4(0, w_ptr);
        float4 w1 = vload4(1, w_ptr);
        float4 w2 = vload4(2, w_ptr);
        float4 w3 = vload4(3, w_ptr);

        float4 in_vals = vload4(0, &local_input[k][l_token_idx * LI_TPT]); 
        
        acc_vec_ptr = (float4*)&acc[0][0];
        acc_vec_ptr[0] = fma((float4)(in_vals.x), w0, acc_vec_ptr[0]);
        acc_vec_ptr[1] = fma((float4)(in_vals.x), w1, acc_vec_ptr[1]);
        acc_vec_ptr[2] = fma((float4)(in_vals.x), w2, acc_vec_ptr[2]);
        acc_vec_ptr[3] = fma((float4)(in_vals.x), w3, acc_vec_ptr[3]);

        acc_vec_ptr = (float4*)&acc[1][0];
        acc_vec_ptr[0] = fma((float4)(in_vals.y), w0, acc_vec_ptr[0]);
        acc_vec_ptr[1] = fma((float4)(in_vals.y), w1, acc_vec_ptr[1]);
        acc_vec_ptr[2] = fma((float4)(in_vals.y), w2, acc_vec_ptr[2]);
        acc_vec_ptr[3] = fma((float4)(in_vals.y), w3, acc_vec_ptr[3]);

        acc_vec_ptr = (float4*)&acc[2][0];
        acc_vec_ptr[0] = fma((float4)(in_vals.z), w0, acc_vec_ptr[0]);
        acc_vec_ptr[1] = fma((float4)(in_vals.z), w1, acc_vec_ptr[1]);
        acc_vec_ptr[2] = fma((float4)(in_vals.z), w2, acc_vec_ptr[2]);
        acc_vec_ptr[3] = fma((float4)(in_vals.z), w3, acc_vec_ptr[3]);

        acc_vec_ptr = (float4*)&acc[3][0];
        acc_vec_ptr[0] = fma((float4)(in_vals.w), w0, acc_vec_ptr[0]);
        acc_vec_ptr[1] = fma((float4)(in_vals.w), w1, acc_vec_ptr[1]);
        acc_vec_ptr[2] = fma((float4)(in_vals.w), w2, acc_vec_ptr[2]);
        acc_vec_ptr[3] = fma((float4)(in_vals.w), w3, acc_vec_ptr[3]);
    }
}

inline void load_inputs (
    __global const float* input,
    __local float local_input[LI_STRIDE_IN][LI_LWS_TOKEN * LI_TPT + 4],
    int k_curr, int K, int M,
    int g_token_base, 
    int l_token_idx, 
    int l_out_idx ) {

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int l_row = l_token_idx * LI_TPT + t;
        int g_row = g_token_base + t;
        int k_offset = l_out_idx * 4;

        float4 val = vload4(0, &input[g_row * K + (k_curr + k_offset)]);

        local_input[k_offset + 0][l_row] = val.x;
        local_input[k_offset + 1][l_row] = val.y;
        local_input[k_offset + 2][l_row] = val.z;
        local_input[k_offset + 3][l_row] = val.w;
    }
}

inline void store_linear (
    __global float* output,
    __global const float* bias,
    float acc[LI_TPT][LI_OPT],
    int g_token_base, int g_out_base,
    int M, 
    int N,
    int gelu ) {

    float4 b0 = vload4(0, &bias[g_out_base + 0]);
    float4 b1 = vload4(0, &bias[g_out_base + 4]);
    float4 b2 = vload4(0, &bias[g_out_base + 8]);
    float4 b3 = vload4(0, &bias[g_out_base + 12]);

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int curr_g_token = g_token_base + t;
        if (curr_g_token < M) {
            float4 res0 = (float4)(acc[t][0], acc[t][1], acc[t][2], acc[t][3]) + b0;
            float4 res1 = (float4)(acc[t][4], acc[t][5], acc[t][6], acc[t][7]) + b1;
            float4 res2 = (float4)(acc[t][8], acc[t][9], acc[t][10], acc[t][11]) + b2;
            float4 res3 = (float4)(acc[t][12], acc[t][13], acc[t][14], acc[t][15]) + b3;

            if (gelu) {
                res0 = gelu4(res0);
                res1 = gelu4(res1);
                res2 = gelu4(res2);
                res3 = gelu4(res3);
            }

            __global float* out_ptr = &output[curr_g_token * N + g_out_base];
            vstore4(res0, 0, out_ptr + 0);
            vstore4(res1, 0, out_ptr + 4);
            vstore4(res2, 0, out_ptr + 8);
            vstore4(res3, 0, out_ptr + 12);
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

    __local float local_input[LI_STRIDE_IN][LI_LWS_TOKEN * LI_TPT];
    __local float local_weights[LI_TILE][LI_STRIDE_WEIGHT];

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
        load_inputs(input, local_input, k_curr, K, M, g_token_base, l_token_idx, l_out_idx);
        load_weights(weights, local_weights, k_curr, K, N, g_out_group_start, l_flat, l_token_idx, l_out_idx);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        gemm(local_input, local_weights, acc, l_token_idx, l_out_idx);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_linear(output, bias, acc, g_token_base, g_out_base, M, N, 0);
}

__kernel void linear_gelu (
    __global const float* input,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M, 
    const int K, 
    const int N ) {

    __local float local_input[LI_STRIDE_IN][LI_LWS_TOKEN * LI_TPT + 4];
    __local float local_weights[LI_TILE][LI_STRIDE_WEIGHT];

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
        load_inputs(input, local_input, k_curr, K, M, g_token_base, l_token_idx, l_out_idx);
        load_weights(weights, local_weights, k_curr, K, N, g_out_group_start, l_flat, l_token_idx, l_out_idx);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        gemm(local_input, local_weights, acc, l_token_idx, l_out_idx);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_linear(output, bias, acc, g_token_base, g_out_base, M, N, 1);
}

__kernel void linear_conv2d(
    __global const float* input_img,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M, 
    const int K, 
    const int N ) {

    __local float local_input[LI_STRIDE_IN][LI_LWS_TOKEN * LI_TPT + 4];
    __local float local_weights[LI_TILE][LI_STRIDE_WEIGHT];

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

            local_input[k_offset + 0][l_row] = val.x;
            local_input[k_offset + 1][l_row] = val.y;
            local_input[k_offset + 2][l_row] = val.z;
            local_input[k_offset + 3][l_row] = val.w;
        }

        load_weights(weights, local_weights, k_curr, K, N, g_out_group_start, l_flat, l_token_idx, l_out_idx);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        gemm(local_input, local_weights, acc, l_token_idx, l_out_idx);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_linear(output, bias, acc, g_token_base, g_out_base, M, N, 0);
}

inline void load_Q(
    __global const float* QKV,
    __local float local_input[LI_STRIDE_IN][LI_LWS_TOKEN * LI_TPT + 4],
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

        int addr = batch_head_offset + (g_row * QKV_DIM) + (k_curr + k_offset);
        float4 val = vload4(0, &QKV[addr]);

        local_input[k_offset + 0][l_row] = val.x;
        local_input[k_offset + 1][l_row] = val.y;
        local_input[k_offset + 2][l_row] = val.z;
        local_input[k_offset + 3][l_row] = val.w;
    }
}

inline void load_K (
    __global const float* QKV,
    __local float local_weights[LI_TILE][LI_STRIDE_WEIGHT],
    int k_curr,
    int batch_head_offset,
    int g_out_group_start, 
    int l_flat, int 
    l_token_idx, 
    int l_out_idx ) {

    int k_start_offset = batch_head_offset + EMBED_DIM; 

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int load_idx = l_flat * 4 + i;
        int w_r = load_idx & (LI_TILE - 1);
        int w_c = load_idx >> 4;

        int target_token = g_out_group_start + w_c;
        int target_dim = k_curr + w_r; 

        int addr = k_start_offset + (target_token * QKV_DIM) + target_dim;
        local_weights[w_r][w_c] = QKV[addr];
    }
}

inline void store_score (
    __global float* scores,
    float acc[LI_TPT][LI_OPT],
    int g_token_base, 
    int g_out_base,
    int batch_head_idx ) {

    int out_global_offset = batch_head_idx * (TOKENS * TOKENS);
    const float scale = 0.125f;

    #pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int curr_g_token = g_token_base + t;
        
        if (curr_g_token < TOKENS) {
            float vals[LI_OPT];

            #pragma unroll
            for (int i = 0; i < LI_OPT; i++) {
                vals[i] = acc[t][i] * scale;
            }

            int row_start = out_global_offset + curr_g_token * TOKENS;
            int col_idx = g_out_base;

            #pragma unroll
            for (int i = 0; i < LI_OPT; ++i) {
                if (col_idx + i < TOKENS) {
                    scores[row_start + col_idx + i] = vals[i];
                }
            }
        }
    }
}

__kernel void attn_score(
    __global const float* QKV,
    __global float* scores ) {
    __local float local_input[LI_STRIDE_IN][LI_LWS_TOKEN * LI_TPT + 4];
    __local float local_weights[LI_TILE][LI_STRIDE_WEIGHT];

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
        load_Q(QKV, local_input, k_curr, cur_bh_offset, g_token_base, l_token_idx, l_out_idx);
        load_K(QKV, local_weights, k_curr, cur_bh_offset, g_out_group_start, l_flat, l_token_idx, l_out_idx);
        
        barrier(CLK_LOCAL_MEM_FENCE);

        gemm(local_input, local_weights, acc, l_token_idx, l_out_idx);
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_score(scores, acc, g_token_base, g_out_base, batch_head_idx);
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
    __global float* attn_out
) {
    const int g_row_block = get_global_id(0);
    const int g_dim_idx = get_global_id(1);
    const int g_batch_head_idx = get_global_id(2);

    const int row_base = g_row_block << 2;

    if (row_base >= TOKENS) return;

    const int batch_idx = g_batch_head_idx / NUM_HEADS;
    const int head_idx = g_batch_head_idx % NUM_HEADS;

    const int d_offset = g_dim_idx << 2;
    const int head_dim_offset = head_idx * HEAD_DIM;
    const int batch_offset = batch_idx * TOKENS;

    const int score_head_offset = g_batch_head_idx * TOKENS * TOKENS;
    const __global float* s_ptr_base = scores + score_head_offset + row_base * TOKENS;

    const int qkv_base = batch_offset * QKV_DIM;
    const int v_offset = (EMBED_DIM << 1) + head_dim_offset + d_offset;
    const __global float* v_ptr = QKV + qkv_base + v_offset;

    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);

    const __global float* s_ptr0 = s_ptr_base;
    const __global float* s_ptr1 = s_ptr_base + TOKENS;
    const __global float* s_ptr2 = s_ptr_base + (TOKENS << 1);
    const __global float* s_ptr3 = s_ptr_base + (TOKENS * 3);

    const bool r1_valid = (row_base + 1 < TOKENS);
    const bool r2_valid = (row_base + 2 < TOKENS);
    const bool r3_valid = (row_base + 3 < TOKENS);

    for (int j = 0; j < TOKENS; ++j) {
        float4 v_val = vload4(0, v_ptr);
        v_ptr += QKV_DIM;

        float s0 = *s_ptr0++;
        acc0 = fma(v_val, (float4)(s0), acc0);

        if (r1_valid) {
            float s1 = *s_ptr1++;
            acc1 = fma(v_val, (float4)(s1), acc1);
        }
        if (r2_valid) {
            float s2 = *s_ptr2++;
            acc2 = fma(v_val, (float4)(s2), acc2);
        }
        if (r3_valid) {
            float s3 = *s_ptr3++;
            acc3 = fma(v_val, (float4)(s3), acc3);
        }
    }

    const int out_base = (batch_offset + row_base) * EMBED_DIM + head_dim_offset + d_offset;
    __global float* out_ptr = attn_out + out_base;

    vstore4(acc0, 0, out_ptr);
    if (r1_valid) vstore4(acc1, 0, out_ptr + EMBED_DIM);
    if (r2_valid) vstore4(acc2, 0, out_ptr + (EMBED_DIM << 1));
    if (r3_valid) vstore4(acc3, 0, out_ptr + (EMBED_DIM * 3));
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