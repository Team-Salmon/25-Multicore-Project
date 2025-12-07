#define LI_INPUT_STRIDE (LI_LWS_TOKEN * LI_TPT + 4)

inline float4 gelu4(float4 x) {
    return 0.5f * x * (1.0f + erf(x / sqrt(2.0f)));
}

inline void load_weights (
    __global const float* weights,
    __local float l_weights[LI_STRIDE_IN][LI_STRIDE_WEIGHT],
    int K, 
    int N,
    int g_row_base,
    int g_col_base,
    int l_row,
    int l_col ) {

    l_col <<= 2; // float4 load

    #pragma unroll
    for (int loop = 0; loop < 2; loop ++) {
        int col = l_col + (loop << 4);
        int g_col = g_col_base + col;

        int row = l_row; 
        int g_row = g_row_base + row;

        float4 w = (float4)0.0f;
        if (g_row < N && g_col < K) {
            w = vload4(0, &weights[g_row * K + g_col]);
        }

        l_weights[col + 0][row] = w.x;
        l_weights[col + 1][row] = w.y;
        l_weights[col + 2][row] = w.z;
        l_weights[col + 3][row] = w.w;
    }
}

inline void load_inputs (
    __global const float* input,
    __local float l_input[LI_STRIDE_IN][LI_INPUT_STRIDE],
    int K,
    int M,
    int g_row_base,
    int g_col_base,
    int l_row,
    int l_col ) {

    l_col <<= 2;

#pragma unroll
    for (int loop = 0; loop < 2; loop ++) {
        int col = l_col + (loop << 4);
        int g_col = g_col_base + col;

#pragma unroll
        for (int t = 0; t < LI_TPT; ++t) {
            int row = l_row * LI_TPT + t;
            int g_row = g_row_base + t;

            float4 in = (float4)0.0f;
            if (g_row < M && g_col < K) {
                in = vload4(0, &input[g_row * K + g_col]);
            }

            l_input[col + 0][row] = in.x;
            l_input[col + 1][row] = in.y;
            l_input[col + 2][row] = in.z;
            l_input[col + 3][row] = in.w;
        }
    }
}

inline void gemm (
    __local float l_input[LI_STRIDE_IN][LI_INPUT_STRIDE],
    __local float l_weights[LI_STRIDE_IN][LI_STRIDE_WEIGHT],
    float acc[LI_TPT][LI_OPT],
    int l_row, 
    int l_col) {

    float4* p_acc0 = (float4*)&acc[0][0];
    float4* p_acc1 = (float4*)&acc[1][0];
    float4* p_acc2 = (float4*)&acc[2][0];
    float4* p_acc3 = (float4*)&acc[3][0];

    int w_base = l_col * LI_OPT;
    int in_base = l_row * LI_TPT;

    for (int k = 0; k < LI_TILE; ++k) {
        __local float* w_ptr = &l_weights[k][w_base];
        
        float4 w0 = vload4(0, w_ptr);
        float4 w1 = vload4(1, w_ptr);
        float4 w2 = vload4(2, w_ptr);
        float4 w3 = vload4(3, w_ptr);

        float4 in = vload4(0, &l_input[k][in_base]); 
        
        p_acc0[0] = fma((float4)(in.x), w0, p_acc0[0]);
        p_acc0[1] = fma((float4)(in.x), w1, p_acc0[1]);
        p_acc0[2] = fma((float4)(in.x), w2, p_acc0[2]);
        p_acc0[3] = fma((float4)(in.x), w3, p_acc0[3]);
        
        p_acc1[0] = fma((float4)(in.y), w0, p_acc1[0]);
        p_acc1[1] = fma((float4)(in.y), w1, p_acc1[1]);
        p_acc1[2] = fma((float4)(in.y), w2, p_acc1[2]);
        p_acc1[3] = fma((float4)(in.y), w3, p_acc1[3]);

        p_acc2[0] = fma((float4)(in.z), w0, p_acc2[0]);
        p_acc2[1] = fma((float4)(in.z), w1, p_acc2[1]);
        p_acc2[2] = fma((float4)(in.z), w2, p_acc2[2]);
        p_acc2[3] = fma((float4)(in.z), w3, p_acc2[3]);

        p_acc3[0] = fma((float4)(in.w), w0, p_acc3[0]);
        p_acc3[1] = fma((float4)(in.w), w1, p_acc3[1]);
        p_acc3[2] = fma((float4)(in.w), w2, p_acc3[2]);
        p_acc3[3] = fma((float4)(in.w), w3, p_acc3[3]);
    }
}

inline void store_linear (
    __global float* output,
    __global const float* bias,
    float acc[LI_TPT][LI_OPT],
    int M,
    int N,
    int g_row_base,
    int g_col_base,
    int gelu ) {

    float4 b0 = vload4(0, &bias[g_col_base + 0]);
    float4 b1 = vload4(0, &bias[g_col_base + 4]);
    float4 b2 = vload4(0, &bias[g_col_base + 8]);
    float4 b3 = vload4(0, &bias[g_col_base + 12]);

#pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int g_row = g_row_base + t;
        if (g_row < M) {
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

            __global float* p_out = &output[g_row * N + g_col_base];
            if (g_col_base + 4 <= N) vstore4(res0, 0, p_out + 0);
            if (g_col_base + 8 <= N) vstore4(res1, 0, p_out + 4);
            if (g_col_base + 12 <= N) vstore4(res2, 0, p_out + 8);
            if (g_col_base + 16 <= N) vstore4(res3, 0, p_out + 12);
        }
    }
}

__kernel void linear_layer (
    __global const float* input,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M, 
    const int K, 
    const int N,
    const int gelu ) {

    __local float l_input[LI_STRIDE_IN][LI_INPUT_STRIDE];
    __local float l_weights[LI_STRIDE_IN][LI_STRIDE_WEIGHT];

    int g_col = get_global_id(0) * LI_OPT;
    int g_row = get_global_id(1) * LI_TPT;

    int l_row = get_local_id(1);
    int l_col = get_local_id(0);

    int col_base = get_group_id(0) * LI_LWS_OUT * LI_OPT;

    float acc[LI_TPT][LI_OPT];

#pragma unroll
    for (int t = 0; t < LI_TPT; ++t)
#pragma unroll
        for (int c = 0; c < LI_OPT; ++c) 
            acc[t][c] = 0.0f;

    for (int k = 0; k < K; k += LI_TILE) {
        load_inputs(input, l_input, K, M, g_row, k, l_row, l_col);
        load_weights(weights, l_weights, K, N, col_base, k, l_row, l_col);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        gemm(l_input, l_weights, acc, l_row, l_col);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_linear(output, bias, acc, M, N, g_row, g_col, gelu);
}

inline void load_conv2d (
    __global const float* img,
    __local float l_input[LI_STRIDE_IN][LI_INPUT_STRIDE],
    int* p_patch,
    int K, 
    int M,
    int g_row_base, 
    int g_col_base,
    int l_row, 
    int l_col ) {

    l_col <<= 2;

#pragma unroll
    for (int loop = 0; loop < 2; loop ++) {
        int col = l_col + (loop << 4);
        int g_col = g_col_base + col;

#pragma unroll
        for (int t = 0; t < LI_TPT; ++t) {
            int row = l_row * LI_TPT + t;
            int g_row = g_row_base + t;

            float4 val = (float4)0.0f;
            if (g_row < M && g_col < K) {
                int ch = g_col >> 8;
                int patch_offset = g_col & 255;
                
                int py = patch_offset >> 4;
                int px = patch_offset & 15;
                int addr = p_patch[t] + ch * (IMG_SIZE * IMG_SIZE) + py * IMG_SIZE + px;
                val = vload4(0, &img[addr]);
            }

            l_input[col + 0][row] = val.x;
            l_input[col + 1][row] = val.y;
            l_input[col + 2][row] = val.z;
            l_input[col + 3][row] = val.w;
        }
    }
}

__kernel void linear_conv2d(
    __global const float* img,
    __global float* output,
    __global const float* weights,
    __global const float* bias,
    const int M, 
    const int K, 
    const int N ) {

    __local float l_input[LI_STRIDE_IN][LI_INPUT_STRIDE];
    __local float l_weights[LI_STRIDE_IN][LI_STRIDE_WEIGHT];

    int g_col = get_global_id(0) * LI_OPT;
    int g_row = get_global_id(1) * LI_TPT;
    
    int l_row = get_local_id(1);
    int l_col = get_local_id(0);

    int col_base = get_group_id(0) * LI_LWS_OUT * LI_OPT;

    int p_patch[LI_TPT];
#pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int patch = g_row + t;

        if (patch < M) {
            int batch = patch / (OUTPUT_SIZE * OUTPUT_SIZE);
            int in_batch = patch % (OUTPUT_SIZE * OUTPUT_SIZE);

            int py = in_batch / OUTPUT_SIZE;
            int px = in_batch % OUTPUT_SIZE;

            int img_y = py << 4;
            int img_x = px << 4;

            p_patch[t] = (batch * CHANNELS) * (IMG_SIZE * IMG_SIZE) + img_y * IMG_SIZE + img_x;
        }
    }

    float acc[LI_TPT][LI_OPT];

#pragma unroll
    for (int t = 0; t < LI_TPT; ++t)
#pragma unroll
        for (int c = 0; c < LI_OPT; ++c) 
            acc[t][c] = 0.0f;

    for (int k = 0; k < K; k += LI_TILE) {
        load_conv2d(img, l_input, p_patch, K, M, g_row, k, l_row, l_col);
        load_weights(weights, l_weights, K, N, col_base, k, l_row, l_col);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        gemm(l_input, l_weights, acc, l_row, l_col);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_linear(output, bias, acc, M, N, g_row, g_col, 0);
}

inline void load_q(
    __global const float* QKV,
    __local float l_input[LI_STRIDE_IN][LI_INPUT_STRIDE],
    int bh_offset, 
    int g_row_base, 
    int g_col_base,
    int l_row, 
    int l_col ) {

    l_col <<= 2;

#pragma unroll
    for (int loop = 0; loop < 2; loop++) {
        int col = l_col + (loop << 4);
        int g_col = g_col_base + col; // 주의: g_col_base + offset

#pragma unroll
        for (int t = 0; t < LI_TPT; ++t) {
            int row = l_row * LI_TPT + t;
            int g_row = g_row_base + t;

            int addr = bh_offset + g_row * QKV_DIM + g_col;
            float4 val = (float4)0.0f;
            if (g_row < TOKENS) {
                val = vload4(0, &QKV[addr]);
            }

            l_input[col + 0][row] = val.x;
            l_input[col + 1][row] = val.y;
            l_input[col + 2][row] = val.z;
            l_input[col + 3][row] = val.w;
        }
    }
}

inline void load_k (
    __global const float* QKV,
    __local float l_weights[LI_STRIDE_IN][LI_STRIDE_WEIGHT],
    int bh_offset,
    int token_base,
    int dim_base,
    int l_token,
    int l_dim ) {
    int qkv_base = bh_offset + EMBED_DIM;
    int token = token_base + l_token;

    #pragma unroll
    for (int loop = 0; loop < 2; ++loop) {
        int dim = (l_dim << 2) + (loop << 4);

        int head_dim = dim_base + dim;
        int addr = qkv_base + token * QKV_DIM + head_dim;

        float4 val = (float4)0.0f;
        
        if (token < TOKENS && head_dim < HEAD_DIM) {
            val = vload4(0, &QKV[addr]);
        }

        l_weights[dim + 0][l_token] = val.x;
        l_weights[dim + 1][l_token] = val.y;
        l_weights[dim + 2][l_token] = val.z;
        l_weights[dim + 3][l_token] = val.w;
    }
}

inline void store_score (
    __global float* scores,
    float acc[LI_TPT][LI_OPT],
    int g_row, 
    int g_col,
    int bh_idx ) {

    int offset = bh_idx * (TOKENS * TOKENS);
    const float scale = 0.125f;

#pragma unroll
    for (int t = 0; t < LI_TPT; ++t) {
        int token = g_row + t;
        
        if (token < TOKENS) {
            float vals[LI_OPT];

#pragma unroll
            for (int i = 0; i < LI_OPT; i++) {
                vals[i] = acc[t][i] * scale;
            }

            int base = offset + token * TOKENS;
            int col = g_col;

#pragma unroll
            for (int i = 0; i < LI_OPT; ++i) {
                if (col + i < TOKENS) {
                    scores[base + col + i] = vals[i];
                }
            }
        }
    }
}

__kernel void attn_score(
    __global const float* QKV,
    __global float* scores ) {
    __local float l_input[LI_STRIDE_IN][LI_INPUT_STRIDE];
    __local float l_weights[LI_STRIDE_IN][LI_STRIDE_WEIGHT];

    int g_col = get_global_id(0) * LI_OPT;
    int g_row = get_global_id(1) * LI_TPT;

    int l_row = get_local_id(1);
    int l_col = get_local_id(0);

    int col_base = get_group_id(0) * LI_LWS_OUT * LI_OPT;
    
    int bh_idx = get_global_id(2); 
    int batch = bh_idx / NUM_HEADS;
    int head = bh_idx % NUM_HEADS;

    int batch_offset = batch * TOKENS * QKV_DIM;
    int head_offset = head * HEAD_DIM;
    int bh_offset = batch_offset + head_offset;

    float acc[LI_TPT][LI_OPT];
    
#pragma unroll
    for (int t = 0; t < LI_TPT; ++t)
#pragma unroll
        for (int c = 0; c < LI_OPT; ++c) 
            acc[t][c] = 0.0f;

    for (int k = 0; k < HEAD_DIM; k += LI_TILE) {
        load_q(QKV, l_input, bh_offset, g_row, k, l_row, l_col);
        load_k(QKV, l_weights, bh_offset, col_base, k, l_row, l_col);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        gemm(l_input, l_weights, acc, l_row, l_col);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_score(scores, acc, g_row, g_col, bh_idx);
}

__kernel void softmax(
    __global float* scores,
    const int size) {

    int row = get_group_id(0);
    int l_idx = get_local_id(0);
    int row_offset = row * size;

    __local float l_cache[256];

    float thread_max = -INFINITY;
    int i = l_idx << 2;
    
    while (i < size) {
        if (i + 3 < size) {
            float4 val = vload4(0, &scores[row_offset + i]);
            thread_max = fmax(thread_max, val.x);
            thread_max = fmax(thread_max, val.y);
            thread_max = fmax(thread_max, val.z);
            thread_max = fmax(thread_max, val.w);
        }
        else {
            for (int k = 0; k < 4 && (i + k) < size; ++k) {
                float val = scores[row_offset + i + k];
                if (val > thread_max) thread_max = val;
            }
        }
        i += 1024;
    }

    l_cache[l_idx] = thread_max;
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (l_idx < stride) {
            l_cache[l_idx] = fmax(l_cache[l_idx], l_cache[l_idx + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float g_max = l_cache[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float thread_sum = 0.0f;
    i = l_idx << 2;

    while (i < size) {
        if (i + 3 < size) {
            float4 val = vload4(0, &scores[row_offset + i]);

            val.x = exp(val.x - g_max);
            val.y = exp(val.y - g_max);
            val.z = exp(val.z - g_max);
            val.w = exp(val.w - g_max);

            vstore4(val, 0, &scores[row_offset + i]);
            thread_sum += (val.x + val.y + val.z + val.w);
        }
        else {
            for (int k = 0; k < 4 && (i + k) < size; ++k) {
                float val = scores[row_offset + i + k];
                val = exp(val - g_max);
                scores[row_offset + i + k] = val;
                thread_sum += val;
            }
        }

        i += 1024;
    }

    l_cache[l_idx] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (l_idx < stride) {
            l_cache[l_idx] += l_cache[l_idx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float inv_sum = 1.0f / l_cache[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    i = l_idx << 2;

    while (i < size) {
        if (i + 3 < size) {
            float4 val = vload4(0, &scores[row_offset + i]);
            val *= inv_sum;
            vstore4(val, 0, &scores[row_offset + i]);
        }
        else {
            for (int k = 0; k < 4 && (i + k) < size; ++k) {
                scores[row_offset + i + k] *= inv_sum;
            }
        }
        i += 1024;
    }
}

__kernel void attn_context(
    __global const float* scores,
    __global const float* QKV,
    __global float* attn_out ) {
    const int g_row = get_global_id(0) << 2;
    const int g_dim = get_global_id(1) << 2;
    const int bh_idx = get_global_id(2);

    if (g_row >= TOKENS) return;

    const int batch = bh_idx / NUM_HEADS;
    const int head = bh_idx % NUM_HEADS;

    const int head_dim_offset = head * HEAD_DIM;
    const int batch_offset = batch * TOKENS;

    const int score_bh_offset = bh_idx * TOKENS * TOKENS;
    const __global float* p_score_base = scores + score_bh_offset + g_row * TOKENS;

    const int qkv_bh_offset = batch_offset * QKV_DIM;
    const int v_offset = (EMBED_DIM << 1) + head_dim_offset + g_dim;
    const __global float* v_ptr = QKV + qkv_bh_offset + v_offset;

    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);

    const __global float* p_score0 = p_score_base;
    const __global float* p_score1 = p_score_base + TOKENS;
    const __global float* p_score2 = p_score_base + (TOKENS << 1);
    const __global float* p_score3 = p_score_base + (TOKENS * 3);

    const bool v_row0 = (g_row + 1 < TOKENS);
    const bool v_row1 = (g_row + 2 < TOKENS);
    const bool v_row2 = (g_row + 3 < TOKENS);

    for (int token = 0; token < TOKENS; ++token) {
        float4 v_val = vload4(0, v_ptr);
        v_ptr += QKV_DIM;

        float score0 = *p_score0++;
        acc0 = fma(v_val, (float4)(score0), acc0);

        if (v_row0) {
            float score1 = *p_score1++;
            acc1 = fma(v_val, (float4)(score1), acc1);
        }
        if (v_row1) {
            float score2 = *p_score2++;
            acc2 = fma(v_val, (float4)(score2), acc2);
        }
        if (v_row2) {
            float score3 = *p_score3++;
            acc3 = fma(v_val, (float4)(score3), acc3);
        }
    }

    const int out_base = (batch_offset + g_row) * EMBED_DIM + head_dim_offset + g_dim;
    __global float* p_out = attn_out + out_base;

    vstore4(acc0, 0, p_out);
    if (v_row0) vstore4(acc1, 0, p_out + EMBED_DIM);
    if (v_row1) vstore4(acc2, 0, p_out + (EMBED_DIM << 1));
    if (v_row2) vstore4(acc3, 0, p_out + (EMBED_DIM * 3));
}

__kernel void layer_norm(
    __global const float* input,
    __global float* output,
    __global const float* weight,
    __constant float* bias) {

    int token = get_global_id(0);
    if (token >= TOTAL_TOKENS) return;

    int offset = token * EMBED_DIM;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < EMBED_DIM; i++) {
        float val = input[offset + i];
        sum += val;
        sum_sq += val * val;
    }

    const float inv_dim = 1.0f / EMBED_DIM;

    float mean = sum * inv_dim;
    float var = sum_sq * inv_dim - mean * mean;

    float inv_std = rsqrt(var + EPS);

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
    // if (i >= size) return;

    output[i] = a[i] + b[i];
}

__kernel void pos_embedding(
    __global const float* patches,
    __global const float* cls_token,
    __global const float* pos_emb,
    __global float* output) {

    int dim = get_global_id(0);
    int token = get_global_id(1);
    int batch = get_global_id(2);

    if (token >= TOKENS) return;

    int out_idx = batch * (TOKENS * EMBED_DIM) + token * EMBED_DIM + dim;

    float pos_val = pos_emb[token * EMBED_DIM + dim];
    float token_val = (token == 0) ? cls_token[dim] : patches[batch * (NUM_PATCHES * EMBED_DIM) + (token - 1) * EMBED_DIM + dim];

    output[out_idx] = token_val + pos_val;
}

__kernel void extract_cls(
    __global const float* input,
    __global float* output) {

    int batch = get_global_id(0);
    int dim = get_global_id(1);

    int src_idx = batch * (TOKENS * EMBED_DIM) + dim;
    int dst_idx = batch * EMBED_DIM + dim;

    output[dst_idx] = input[src_idx];
}