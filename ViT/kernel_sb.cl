#define IMG_SIZE 224
#define PATCH_SIZE 16
#define CHANNELS 3

__kernel void linear_layer(
	__global const float* input,
	__global float* output,
	__global const float* weights,
	__global const float* bias,
	const int token_size,
	const int input_size,
	const int output_size) {
	
	int token_index = get_global_id(0);
	int index = get_global_id(1);

	if (token_index >= token_size) return;
	if (index >= output_size) return;

	float sum = bias[index];

    for (int i = 0; i < input_size; i++) {
        sum += input[token_index * input_size + i] * weights[index * input_size + i];
    }

    output[token_index * output_size + index] = sum;
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
    const int tokens,
    const int head_dim,
    const int embed_dim,
    const int head_offset,
    const int qkv_dim,
    const int batch_size ) {

    int i = get_global_id(0); 
    int j = get_global_id(1);
    int b = get_global_id(2);

    if (i >= tokens || j >= tokens || b >= batch_size) return;

    int qkv_batch_offset = b * (tokens * qkv_dim);
    int score_batch_offset = b * (tokens * tokens);

    float score = 0.0f;
    float scale = 1.0f / sqrt((float)head_dim);

    int q_base = qkv_batch_offset + i * qkv_dim + head_offset;
    int k_base = qkv_batch_offset + j * qkv_dim + embed_dim + head_offset;

    for (int d = 0; d < head_dim; d++) {
        float q = QKV[q_base + d];
        float k = QKV[k_base + d];

        score += q * k;
    }

    scores[score_batch_offset + i * tokens + j] = score * scale;
}

__kernel void softmax(__global float* scores, const int tokens, int batch_size) {
    int row = get_global_id(0);
    
    if (row >= tokens * batch_size) return;

    int offset = row * tokens;

    float max_val = scores[offset];
    for (int j = 1; j < tokens; j++) {
        float val = scores[offset + j];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < tokens; j++) {
        float exp_val = exp(scores[offset + j] - max_val);
        scores[offset + j] = exp_val;
        sum_exp += exp_val;
    }

    for (int j = 0; j < tokens; j++) {
        scores[offset + j] /= sum_exp;
    }
}

__kernel void context (
    __global const float* scores,
    __global const float* QKV,
    __global float* attn_out,
    const int tokens,
    const int head_dim,
    const int embed_dim,
    const int head_offset,
    const int qkv_dim,
    const int batch_size ) {

    int i = get_global_id(0);
    int d = get_global_id(1);
    int b = get_global_id(2);

    if (i >= tokens || d >= head_dim || b >= batch_size) return;

    int batch_score_offset = b * (tokens * tokens);
    int batch_qkv_offset = b * (tokens * qkv_dim);
    int batch_out_offset = b * (tokens * embed_dim);

    float sum = 0.0f;

    for (int j = 0; j < tokens; j++) {
        float s = scores[batch_score_offset + i * tokens + j];
        float v = QKV[batch_qkv_offset + j * qkv_dim + (2 * embed_dim) + head_offset + d];

        sum += s * v;
    }

    int out_idx = batch_out_offset + i * embed_dim + head_offset + d;
    attn_out[out_idx] = sum;
}

__kernel void conv2d (
    __global const float* input,
    __global float* output,
    __global const float* weight,
    __global const float* bias,
    const int output_size,
    const int embed_dim,
    const int batch_size) {

    int oc = get_global_id(0);

    int patch_index = get_global_id(1);
    int batch_index = get_global_id(2);

    int oh = patch_index / output_size;
    int ow = patch_index % output_size;

    if (oc >= embed_dim || oh >= output_size || ow >= output_size || batch_index >= batch_size) return;

    int batch_offset = batch_index * (output_size * output_size * embed_dim);

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

    batch_offset = batch_index * (output_size * output_size * embed_dim);
    int out_idx = batch_offset + patch_index * embed_dim + oc;
    
    output[out_idx] = sum;
}

__kernel void layer_norm (
    __global const float* input,
    __global float* output,
    __global const float* weight,
    __global const float* bias,
    const int tokens,
    const int dim) {

    int t = get_global_id(0);
    if (t >= tokens) return;

    int offset = t * dim;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = 0; i < dim; i++) {
        float val = input[offset + i];
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / dim;
    float var = sum_sq / dim - mean * mean;
    
    float inv_std = rsqrt(var + 0.000001f);

    for (int i = 0; i < dim; i++) {
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
    __global float* output,
    const int dim,
    const int num_patches ) {

    int d = get_global_id(0);
    int t = get_global_id(1);
    int b = get_global_id(2);

    if (t >= num_patches + 1) return;

    int out_idx = b * ((num_patches + 1) * dim) + t * dim + d;
    
    float pos_val = pos_emb[t * dim + d];
    float token_val = (t == 0) ? cls_token[d] : patches[b * (num_patches * dim) + (t - 1) * dim + d];

    output[out_idx] = token_val + pos_val;
}

__kernel void extract_cls (
    __global const float* input,
    __global float* output,
    const int tokens,
    const int embed_dim ) {

    int b = get_global_id(0);
    int d = get_global_id(1);

    int src_idx = b * (tokens * embed_dim) + d;
    int dst_idx = b * embed_dim + d;

    output[dst_idx] = input[src_idx];
}