#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "Network.h"
#include "debug.h"
#include "ViT_seq.h"
#define img_size 224
#define patch_size 16
#define in_chans 3
#define num_classes 1000
#define embed_dim 768
#define depth 12
#define num_heads 12
#define mlp_ratio 4.0
#define dropout 0.0
#define attn_dropout 0.0
#define drop_path_rate 0.0
#define eps 1e-6

typedef struct __cl_context {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    cl_kernel conv2d_kernel;
    cl_kernel linear_kernel;
    cl_kernel gelu_kernel;
    cl_kernel score_kernel;
    cl_kernel softmax_kernel;
    cl_kernel context_kernel;
	cl_kernel normalize_kernel;

    cl_kernel add_kernel;
} CLContext;

static CLContext ctx = { 0 };

static void linear_layer(float* input, float* output, int tokens, int in_features, int out_features, Network weight, Network bias);
static void linear_layer_gpu(cl_mem input, cl_mem output, int tokens, int in_features, int out_features, Network weight, Network bias);
static void add_gpu(cl_mem a, cl_mem b, cl_mem output, int size);

////////////////////////////////////// ViT function //////////////////////////////////////

// input : (3, 224, 224)
// output : (768, 14, 14)

static void Conv2d(float* input, float* output, Network weight, Network bias) {
    int output_size = img_size / patch_size;
	int dim = embed_dim;

	cl_mem input_buf, output_buf;
	cl_int err;

	input_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * in_chans * img_size * img_size, input, &err); CHECK_ERROR(err);
	output_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * dim * output_size * output_size, NULL, &err); CHECK_ERROR(err);

	err = clSetKernelArg(ctx.conv2d_kernel, 0, sizeof(cl_mem), &input_buf); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.conv2d_kernel, 1, sizeof(cl_mem), &output_buf); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.conv2d_kernel, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.conv2d_kernel, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.conv2d_kernel, 4, sizeof(int), &output_size); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.conv2d_kernel, 5, sizeof(int), &dim); CHECK_ERROR(err);

    size_t global_work_size[3] = { (size_t)dim, (size_t)output_size, (size_t)output_size };

	err = clEnqueueNDRangeKernel(ctx.queue, ctx.conv2d_kernel, 3, NULL, global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
	err = clEnqueueReadBuffer(ctx.queue, output_buf, CL_TRUE, 0, sizeof(float) * dim * output_size * output_size, output, 0, NULL, NULL); CHECK_ERROR(err);
}

static void Conv2d_gpu(cl_mem input, cl_mem output, Network weight, Network bias) {
    int output_size = img_size / patch_size;
    int dim = embed_dim;

    cl_int err;

    err = clSetKernelArg(ctx.conv2d_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.conv2d_kernel, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.conv2d_kernel, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.conv2d_kernel, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.conv2d_kernel, 4, sizeof(int), &output_size); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.conv2d_kernel, 5, sizeof(int), &dim); CHECK_ERROR(err);

    size_t global_work_size[3] = { (size_t)dim, (size_t)output_size, (size_t)output_size };

    err = clEnqueueNDRangeKernel(ctx.queue, ctx.conv2d_kernel, 3, NULL, global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

// input : (768, 14, 14)
// output : (num_patches, embed_dim) -> (196, 768)

static void flatten_transpose(float* input, float* output) {
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;

    // 각 공간 위치(oh, ow)를 하나의 패치로 취급하여 patch index 계산
    for (int oh = 0; oh < output_size; oh++) {
        for (int ow = 0; ow < output_size; ow++) {
            int patch_idx = oh * output_size + ow;
            for (int oc = 0; oc < embed_dim; oc++) {
                // 기존 입력은 (oc, oh, ow)
                int idx_input = (oc * output_size + oh) * output_size + ow;
                // 원하는 출력은 (patch_idx, oc)
                int idx_output = patch_idx * embed_dim + oc;
                output[idx_output] = input[idx_input];
                //printf("%f ",output[idx_output]);
            }
        }
    }
}

static void class_token(float* patch_tokens, float* final_tokens, Network cls_tk) {
    // 이미지의 패치 수 계산: output_size = img_size / patch_size, num_patches = output_size^2
    int output_size = img_size / patch_size;
    int num_patches = output_size * output_size;

    // 1. 첫 번째 토큰에 class token 복사 (networks[0].data에 저장됨, embed_dim 길이)
    for (int j = 0; j < embed_dim; j++) {
        final_tokens[j] = cls_tk.data[j];
    }

    // 2. 이후 patch_tokens를 이어붙임
    // final_tokens의 인덱스 embed_dim부터, patch_tokens 전체(embed_dim * num_patches) 복사
    memcpy(final_tokens + embed_dim, patch_tokens, sizeof(float) * embed_dim * num_patches);

    int total_tokens = num_patches + 1; // class token + patch tokens
    for (int i = 0; i < total_tokens * embed_dim; i++) {
        //("%f ", final_tokens[i]);
    }
    //printf("\n");
}

static void pos_emb(float* input, float* output, Network pos_emb) {
    // output_size: 한 변의 패치 수, num_patches: 전체 패치 수, total_tokens: class token + patch tokens
    int output_size = img_size / patch_size;
    int total_elements = (output_size * output_size + 1) * embed_dim;

    for (int i = 0; i < total_elements; i++) {
        output[i] = input[i] + pos_emb.data[i];
    }
}

static void pos_emb_gpu(cl_mem input, cl_mem output, Network pos_emb) {
    int output_size = img_size / patch_size;
    int total_elements = (output_size * output_size + 1) * embed_dim;


}

static void layer_norm(float* input, float* output, Network weight, Network bias) {
    int token = ((img_size / patch_size) * (img_size / patch_size)) + 1;

    for (int t = 0; t < token; t++) {
        float sum = 0.0, sum_sq = 0.0;
        for (int i = 0; i < embed_dim; i++) {
            float val = input[t * embed_dim + i];
            sum += val;
            sum_sq += val * val;
        }
        float mean = sum / embed_dim;
        float var = sum_sq / embed_dim - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < embed_dim; i++) {
            int idx = t * embed_dim + i;
            output[idx] = (input[idx] - mean) * inv_std * weight.data[i] + bias.data[i];
        }
    }
}

static void layer_norm_gpu(cl_mem input, cl_mem ouput, Network weight, Network bias) {
	int token = ((img_size / patch_size) * (img_size / patch_size)) + 1;
	int dim = embed_dim;

	cl_int err;

	err = clSetKernelArg(ctx.normalize_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.normalize_kernel, 1, sizeof(cl_mem), &ouput); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.normalize_kernel, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.normalize_kernel, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.normalize_kernel, 4, sizeof(int), &token); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.normalize_kernel, 5, sizeof(int), &dim); CHECK_ERROR(err);

	size_t global_work_size = (size_t)token;
	err = clEnqueueNDRangeKernel(ctx.queue, ctx.normalize_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

static void multihead_attn(float* input, float* output,
    Network in_weight, Network in_bias, Network out_weight, Network out_bias) {

    int head_dim = embed_dim / num_heads, tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;

    /*Allocate Q, K, V : tokens * dim*/
	int QKV_dim = embed_dim * 3;
	float* QKV = (float*)malloc(sizeof(float) * tokens * QKV_dim);

	linear_layer(input, QKV, tokens, embed_dim, QKV_dim, in_weight, in_bias);

    int print_tokens = tokens < 5 ? tokens : 5;
    int print_dims = embed_dim < 10 ? embed_dim : 10;

	float* attn_output = (float*)calloc(sizeof(float), tokens * embed_dim);

    cl_int err;

	cl_mem attn_output_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * tokens * embed_dim, attn_output, &err); CHECK_ERROR(err);
	cl_mem QKV_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * tokens * QKV_dim, QKV, &err); CHECK_ERROR(err);

    /*head별로 attn 수행*/
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;
        int dim = embed_dim;

        // attn_score 저장 공간
        float* scores = (float*)malloc(sizeof(float) * tokens * tokens);
		cl_mem scores_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * tokens * tokens, NULL, &err); CHECK_ERROR(err);

		// Attention Score 계산
		err = clSetKernelArg(ctx.score_kernel, 0, sizeof(cl_mem), &QKV_buf); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.score_kernel, 1, sizeof(cl_mem), &scores_buf); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.score_kernel, 2, sizeof(int), &tokens); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.score_kernel, 3, sizeof(int), &head_dim); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.score_kernel, 4, sizeof(int), &dim); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.score_kernel, 5, sizeof(int), &head_offset); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.score_kernel, 6, sizeof(int), &QKV_dim); CHECK_ERROR(err);

		size_t global_work_size[2] = { (size_t)tokens, (size_t)tokens };
		err = clEnqueueNDRangeKernel(ctx.queue, ctx.score_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);

		// Softmax 계산
		err = clSetKernelArg(ctx.softmax_kernel, 0, sizeof(cl_mem), &scores_buf); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.softmax_kernel, 1, sizeof(int), &tokens); CHECK_ERROR(err);

		size_t global_work_size_softmax = (size_t)tokens;
		err = clEnqueueNDRangeKernel(ctx.queue, ctx.softmax_kernel, 1, NULL, &global_work_size_softmax, NULL, 0, NULL, NULL); CHECK_ERROR(err);

		// Context Vector 계산
		err = clSetKernelArg(ctx.context_kernel, 0, sizeof(cl_mem), &scores_buf); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.context_kernel, 1, sizeof(cl_mem), &QKV_buf); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.context_kernel, 2, sizeof(cl_mem), &attn_output_buf); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.context_kernel, 3, sizeof(int), &tokens); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.context_kernel, 4, sizeof(int), &head_dim); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.context_kernel, 5, sizeof(int), &dim); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.context_kernel, 6, sizeof(int), &head_offset); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.context_kernel, 7, sizeof(int), &QKV_dim); CHECK_ERROR(err);

		size_t global_work_size_context[2] = { (size_t)tokens, (size_t)head_dim };
		err = clEnqueueNDRangeKernel(ctx.queue, ctx.context_kernel, 2, NULL, global_work_size_context, NULL, 0, NULL, NULL); CHECK_ERROR(err);
		err = clEnqueueReadBuffer(ctx.queue, attn_output_buf, CL_TRUE, 0, sizeof(float) * tokens * embed_dim, attn_output, 0, NULL, NULL); CHECK_ERROR(err);

        free(scores);
        clReleaseMemObject(scores_buf);
    }

	linear_layer(attn_output, output, tokens, embed_dim, embed_dim, out_weight, out_bias);

    free(QKV);
    free(attn_output);
	clReleaseMemObject(QKV_buf);
	clReleaseMemObject(attn_output_buf);
}

static void multihead_attn_gpu(cl_mem input, cl_mem output,
    Network in_weight, Network in_bias, Network out_weight, Network out_bias) {

    cl_int err;
    int head_dim = embed_dim / num_heads, tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;

    int QKV_dim = embed_dim * 3;
	cl_mem QKV = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * tokens * QKV_dim, NULL, &err); CHECK_ERROR(err);

    linear_layer_gpu(input, QKV, tokens, embed_dim, QKV_dim, in_weight, in_bias);

    int print_tokens = tokens < 5 ? tokens : 5;
    int print_dims = embed_dim < 10 ? embed_dim : 10;

    cl_mem attn_output_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * tokens * embed_dim, NULL, &err); CHECK_ERROR(err);

    /*head별로 attn 수행*/
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;
        int dim = embed_dim;

        // attn_score 저장 공간
        float* scores = (float*)malloc(sizeof(float) * tokens * tokens);
        cl_mem scores_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * tokens * tokens, NULL, &err); CHECK_ERROR(err);

        // Attention Score 계산
        err = clSetKernelArg(ctx.score_kernel, 0, sizeof(cl_mem), &QKV); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 1, sizeof(cl_mem), &scores_buf); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 2, sizeof(int), &tokens); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 3, sizeof(int), &head_dim); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 4, sizeof(int), &dim); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 5, sizeof(int), &head_offset); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 6, sizeof(int), &QKV_dim); CHECK_ERROR(err);

        size_t global_work_size[2] = { (size_t)tokens, (size_t)tokens };
        err = clEnqueueNDRangeKernel(ctx.queue, ctx.score_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);

        // Softmax 계산
        err = clSetKernelArg(ctx.softmax_kernel, 0, sizeof(cl_mem), &scores_buf); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.softmax_kernel, 1, sizeof(int), &tokens); CHECK_ERROR(err);

        size_t global_work_size_softmax = (size_t)tokens;
        err = clEnqueueNDRangeKernel(ctx.queue, ctx.softmax_kernel, 1, NULL, &global_work_size_softmax, NULL, 0, NULL, NULL); CHECK_ERROR(err);

        // Context Vector 계산
        err = clSetKernelArg(ctx.context_kernel, 0, sizeof(cl_mem), &scores_buf); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 1, sizeof(cl_mem), &QKV); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 2, sizeof(cl_mem), &attn_output_buf); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 3, sizeof(int), &tokens); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 4, sizeof(int), &head_dim); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 5, sizeof(int), &dim); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 6, sizeof(int), &head_offset); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 7, sizeof(int), &QKV_dim); CHECK_ERROR(err);

        size_t global_work_size_context[2] = { (size_t)tokens, (size_t)head_dim };
        err = clEnqueueNDRangeKernel(ctx.queue, ctx.context_kernel, 2, NULL, global_work_size_context, NULL, 0, NULL, NULL); CHECK_ERROR(err);

        free(scores);
        clReleaseMemObject(scores_buf);
    }

    linear_layer_gpu(attn_output_buf, output, tokens, embed_dim, embed_dim, out_weight, out_bias);

    clReleaseMemObject(QKV);
    clReleaseMemObject(attn_output_buf);
}

static void gelu_activation(float* input, int size) {
    cl_int err;
	cl_mem buffer = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * size, input, &err);

    err = clSetKernelArg(ctx.gelu_kernel, 0, sizeof(cl_mem), &buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.gelu_kernel, 1, sizeof(int), &size); CHECK_ERROR(err);

    size_t global_work_size = size;

    err = clEnqueueNDRangeKernel(ctx.queue, ctx.gelu_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

	err = clEnqueueReadBuffer(ctx.queue, buffer, CL_TRUE, 0, sizeof(float) * size, input, 0, NULL, NULL);
}

static void gelu_activation_gpu(cl_mem input, int size) {
	cl_int err;

	err = clSetKernelArg(ctx.gelu_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.gelu_kernel, 1, sizeof(int), &size); CHECK_ERROR(err);

	size_t global_work_size = size;

	err = clEnqueueNDRangeKernel(ctx.queue, ctx.gelu_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);
}

static void linear_layer(float* input, float* output, int tokens, int in_features, int out_features, Network weight, Network bias) {
    cl_int err;

    cl_mem input_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * tokens * in_features, input, &err); CHECK_ERROR(err);
    cl_mem output_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, sizeof(float) * tokens * out_features, NULL, &err); CHECK_ERROR(err);

	err = clSetKernelArg(ctx.linear_kernel, 0, sizeof(cl_mem), &input_buf); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 1, sizeof(cl_mem), &output_buf); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 4, sizeof(int), &tokens); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 5, sizeof(int), &in_features); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 6, sizeof(int), &out_features); CHECK_ERROR(err);

	size_t global_work_size[2] = { (size_t)tokens, (size_t)out_features };

	err = clEnqueueNDRangeKernel(ctx.queue, ctx.linear_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
	err = clEnqueueReadBuffer(ctx.queue, output_buf, CL_TRUE, 0, sizeof(float) * tokens * out_features, output, 0, NULL, NULL); CHECK_ERROR(err);

	clReleaseMemObject(input_buf);
	clReleaseMemObject(output_buf);
}

static void linear_layer_gpu(cl_mem input, cl_mem output, int tokens, int in_features, int out_features, Network weight, Network bias) {
	cl_int err;

	err = clSetKernelArg(ctx.linear_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 4, sizeof(int), &tokens); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 5, sizeof(int), &in_features); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 6, sizeof(int), &out_features); CHECK_ERROR(err);

	size_t global_work_size[2] = { (size_t)tokens, (size_t)out_features };

	err = clEnqueueNDRangeKernel(ctx.queue, ctx.linear_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

static void mlp_block(float* input, float* output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias) {
    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; //197
    int Embed_dim = embed_dim; //768
    int hidden_dim = ((int)(embed_dim * mlp_ratio)); //3072

    float* fc1_out = (float*)malloc(sizeof(float) * tokens * hidden_dim);

    linear_layer(input, fc1_out, tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias);
	gelu_activation(fc1_out, tokens * hidden_dim);
    linear_layer(fc1_out, output, tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias);

    free(fc1_out);
}

static void mlp_block_gpu(cl_mem input, cl_mem output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias) {
	int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; //197
	int Embed_dim = embed_dim; //768
	int hidden_dim = ((int)(embed_dim * mlp_ratio)); //3072

	cl_int err;
	cl_mem fc1_out = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * tokens * hidden_dim, NULL, &err); CHECK_ERROR(err);

	linear_layer_gpu(input, fc1_out, tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias);
	gelu_activation_gpu(fc1_out, tokens * hidden_dim);
	linear_layer_gpu(fc1_out, output, tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias);

	clReleaseMemObject(fc1_out);
}

////////////////////////////////////// Encoder Architecture //////////////////////////////////////
static void Encoder(float* input, float* output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b) {

    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    float* ln1_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* attn_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* residual = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* ln2_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
    float* mlp_out = (float*)malloc(sizeof(float) * tokens * embed_dim);

    /*LN1*/
    layer_norm(input, ln1_out, ln1_w, ln1_b);

    /*Attn*/
    // start_timer();
    multihead_attn(ln1_out, attn_out, attn_w, attn_b, attn_out_w, attn_out_b);
	// stop_timer("Multi-Head Attention Time: ");

    /*Residual1*/
    for (int i = 0; i < tokens * embed_dim; i++) {
        residual[i] = input[i] + attn_out[i];
    }

    /*LN2*/
    layer_norm(residual, ln2_out, ln2_w, ln2_b);

    /*MLP*/
    mlp_block(ln2_out, mlp_out, mlp1_w, mlp1_b, mlp2_w, mlp2_b);

    /*Residual2*/
    for (int i = 0; i < tokens * embed_dim; i++) {
        output[i] = residual[i] + mlp_out[i];
    }

    free(ln1_out); free(attn_out); free(residual); free(ln2_out); free(mlp_out);
}

static void Encoder_gpu(cl_mem input, cl_mem output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b) {

    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
    int total_elements = tokens * embed_dim; 
    size_t buffer_bytes = sizeof(float) * total_elements;

    cl_int err;

    cl_mem buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, buffer_bytes, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueCopyBuffer(ctx.queue, input, output, 0, 0, buffer_bytes, 0, NULL, NULL);
    CHECK_ERROR(err);

    layer_norm_gpu(output, buf, ln1_w, ln1_b);
    multihead_attn_gpu(buf, input, attn_w, attn_b, attn_out_w, attn_out_b);
    add_gpu(output, input, output, total_elements);

    layer_norm_gpu(output, buf, ln2_w, ln2_b);
    mlp_block_gpu(buf, input, mlp1_w, mlp1_b, mlp2_w, mlp2_b);
    add_gpu(output, input, output, total_elements);

    clReleaseMemObject(buf);
}

static void add_gpu(cl_mem a, cl_mem b, cl_mem output, int size) {
    cl_int err;

    err = clSetKernelArg(ctx.add_kernel, 0, sizeof(cl_mem), &a); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.add_kernel, 1, sizeof(cl_mem), &b); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.add_kernel, 2, sizeof(cl_mem), &output); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.add_kernel, 3, sizeof(int), &size); CHECK_ERROR(err);

	size_t global_work_size = (size_t)size;
	err = clEnqueueNDRangeKernel(ctx.queue, ctx.add_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

static void Softmax(float* logits, float* probabilities, int length) {
    float max_val = logits[0];
    for (int i = 1; i < length; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < length; i++) {
        probabilities[i] = expf(logits[i] - max_val);
        sum_exp += probabilities[i];
    }

    for (int i = 0; i < length; i++) {
        probabilities[i] /= sum_exp;
    }
}

////////////////////////////////////// layer별 size //////////////////////////////////////
static const int size[] = {
    embed_dim * (img_size / patch_size) * (img_size / patch_size), // conv2D
    embed_dim * (img_size / patch_size) * (img_size / patch_size), // flatten and transpose
    embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1), // class token
    embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1) // position embedding
};

static const int enc_size = embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1);

void ViT_seq_sb(ImageData* image, Network* networks, float** probabilities) {
    cl_int err;

    err = clGetPlatformIDs(1, &ctx.platform, NULL);
    CHECK_ERROR(err);

    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, NULL);
    CHECK_ERROR(err);

    ctx.context = clCreateContext(NULL, 1, &ctx.device, NULL, NULL, &err);
    CHECK_ERROR(err);

    ctx.queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, 0, &err);
    CHECK_ERROR(err);

    size_t kernel_source_size;
    char* kernel_source = get_source_code("kernel_sb.cl", &kernel_source_size);
    ctx.program = clCreateProgramWithSource(ctx.context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    err = clBuildProgram(ctx.program, 1, &ctx.device, "", NULL, NULL);
    build_error(ctx.program, ctx.device, err);
    CHECK_ERROR(err);

    for (int i = 0; i < 152; i++) {
        if (networks[i].data == NULL) continue;
		if (networks[i].size <= 0) continue;

        networks[i].buffer = clCreateBuffer(ctx.context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(float) * networks[i].size,
            networks[i].data,
            &err);

        CHECK_ERROR(err);
    }

	ctx.conv2d_kernel = clCreateKernel(ctx.program, "conv2d", &err); CHECK_ERROR(err);

	ctx.linear_kernel = clCreateKernel(ctx.program, "linear_layer", &err); CHECK_ERROR(err);
	ctx.gelu_kernel = clCreateKernel(ctx.program, "gelu_activation", &err); CHECK_ERROR(err);
	ctx.score_kernel = clCreateKernel(ctx.program, "attention_score", &err); CHECK_ERROR(err);
	ctx.softmax_kernel = clCreateKernel(ctx.program, "softmax", &err); CHECK_ERROR(err);
	ctx.context_kernel = clCreateKernel(ctx.program, "context", &err); CHECK_ERROR(err);
	ctx.normalize_kernel = clCreateKernel(ctx.program, "layer_norm", &err); CHECK_ERROR(err);
    ctx.add_kernel = clCreateKernel(ctx.program, "add", &err); CHECK_ERROR(err);

    // below : kernel creation, buffer allocation, data transfer, kernel execution, result retrieval, cleanup //////////////

    int token_size = ((img_size / patch_size) * (img_size / patch_size) + 1); // 197
    float* layer[4];
    float* enc_layer[12];

    cl_mem enc_buf[12];

    float* enc_output;
    int hidden_dim = ((int)(embed_dim * mlp_ratio)); // 3072s

    // printf("%d %d = %d\n", token_size, hidden_dim, token_size * hidden_dim);

    for (int i = 0; i < 4; i++) {
        layer[i] = (float*)malloc(sizeof(float) * size[i]);
    }

    for (int i = 0; i < 12; i++) {
		enc_buf[i] = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * enc_size, NULL, &err); CHECK_ERROR(err);
    }

    enc_output = (float*)malloc(sizeof(float) * enc_size);

    int num_patches = (img_size / patch_size) * (img_size / patch_size);
    size_t total_bytes = sizeof(float) * (num_patches + 1) * embed_dim;

    cl_mem input = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, sizeof(float) * in_chans * img_size * img_size, NULL, &err); CHECK_ERROR(err);
	cl_mem output = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * enc_size, NULL, &err); CHECK_ERROR(err);

	cl_mem buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * size[0], NULL, &err); CHECK_ERROR(err);
	cl_mem layer_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * size[2], NULL, &err); CHECK_ERROR(err);

    cl_mem cls_output = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * num_classes, NULL, &err);

    start_timer();
    for (int i = 0; i < image->n; i++) {
        printf("Processing image %d/%d - ", i + 1, image->n);
        stop_timer("elapsed time");

        err = clEnqueueWriteBuffer(ctx.queue, input, CL_FALSE, 0, sizeof(float) * in_chans * img_size * img_size, image[i].data, 0, NULL, NULL); CHECK_ERROR(err);

		// Patch Embedding
        Conv2d_gpu(input, buf, networks[1], networks[2]);

		// class token
        err = clEnqueueCopyBuffer(ctx.queue, networks[0].buffer, layer_buf, 0, 0, sizeof(float) * embed_dim, 0, NULL, NULL); CHECK_ERROR(err);

		// class token prepended + patch tokens
        err = clEnqueueCopyBuffer(ctx.queue, buf, layer_buf, 0, sizeof(float) * embed_dim, sizeof(float) * size[0], 0, NULL, NULL); CHECK_ERROR(err);

        // positional encoding
        add_gpu(layer_buf, networks[3].buffer, layer_buf, (int)(size[2] / sizeof(float)));

        Encoder_gpu(layer_buf, enc_buf[0],
            networks[4], networks[5], networks[6], networks[7],
            networks[8], networks[9], networks[10], networks[11],
            networks[12], networks[13], networks[14], networks[15]);

        Encoder_gpu(enc_buf[0], enc_buf[1],
            networks[16], networks[17], networks[18], networks[19],
            networks[20], networks[21], networks[22], networks[23],
            networks[24], networks[25], networks[26], networks[27]);

        Encoder_gpu(enc_buf[1], enc_buf[2],
            networks[28], networks[29], networks[30], networks[31],
            networks[32], networks[33], networks[34], networks[35],
            networks[36], networks[37], networks[38], networks[39]);

        Encoder_gpu(enc_buf[2], enc_buf[3],
            networks[40], networks[41], networks[42], networks[43],
            networks[44], networks[45], networks[46], networks[47],
            networks[48], networks[49], networks[50], networks[51]);

        Encoder_gpu(enc_buf[3], enc_buf[4],
            networks[52], networks[53], networks[54], networks[55],
            networks[56], networks[57], networks[58], networks[59],
            networks[60], networks[61], networks[62], networks[63]);

        Encoder_gpu(enc_buf[4], enc_buf[5],
            networks[64], networks[65], networks[66], networks[67],
            networks[68], networks[69], networks[70], networks[71],
            networks[72], networks[73], networks[74], networks[75]);

        Encoder_gpu(enc_buf[5], enc_buf[6],
            networks[76], networks[77], networks[78], networks[79],
            networks[80], networks[81], networks[82], networks[83],
            networks[84], networks[85], networks[86], networks[87]);

        Encoder_gpu(enc_buf[6], enc_buf[7],
            networks[88], networks[89], networks[90], networks[91],
            networks[92], networks[93], networks[94], networks[95],
            networks[96], networks[97], networks[98], networks[99]);

        Encoder_gpu(enc_buf[7], enc_buf[8],
            networks[100], networks[101], networks[102], networks[103],
            networks[104], networks[105], networks[106], networks[107],
            networks[108], networks[109], networks[110], networks[111]);

        Encoder_gpu(enc_buf[8], enc_buf[9],
            networks[112], networks[113], networks[114], networks[115],
            networks[116], networks[117], networks[118], networks[119],
            networks[120], networks[121], networks[122], networks[123]);

        Encoder_gpu(enc_buf[9], enc_buf[10],
            networks[124], networks[125], networks[126], networks[127],
            networks[128], networks[129], networks[130], networks[131],
            networks[132], networks[133], networks[134], networks[135]);

        Encoder_gpu(enc_buf[10], enc_buf[11],
            networks[136], networks[137], networks[138], networks[139],
            networks[140], networks[141], networks[142], networks[143],
            networks[144], networks[145], networks[146], networks[147]);

        layer_norm_gpu(enc_buf[11], output, networks[148], networks[149]);

        
        CHECK_ERROR(err);

        linear_layer_gpu(output, cls_output, 1, embed_dim, num_classes, networks[150], networks[151]);

        int classes = num_classes;
        err = clSetKernelArg(ctx.softmax_kernel, 0, sizeof(cl_mem), &cls_output); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.softmax_kernel, 1, sizeof(int), &classes); CHECK_ERROR(err);

        size_t global_soft = 1;
        err = clEnqueueNDRangeKernel(ctx.queue, ctx.softmax_kernel, 1, NULL, &global_soft, NULL, 0, NULL, NULL); CHECK_ERROR(err);
        err = clEnqueueReadBuffer(ctx.queue, cls_output, CL_TRUE, 0, sizeof(float) * num_classes, probabilities[i], 0, NULL, NULL); CHECK_ERROR(err);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    free(kernel_source);

    clReleaseMemObject(input);
    clReleaseMemObject(buf);
    clReleaseMemObject(layer_buf);
    clReleaseMemObject(cls_output);

	clReleaseKernel(ctx.linear_kernel);
    clReleaseCommandQueue(ctx.queue);
    clReleaseContext(ctx.context);
    clReleaseProgram(ctx.program);
}