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

// custom defines
#define output_size img_size / patch_size
#define num_patches output_size * output_size
#define tokens (num_patches + 1)
#define total_tokens tokens * batch_size

#define head_dim embed_dim / num_heads
#define qkv_dim embed_dim * 3
#define hidden_dim (int)(embed_dim * mlp_ratio)

#define enc_size tokens * embed_dim

#define batch_size 4

typedef struct __cl_context {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;

	cl_command_queue input_queue;
	cl_command_queue compute_queue;

    cl_program program;

    cl_kernel conv2d_kernel;
    cl_kernel linear_kernel;
    cl_kernel gelu_kernel;
    cl_kernel score_kernel;
    cl_kernel softmax_kernel;
    cl_kernel context_kernel;
	cl_kernel normalize_kernel;

    cl_kernel add_kernel;
    cl_kernel prepare_kernel;
	cl_kernel extract_cls_kernel;
} CLContext;

static CLContext ctx = { 0 };

static void linear_layer(cl_mem input, cl_mem output, int token_size, int in_features, int out_features, Network weight, Network bias);
static void add(cl_mem a, cl_mem b, cl_mem output, int size);

////////////////////////////////////// ViT function //////////////////////////////////////

// input : (3, 224, 224)
// output : (768, 14, 14)

static void Conv2d(cl_mem input, cl_mem output, Network weight, Network bias) {
    cl_int err;

    err = clSetKernelArg(ctx.conv2d_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.conv2d_kernel, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.conv2d_kernel, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.conv2d_kernel, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);

    size_t global_work_size[3] = { 
        (size_t)embed_dim, 
        (size_t)num_patches, 
        (size_t)batch_size 
    };

    err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.conv2d_kernel, 3, NULL, global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

static void layer_norm(cl_mem input, cl_mem ouput, Network weight, Network bias) {
	cl_int err;

	err = clSetKernelArg(ctx.normalize_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.normalize_kernel, 1, sizeof(cl_mem), &ouput); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.normalize_kernel, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.normalize_kernel, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);

	size_t global_work_size = (size_t)total_tokens;
	err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.normalize_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

static void multihead_attn(cl_mem input, cl_mem output,
    Network in_weight, Network in_bias, Network out_weight, Network out_bias) {

    cl_int err;

    int total_token = tokens * batch_size;

	cl_mem QKV = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_token * qkv_dim, NULL, &err); CHECK_ERROR(err);

    linear_layer(input, QKV, total_token, embed_dim, qkv_dim, in_weight, in_bias);

    cl_mem attn_output_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_token * embed_dim, NULL, &err); CHECK_ERROR(err);
    cl_mem scores_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_token * tokens, NULL, &err); CHECK_ERROR(err);

    /*head별로 attn 수행*/
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        // Attention Score 계산
        err = clSetKernelArg(ctx.score_kernel, 0, sizeof(cl_mem), &QKV); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 1, sizeof(cl_mem), &scores_buf); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 2, sizeof(int), &head_offset); CHECK_ERROR(err);

        size_t score_size[3] = {
            (size_t)tokens,
            (size_t)tokens,
			(size_t)batch_size
        };

        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.score_kernel, 3, NULL, score_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);

        // Softmax 계산

		int token_size = tokens;
        err = clSetKernelArg(ctx.softmax_kernel, 0, sizeof(cl_mem), &scores_buf); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.softmax_kernel, 1, sizeof(int), &token_size); CHECK_ERROR(err);

        size_t soft_size = (size_t)tokens * batch_size;
        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.softmax_kernel, 1, NULL, &soft_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);

        // Context Vector 계산
        err = clSetKernelArg(ctx.context_kernel, 0, sizeof(cl_mem), &scores_buf); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 1, sizeof(cl_mem), &QKV); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 2, sizeof(cl_mem), &attn_output_buf); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 3, sizeof(int), &head_offset); CHECK_ERROR(err);

        size_t context_size[3] = { 
            (size_t)tokens, 
            (size_t)head_dim,
			(size_t)batch_size
        };

        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.context_kernel, 3, NULL, context_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
    }

    linear_layer(attn_output_buf, output, total_token, embed_dim, embed_dim, out_weight, out_bias);

    clReleaseMemObject(scores_buf);
    clReleaseMemObject(QKV);
    clReleaseMemObject(attn_output_buf);
}

static void gelu_activation(cl_mem input, int size) {
	cl_int err;

	err = clSetKernelArg(ctx.gelu_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.gelu_kernel, 1, sizeof(int), &size); CHECK_ERROR(err);

	size_t global_work_size = size;

	err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.gelu_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);
}

static void linear_layer(cl_mem input, cl_mem output, int token_size, int in_features, int out_features, Network weight, Network bias) {
	cl_int err;

	err = clSetKernelArg(ctx.linear_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 4, sizeof(int), &token_size); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 5, sizeof(int), &in_features); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.linear_kernel, 6, sizeof(int), &out_features); CHECK_ERROR(err);

	size_t global_work_size[2] = { (size_t)token_size, (size_t)out_features };

	err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.linear_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

static void mlp_block(cl_mem input, cl_mem output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias) {
	cl_int err;

	cl_mem fc1_out = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens * hidden_dim, NULL, &err); CHECK_ERROR(err);

	linear_layer(input, fc1_out, total_tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias);
	gelu_activation(fc1_out, total_tokens * hidden_dim);
	linear_layer(fc1_out, output, total_tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias);

	clReleaseMemObject(fc1_out);
}

static void Encoder(cl_mem input, cl_mem output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b) {

    int total_elements = batch_size * tokens * embed_dim; 
    size_t buffer_bytes = sizeof(float) * total_elements;

    cl_int err;

    cl_mem buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, buffer_bytes, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueCopyBuffer(ctx.compute_queue, input, output, 0, 0, buffer_bytes, 0, NULL, NULL);
    CHECK_ERROR(err);

    layer_norm(output, buf, ln1_w, ln1_b);
    multihead_attn(buf, input, attn_w, attn_b, attn_out_w, attn_out_b);
    add(output, input, output, total_elements);

    layer_norm(output, buf, ln2_w, ln2_b);
    mlp_block(buf, input, mlp1_w, mlp1_b, mlp2_w, mlp2_b);
    add(output, input, output, total_elements);

    clReleaseMemObject(buf);
}

static void add(cl_mem a, cl_mem b, cl_mem output, int size) {
    cl_int err;

    err = clSetKernelArg(ctx.add_kernel, 0, sizeof(cl_mem), &a); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.add_kernel, 1, sizeof(cl_mem), &b); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.add_kernel, 2, sizeof(cl_mem), &output); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.add_kernel, 3, sizeof(int), &size); CHECK_ERROR(err);

	size_t global_work_size = (size_t)size;
	err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.add_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

static void prepare_input(cl_mem input, cl_mem cls, cl_mem pos, cl_mem output) {
	cl_int err;

	err = clSetKernelArg(ctx.prepare_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.prepare_kernel, 1, sizeof(cl_mem), &cls); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.prepare_kernel, 2, sizeof(cl_mem), &pos); CHECK_ERROR(err);
	err = clSetKernelArg(ctx.prepare_kernel, 3, sizeof(cl_mem), &output); CHECK_ERROR(err);

    size_t global_work_size[3] = {
        (size_t)embed_dim,
        (size_t)tokens,
		(size_t)batch_size
    };

	err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.prepare_kernel, 3, NULL, global_work_size, NULL, 0, NULL, NULL); 
    CHECK_ERROR(err);
}

////////////////////////////////////// layer별 size //////////////////////////////////////
static const int size[] = {
    embed_dim * (img_size / patch_size) * (img_size / patch_size), // conv2D
    embed_dim * (img_size / patch_size) * (img_size / patch_size), // flatten and transpose
    embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1), // class token
    embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1) // position embedding
};

static void initialize_kernel(Network* networks) {
    cl_int err;

    err = clGetPlatformIDs(1, &ctx.platform, NULL); CHECK_ERROR(err);
    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, NULL); CHECK_ERROR(err);

    ctx.context = clCreateContext(NULL, 1, &ctx.device, NULL, NULL, &err); CHECK_ERROR(err);

    cl_queue_properties props[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,
        0
    };

    ctx.input_queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);
    ctx.compute_queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);

    size_t kernel_source_size;
    char* kernel_source = get_source_code("kernel_sb.cl", &kernel_source_size);
    ctx.program = clCreateProgramWithSource(ctx.context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    char build_options[1024];

    snprintf(build_options, sizeof(build_options),
        "-D BATCH_SIZE=%d "
        "-D IMG_SIZE=%d "
        "-D PATCH_SIZE=%d "
        "-D CHANNELS=%d "
        "-D EMBED_DIM=%d "
        "-D NUM_HEADS=%d "
        "-D OUTPUT_SIZE=%d "
        "-D NUM_PATCHES=%d "
        "-D TOKENS=%d "
        "-D TOTAL_TOKENS=%d "
        "-D HEAD_DIM=%d "
        "-D QKV_DIM=%d ",
        batch_size,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        num_heads,
        output_size,
        num_patches,
        tokens,
        total_tokens,
        head_dim,
        qkv_dim
    );

    err = clBuildProgram(ctx.program, 1, &ctx.device, build_options, NULL, NULL);
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
    ctx.prepare_kernel = clCreateKernel(ctx.program, "prepare_input", &err); CHECK_ERROR(err);
    ctx.extract_cls_kernel = clCreateKernel(ctx.program, "extract_cls", &err); CHECK_ERROR(err);

	free(kernel_source);
}

void ViT_seq_sb(ImageData* image, Network* networks, float** probabilities) {
    cl_int err;

	initialize_kernel(networks);

    // below : kernel creation, buffer allocation, data transfer, kernel execution, result retrieval, cleanup //////////////

    int token_size = ((img_size / patch_size) * (img_size / patch_size) + 1); // 197

    cl_mem enc_buf[12];

    float* enc_output;

    for (int i = 0; i < 12; i++) {
		enc_buf[i] = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * enc_size, NULL, &err); CHECK_ERROR(err);
    }

    enc_output = (float*)malloc(sizeof(float) * enc_size);

    cl_mem input = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, sizeof(float) * batch_size * in_chans * img_size * img_size, NULL, &err); CHECK_ERROR(err);
	cl_mem output = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * enc_size, NULL, &err); CHECK_ERROR(err);

	cl_mem buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * size[0], NULL, &err); CHECK_ERROR(err);
	cl_mem layer_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * size[2], NULL, &err); CHECK_ERROR(err);

	cl_mem cls_tokens = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, sizeof(float) * batch_size * embed_dim, NULL, &err);
    cl_mem cls_output = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * num_classes, NULL, &err);

    int current_batch_size;
	float image_bytes = sizeof(float) * in_chans * img_size * img_size;

    int steps = 0;
    cl_event input_event = NULL, done_event[2] = { NULL, NULL };

    for (int i = 0; i < image->n; i += batch_size) {
        steps = i % 2;

		if (done_event[steps]) {
			clWaitForEvents(1, &done_event[steps]);

			clReleaseEvent(done_event[steps]);
			done_event[steps] = NULL;
		}

        printf("Processing image %d/%d\n", i + 1, image->n);
        current_batch_size = (image->n - i) < batch_size ? (image->n - i) : batch_size;

        for (int j = 0; j < current_batch_size; j++) {
            cl_event* ptr = (j < current_batch_size - 1) ? NULL : &input_event;

            err = clEnqueueWriteBuffer(ctx.input_queue, input, CL_FALSE, image_bytes * j,
                image_bytes, image[i + j].data, 0, NULL, ptr);
            CHECK_ERROR(err);
        }
           
        clEnqueueBarrierWithWaitList(ctx.compute_queue, 1, &input_event, NULL);

        clReleaseEvent(input_event);
        input_event = NULL;

		// Patch Embedding
        Conv2d(input, buf, networks[1], networks[2]);

		// class token                              ┐
		// class token prepended + patch tokens     │- in one kernel
        // positional encoding                      ┘
		prepare_input(buf, networks[0].buffer, networks[3].buffer, layer_buf);

		Encoder(layer_buf, enc_buf[0],
            networks[4], networks[5], networks[6], networks[7],
            networks[8], networks[9], networks[10], networks[11],
            networks[12], networks[13], networks[14], networks[15]);

		Encoder(enc_buf[0], enc_buf[1],
            networks[16], networks[17], networks[18], networks[19],
            networks[20], networks[21], networks[22], networks[23],
            networks[24], networks[25], networks[26], networks[27]);

		Encoder(enc_buf[1], enc_buf[2],
            networks[28], networks[29], networks[30], networks[31],
            networks[32], networks[33], networks[34], networks[35],
            networks[36], networks[37], networks[38], networks[39]);

		Encoder(enc_buf[2], enc_buf[3],
            networks[40], networks[41], networks[42], networks[43],
            networks[44], networks[45], networks[46], networks[47],
            networks[48], networks[49], networks[50], networks[51]);

		Encoder(enc_buf[3], enc_buf[4],
            networks[52], networks[53], networks[54], networks[55],
            networks[56], networks[57], networks[58], networks[59],
            networks[60], networks[61], networks[62], networks[63]);

		Encoder(enc_buf[4], enc_buf[5],
            networks[64], networks[65], networks[66], networks[67],
            networks[68], networks[69], networks[70], networks[71],
            networks[72], networks[73], networks[74], networks[75]);

		Encoder(enc_buf[5], enc_buf[6],
            networks[76], networks[77], networks[78], networks[79],
            networks[80], networks[81], networks[82], networks[83],
            networks[84], networks[85], networks[86], networks[87]);

		Encoder(enc_buf[6], enc_buf[7],
            networks[88], networks[89], networks[90], networks[91],
            networks[92], networks[93], networks[94], networks[95],
            networks[96], networks[97], networks[98], networks[99]);

		Encoder(enc_buf[7], enc_buf[8],
            networks[100], networks[101], networks[102], networks[103],
            networks[104], networks[105], networks[106], networks[107],
            networks[108], networks[109], networks[110], networks[111]);

		Encoder(enc_buf[8], enc_buf[9],
            networks[112], networks[113], networks[114], networks[115],
            networks[116], networks[117], networks[118], networks[119],
            networks[120], networks[121], networks[122], networks[123]);

		Encoder(enc_buf[9], enc_buf[10],
            networks[124], networks[125], networks[126], networks[127],
            networks[128], networks[129], networks[130], networks[131],
            networks[132], networks[133], networks[134], networks[135]);

		Encoder(enc_buf[10], enc_buf[11],
            networks[136], networks[137], networks[138], networks[139],
            networks[140], networks[141], networks[142], networks[143],
            networks[144], networks[145], networks[146], networks[147]);

        layer_norm(enc_buf[11], output, networks[148], networks[149]);

        err = clSetKernelArg(ctx.extract_cls_kernel, 0, sizeof(cl_mem), &output); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.extract_cls_kernel, 1, sizeof(cl_mem), &cls_tokens); CHECK_ERROR(err);

        size_t global_extract[2] = { 
            (size_t)batch_size, 
            (size_t)embed_dim 
        };

        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.extract_cls_kernel, 2, NULL, global_extract, NULL, 0, NULL, NULL);
        CHECK_ERROR(err);

        linear_layer(cls_tokens, cls_output, batch_size, embed_dim, num_classes, networks[150], networks[151]);

		int classes = num_classes;
        err = clSetKernelArg(ctx.softmax_kernel, 0, sizeof(cl_mem), &cls_output); CHECK_ERROR(err);
		err = clSetKernelArg(ctx.softmax_kernel, 1, sizeof(int), &classes); CHECK_ERROR(err);

        size_t soft_size = current_batch_size;
        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.softmax_kernel, 1, NULL, &soft_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);

		for (int b = 0; b < current_batch_size; b++) {
			cl_event* ptr = (b < current_batch_size - 1) ? NULL : &done_event[steps];

			err = clEnqueueReadBuffer(ctx.compute_queue, cls_output, CL_TRUE, sizeof(float) * num_classes * b, sizeof(float) * num_classes, 
                    probabilities[i + b], 0, NULL, ptr); CHECK_ERROR(err); 
		}
	}

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	free(enc_output);

    clReleaseMemObject(input);
    clReleaseMemObject(buf);
    clReleaseMemObject(layer_buf);
    clReleaseMemObject(cls_output);

    clReleaseKernel(ctx.linear_kernel);
	clReleaseKernel(ctx.gelu_kernel);
	clReleaseKernel(ctx.score_kernel);
	clReleaseKernel(ctx.softmax_kernel);
	clReleaseKernel(ctx.context_kernel);
	clReleaseKernel(ctx.normalize_kernel);
	clReleaseKernel(ctx.conv2d_kernel);
	clReleaseKernel(ctx.add_kernel);
	clReleaseKernel(ctx.prepare_kernel);
	clReleaseKernel(ctx.extract_cls_kernel);

	for (int i = 0; i < 12; i++) {
		clReleaseMemObject(enc_buf[i]);
	}

	clReleaseCommandQueue(ctx.input_queue);
    clReleaseCommandQueue(ctx.compute_queue);
    clReleaseContext(ctx.context);
    clReleaseProgram(ctx.program);
}