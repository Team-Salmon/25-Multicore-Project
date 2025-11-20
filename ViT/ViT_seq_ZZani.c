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
#define batch_size 2

#define output_size img_size / patch_size
#define num_patches output_size * output_size
#define tokens (num_patches + 1)
#define total_tokens tokens * batch_size

#define head_dim embed_dim / num_heads
#define qkv_dim embed_dim * 3
#define hidden_dim (int)(embed_dim * mlp_ratio)

#define enc_size tokens * embed_dim

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
// 최적화: Patch Embedding의 Conv2d 연산을 OpenCL 커널로 구현하여 GPU에서 병렬 처리합니다.
// 각 출력 채널과 패치에 대한 연산을 독립적으로 수행하여 가속합니다.
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

// 최적화: Layer Normalization을 OpenCL 커널로 구현하여 GPU에서 병렬 처리합니다.
// 각 토큰별로 정규화 연산이 독립적으로 수행되므로, 전체 토큰에 대해 병렬화하여 처리 속도를 높입니다.
static void layer_norm(cl_mem input, cl_mem ouput, Network weight, Network bias) {
    cl_int err;

    err = clSetKernelArg(ctx.normalize_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.normalize_kernel, 1, sizeof(cl_mem), &ouput); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.normalize_kernel, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.normalize_kernel, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);

    size_t global_work_size = (size_t)total_tokens;
    err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.normalize_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

// 최적화: Multi-Head Attention의 복잡한 행렬 연산들을 여러 OpenCL 커널로 분할하여 GPU에서 순차적으로 실행합니다.
// Q, K, V 생성, Attention Score 계산, Softmax, Context Vector 계산 등 각 단계를 GPU 내에서 처리하여
// CPU-GPU 간의 데이터 전송을 최소화하고 병렬 처리 효율을 극대화합니다.
static void multihead_attn(cl_mem input, cl_mem output,
    Network in_weight, Network in_bias, Network out_weight, Network out_bias) {

    cl_int err;

    int total_token = tokens * batch_size;

    cl_mem QKV = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_token * qkv_dim, NULL, &err); CHECK_ERROR(err);

    linear_layer(input, QKV, total_token, embed_dim, qkv_dim, in_weight, in_bias);

    cl_mem attn_out = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_token * embed_dim, NULL, &err); CHECK_ERROR(err);
    cl_mem scores = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_token * tokens, NULL, &err); CHECK_ERROR(err);

    // 각 head별로 attn 수행
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        // Attention Score 계산
        err = clSetKernelArg(ctx.score_kernel, 0, sizeof(cl_mem), &QKV); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 1, sizeof(cl_mem), &scores); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 2, sizeof(int), &head_offset); CHECK_ERROR(err);

        size_t global_size_score[3] = {
            (size_t)tokens,
            (size_t)tokens,
            (size_t)batch_size
        };

        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.score_kernel, 3, NULL, global_size_score, NULL, 0, NULL, NULL); CHECK_ERROR(err);

        // Softmax 적용
        int token_size = tokens;
        err = clSetKernelArg(ctx.softmax_kernel, 0, sizeof(cl_mem), &scores); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.softmax_kernel, 1, sizeof(int), &token_size); CHECK_ERROR(err);

        size_t global_size_softmax = (size_t)tokens * batch_size;
        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.softmax_kernel, 1, NULL, &global_size_softmax, NULL, 0, NULL, NULL); CHECK_ERROR(err);

        // Context Vector
        err = clSetKernelArg(ctx.context_kernel, 0, sizeof(cl_mem), &scores); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 1, sizeof(cl_mem), &QKV); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 2, sizeof(cl_mem), &attn_out); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.context_kernel, 3, sizeof(int), &head_offset); CHECK_ERROR(err);

        size_t global_size_context[3] = {
            (size_t)tokens,
            (size_t)head_dim,
            (size_t)batch_size
        };

        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.context_kernel, 3, NULL, global_size_context, NULL, 0, NULL, NULL); CHECK_ERROR(err);
    }

    linear_layer(attn_out, output, total_token, embed_dim, embed_dim, out_weight, out_bias);

    clReleaseMemObject(QKV);
    clReleaseMemObject(scores);
    clReleaseMemObject(attn_out);
}

// 최적화: GELU 활성화 함수를 OpenCL 커널로 구현하여 요소별(element-wise) 연산을 GPU에서 병렬 처리합니다.
static void gelu_activation(cl_mem input, int size) {
    cl_int err;

    err = clSetKernelArg(ctx.gelu_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.gelu_kernel, 1, sizeof(int), &size); CHECK_ERROR(err);

    size_t global_work_size = size;

    err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.gelu_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}

// 최적화: 행렬 곱셈과 덧셈으로 구성된 Linear Layer(FC Layer)를 OpenCL 커널로 구현하여 GPU에서 병렬 처리합니다.
// 각 출력 뉴런에 대한 계산을 독립적으로 병렬 수행하여 연산 속도를 크게 향상시킵니다.
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

// 최적화: MLP 블록 전체를 GPU 내에서 처리하도록 구성합니다.
// 두 개의 Linear Layer와 GELU 활성화 함수를 연속적으로 GPU에서 실행하며,
// 중간 결과를 GPU 메모리에 유지하여 불필요한 데이터 전송을 방지합니다.
static void mlp_block(cl_mem input, cl_mem output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias) {
    cl_int err;

    cl_mem fc1_out = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens * hidden_dim, NULL, &err); CHECK_ERROR(err);

    linear_layer(input, fc1_out, total_tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias);
    gelu_activation(fc1_out, total_tokens * hidden_dim);
    linear_layer(fc1_out, output, total_tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias);

    clReleaseMemObject(fc1_out);
}

// 최적화: 하나의 인코더 블록을 구성하는 모든 연산(LayerNorm, Multi-Head Attention, Residual Connection, MLP)을
// GPU 내에서 완결되도록 파이프라인화합니다. 중간 결과물(buf)을 GPU 버퍼에 저장하고 재사용하여 CPU와의 통신 오버헤드를 제거합니다.
static void Encoder(cl_mem input, cl_mem output, cl_mem tmp_buf,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b) {

    int total_elements = batch_size * tokens * embed_dim;
    size_t buffer_bytes = sizeof(float) * total_elements;

    cl_int err;

    layer_norm(input, tmp_buf, ln1_w, ln1_b);
    multihead_attn(tmp_buf, output, attn_w, attn_b, attn_out_w, attn_out_b);
    add(input, output, output, total_elements);

	err = clEnqueueCopyBuffer(ctx.compute_queue, output, tmp_buf, 0, 0, buffer_bytes, 0, NULL, NULL);
    layer_norm(input,tmp_buf, ln2_w, ln2_b);
    mlp_block(tmp_buf, output, mlp1_w, mlp1_b, mlp2_w, mlp2_b);
    add(input, output, output, total_elements);

}

// 최적화: Residual Connection에 사용되는 요소별 덧셈 연산을 OpenCL 커널로 구현하여 GPU에서 병렬 처리합니다.
static void add(cl_mem a, cl_mem b, cl_mem output, int size) {
    cl_int err;

    err = clSetKernelArg(ctx.add_kernel, 0, sizeof(cl_mem), &a); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.add_kernel, 1, sizeof(cl_mem), &b); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.add_kernel, 2, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.add_kernel, 3, sizeof(int), &size); CHECK_ERROR(err);

    size_t global_work_size = (size_t)size;
    err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.add_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

// 최적화: Flatten, Transpose, Class Token 추가, Positional Embedding 덧셈 등
// 순차적으로 처리되던 입력 데이터 준비 과정을 단일 OpenCL 커널로 통합(Kernel Fusion)했습니다.
// 이를 통해 여러 커널 호출에 따른 오버헤드를 줄이고, 메모리 접근을 최적화하여 데이터 준비 단계를 가속합니다.
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

// 최적화: OpenCL 환경 설정, 커널 컴파일 및 가중치 버퍼 생성을 담당합니다.
// 가중치 데이터를 미리 GPU 메모리에 복사(CL_MEM_COPY_HOST_PTR)하여,
// 연산 중 발생하는 데이터 전송 지연을 최소화합니다.
static void initialize_kernel(Network* networks) {
    cl_int err;

    err = clGetPlatformIDs(1, &ctx.platform, NULL); CHECK_ERROR(err);
    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, NULL); CHECK_ERROR(err);

    ctx.context = clCreateContext(NULL, 1, &ctx.device, NULL, NULL, &err); CHECK_ERROR(err);

    cl_queue_properties props[] = { 0 };

    ctx.input_queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);
    ctx.compute_queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);

    size_t kernel_source_size;
    char* kernel_source = get_source_code("kernel_ZZani.cl", &kernel_source_size);
    ctx.program = clCreateProgramWithSource(ctx.context, 1, (const char**)&kernel_source, &kernel_source_size, &err); CHECK_ERROR(err);

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

static const float image_bytes = sizeof(float) * in_chans * img_size * img_size;

// 최적화: 전체 ViT 추론 과정을 GPU 중심으로 재구성합니다.
// Double Buffering을 적용하여 데이터 전송(input_queue)과 커널 실행(compute_queue)을 중첩시켜 I/O 대기 시간을 최소화합니다.
// 또한, 각 커널 실행 후 clFinish()를 호출하여 정확한 동기식 시간 측정을 보장합니다.
void ViT_seq_ZZani(ImageData* image, Network* networks, float** probabilities) {
    cl_int err;

    initialize_kernel(networks);

    const size_t enc_buf_size = sizeof(float) * batch_size * embed_dim;
	cl_mem enc_buf_a = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, enc_buf_size, NULL, &err); CHECK_ERROR(err);
	cl_mem enc_buf_b = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, enc_buf_size, NULL, &err); CHECK_ERROR(err);
	cl_mem enc_tmp_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, enc_buf_size, NULL, &err); CHECK_ERROR(err);

    cl_mem input = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, sizeof(float) * batch_size * in_chans * img_size * img_size, NULL, &err); CHECK_ERROR(err);
    cl_mem output = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * enc_size, NULL, &err); CHECK_ERROR(err);

    cl_mem buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * embed_dim * num_patches, NULL, &err); CHECK_ERROR(err);
    cl_mem layer_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * embed_dim * tokens, NULL, &err); CHECK_ERROR(err);

    cl_mem cls_tokens = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, sizeof(float) * batch_size * embed_dim, NULL, &err);
    cl_mem cls_output = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * num_classes, NULL, &err);

    int current_batch_size;
    int steps = 0;

    cl_event input_event = NULL, done_event[2] = { NULL, NULL };
    
    for (int i = 0; i < image->n; i += batch_size) {
        steps = i % 2;

        if (done_event[steps]) { // double buffering
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
        start_timer();
        Conv2d(input, buf, networks[1], networks[2]);
        clFinish(ctx.compute_queue);
        stop_timer("  - Conv2d");

        // class token, patch tokens, positional encoding을 한번에 처리
        start_timer();
        prepare_input(buf, networks[0].buffer, networks[3].buffer, layer_buf);
        clFinish(ctx.compute_queue);
        stop_timer("  - Prepare Input (cls_token + pos_emb)");

		cl_mem* enc_input = &enc_buf_a;
		cl_mem* enc_output = &enc_buf_b;
        for (int l = 0; l < depth; ++l)
        {
			char timer_msg[64];
			sprintf(timer_msg, "  - Encoder Layer %d", l + 1);

			start_timer();
			Encoder(*enc_input, *enc_output, enc_tmp_buf,
				networks[4 + l * 12], networks[5 + l * 12], networks[6 + l * 12], networks[7 + l * 12],
				networks[8 + l * 12], networks[9 + l * 12], networks[10 + l * 12], networks[11 + l * 12],
				networks[12 + l * 12], networks[13 + l * 12], networks[14 + l * 12], networks[15 + l * 12]);
			clFinish(ctx.compute_queue);
			stop_timer(timer_msg);

			// swap input and output buffers for next layer
			cl_mem* temp = enc_input;
			enc_input = enc_output;
			enc_output = temp;
        }

		cl_mem final_enc_buf = *enc_input;

        start_timer();
        layer_norm(final_enc_buf, output, networks[148], networks[149]);
        clFinish(ctx.compute_queue);
        stop_timer("  - Final Layer Norm");

        start_timer();
        err = clSetKernelArg(ctx.extract_cls_kernel, 0, sizeof(cl_mem), &output); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.extract_cls_kernel, 1, sizeof(cl_mem), &cls_tokens); CHECK_ERROR(err);

        size_t global_size_extract_cls[2] = {
            (size_t)batch_size,
            (size_t)embed_dim
        };

        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.extract_cls_kernel, 2, NULL, global_size_extract_cls, NULL, 0, NULL, NULL); CHECK_ERROR(err);
        clFinish(ctx.compute_queue);
        stop_timer("  - Extract Class Token");

        start_timer();
        linear_layer(cls_tokens, cls_output, batch_size, embed_dim, num_classes, networks[150], networks[151]);
        clFinish(ctx.compute_queue);
        stop_timer("  - Classification Head");

        start_timer();
        int classes = num_classes;
        err = clSetKernelArg(ctx.softmax_kernel, 0, sizeof(cl_mem), &cls_output); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.softmax_kernel, 1, sizeof(int), &classes); CHECK_ERROR(err);

        size_t global_size_softmax = current_batch_size;
        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.softmax_kernel, 1, NULL, &global_size_softmax, NULL, 0, NULL, NULL); CHECK_ERROR(err);
        clFinish(ctx.compute_queue);
        stop_timer("  - Softmax");

        for (int b = 0; b < current_batch_size; b++) {
            cl_event* ptr = (b < current_batch_size - 1) ? NULL : &done_event[steps];

            err = clEnqueueReadBuffer(ctx.compute_queue, cls_output, CL_TRUE, sizeof(float) * num_classes * b, sizeof(float) * num_classes,
                probabilities[i + b], 0, NULL, ptr); CHECK_ERROR(err);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    clReleaseMemObject(input);
    clReleaseMemObject(buf);
    clReleaseMemObject(layer_buf);
    clReleaseMemObject(cls_output);
	clReleaseMemObject(enc_buf_a);
	clReleaseMemObject(enc_buf_b);
	clReleaseMemObject(enc_tmp_buf);

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

    clReleaseCommandQueue(ctx.input_queue);
    clReleaseCommandQueue(ctx.compute_queue);
    clReleaseContext(ctx.context);
    clReleaseProgram(ctx.program);
}