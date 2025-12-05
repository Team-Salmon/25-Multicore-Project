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
#define eps 1e-6f

// custom defines
#define batch_size 8

#define li_lws_out 4
#define li_lws_token 64
#define li_tpt 4 // tokens per thread
#define li_opt 16 // output per thread
#define li_tile 32
#define li_stride_in 33
#define li_stride_weight 65

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
    cl_device_id   device;
    cl_context     context;

    cl_command_queue q_input;
    cl_command_queue q_transfer;
    cl_command_queue q_compute;
    cl_command_queue q_output;

    cl_program program;

    cl_kernel k_patch_embed;
    cl_kernel k_pos_emb;

    cl_kernel k_linear;
    cl_kernel k_linear_gelu;
    size_t    lws_linear[2];

    cl_kernel k_attn_score;
    cl_kernel k_softmax;
    cl_kernel k_attn_context;
    cl_kernel k_layernorm;
    cl_kernel k_add;

    cl_kernel k_extract_cls;

    cl_mem d_networks[152];

    cl_mem d_hidden[2];

    cl_mem d_img[2];
    cl_mem d_patch;
    cl_mem d_input_embed;

    cl_mem d_cls_tokens;
    cl_mem d_logits[2];

    cl_mem d_qkv;
    cl_mem d_attn_map;
    cl_mem d_context_vec;

    cl_mem d_mlp_tmp;
    cl_mem d_enc_tmp;

    cl_event  evt_profile;
    cl_event* evt_ptr;
    cl_event evt_transfer[152];
} CLContext;

static CLContext ctx = { 0 };

static void linear_layer(cl_kernel, cl_mem, cl_mem, int, int, int, cl_mem, cl_mem);
static void add(cl_mem, cl_mem, cl_mem, int);

static void init_kernel(Network*);
static void release_kernel();

static void set_size_1d(size_t*, int);
static void set_size_2d(size_t*, int, int);
static void set_size_3d(size_t*, int, int, int);
static void padding_size(size_t*, const size_t*, int);

static void layer_norm(cl_mem input, cl_mem ouput, cl_mem weight, cl_mem bias) {
    cl_int err;

    err = clSetKernelArg(ctx.k_layernorm, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_layernorm, 1, sizeof(cl_mem), &ouput); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_layernorm, 2, sizeof(cl_mem), &weight); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_layernorm, 3, sizeof(cl_mem), &bias); CHECK_ERROR(err);

    size_t gws_layernorm = (size_t)total_tokens;
    size_t lws_layernorm = (size_t)2;
    padding_size(&gws_layernorm, &lws_layernorm, 1);

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_layernorm, 1, NULL, &gws_layernorm, &lws_layernorm, 0, NULL, NULL); CHECK_ERROR(err);
}

static void multihead_attn(cl_mem input, cl_mem output,
    cl_mem in_weight, cl_mem in_bias, cl_mem out_weight, cl_mem out_bias) {

    cl_int err;
    linear_layer(ctx.k_linear, input, ctx.d_qkv, total_tokens, embed_dim, qkv_dim, in_weight, in_bias);

    err = clSetKernelArg(ctx.k_attn_score, 0, sizeof(cl_mem), &ctx.d_qkv); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_attn_score, 1, sizeof(cl_mem), &ctx.d_attn_map); CHECK_ERROR(err);

    size_t lws_attn_score[3] = { ctx.lws_linear[0], ctx.lws_linear[1], 1 };
    size_t gws_attn_score[3] = {
        (size_t)(tokens + li_opt - 1) / li_opt,
        (size_t)(tokens + li_tpt - 1) / li_tpt,
        (size_t)batch_size * num_heads
    };

    padding_size(gws_attn_score, lws_attn_score, 3);

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_attn_score, 3, NULL, gws_attn_score, lws_attn_score, 0, NULL, NULL); CHECK_ERROR(err);

    // Softmax

    int token_size = tokens;
    err = clSetKernelArg(ctx.k_softmax, 0, sizeof(cl_mem), &ctx.d_attn_map); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_softmax, 1, sizeof(int), &token_size); CHECK_ERROR(err);

    size_t lws_softmax = 256;
    size_t total_rows = (size_t)tokens * batch_size * num_heads;
    size_t gws_softmax = total_rows * lws_softmax;

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_softmax, 1, NULL, &gws_softmax, &lws_softmax, 0, NULL, NULL); CHECK_ERROR(err);

    // Context Vector
    err = clSetKernelArg(ctx.k_attn_context, 0, sizeof(cl_mem), &ctx.d_attn_map); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_attn_context, 1, sizeof(cl_mem), &ctx.d_qkv); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_attn_context, 2, sizeof(cl_mem), &ctx.d_context_vec); CHECK_ERROR(err);

    size_t lws_context[3] = { 4, 16, 1 };
    size_t gws_context[3] = {
        (size_t)((tokens + 3) / 4),
        (size_t)head_dim / 4,
        (size_t)batch_size * num_heads
    };
    padding_size(gws_context, lws_context, 3);

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_attn_context, 3, NULL, gws_context, lws_context, 0, NULL, NULL); CHECK_ERROR(err);

    linear_layer(ctx.k_linear, ctx.d_context_vec, output, total_tokens, embed_dim, embed_dim, out_weight, out_bias);
}

static void linear_layer(cl_kernel kernel, cl_mem input, cl_mem output, int token_size, int in_features, int out_features, cl_mem weight, cl_mem bias) {
    cl_int err;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bias); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &token_size); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &in_features); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 6, sizeof(int), &out_features); CHECK_ERROR(err);

    size_t gws[2] = { (out_features + li_opt - 1) / li_opt, (token_size + li_tpt - 1) / li_tpt };
    padding_size(gws, ctx.lws_linear, 2);

    err = clEnqueueNDRangeKernel(ctx.q_compute, kernel, 2, NULL, gws, ctx.lws_linear, 0, NULL, NULL); CHECK_ERROR(err);
}

static void mlp_block(cl_mem input, cl_mem output, cl_mem fc1_weight, cl_mem fc1_bias, cl_mem fc2_weight, cl_mem fc2_bias) {
    linear_layer(ctx.k_linear_gelu, input, ctx.d_mlp_tmp, total_tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias);
    linear_layer(ctx.k_linear, ctx.d_mlp_tmp, output, total_tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias);
}

static void Encoder(cl_mem input, cl_mem output,
    cl_mem ln1_w, cl_mem ln1_b, cl_mem attn_w, cl_mem attn_b, cl_mem attn_out_w, cl_mem attn_out_b,
    cl_mem ln2_w, cl_mem ln2_b, cl_mem mlp1_w, cl_mem mlp1_b, cl_mem mlp2_w, cl_mem mlp2_b) {

    layer_norm(input, ctx.d_enc_tmp, ln1_w, ln1_b);
    multihead_attn(ctx.d_enc_tmp, output, attn_w, attn_b, attn_out_w, attn_out_b);
    add(input, output, output, batch_size * tokens * embed_dim);

    layer_norm(output, ctx.d_enc_tmp, ln2_w, ln2_b);
    mlp_block(ctx.d_enc_tmp, input, mlp1_w, mlp1_b, mlp2_w, mlp2_b);
    add(output, input, output, batch_size * tokens * embed_dim);
}

static void add(cl_mem a, cl_mem b, cl_mem output, int size) {
    cl_int err;

    err = clSetKernelArg(ctx.k_add, 0, sizeof(cl_mem), &a); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_add, 1, sizeof(cl_mem), &b); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_add, 2, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_add, 3, sizeof(int), &size); CHECK_ERROR(err);

    size_t gws = (size_t)size;

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_add, 1, NULL, &gws, NULL, 0, NULL, NULL); CHECK_ERROR(err);
}

static void pos_embedding(cl_mem input, cl_mem cls, cl_mem pos, cl_mem output) {
    cl_int err;

    err = clSetKernelArg(ctx.k_pos_emb, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_pos_emb, 1, sizeof(cl_mem), &cls); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_pos_emb, 2, sizeof(cl_mem), &pos); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_pos_emb, 3, sizeof(cl_mem), &output); CHECK_ERROR(err);

    size_t gws[3] = { embed_dim, tokens, batch_size };

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_pos_emb, 3, NULL, gws, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);
}

static void init_kernel(Network* networks) {
    cl_int err;

    err = clGetPlatformIDs(1, &ctx.platform, NULL); CHECK_ERROR(err);
    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, NULL); CHECK_ERROR(err);

    ctx.context = clCreateContext(NULL, 1, &ctx.device, NULL, NULL, &err); CHECK_ERROR(err);

    cl_queue_properties props[] = { 0 };

    ctx.q_input = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);
	ctx.q_transfer = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);
    ctx.q_compute = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);
	ctx.q_output = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);

    size_t kernel_source_size;
    char* kernel_source = get_source_code("kernel_sb.cl", &kernel_source_size);
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
        "-D QKV_DIM=%d "
        "-D EPS=%f "
        "-D LI_LWS_OUT=%d "
        "-D LI_LWS_TOKEN=%d "
        "-D LI_TPT=%d "
        "-D LI_OPT=%d "
        "-D LI_TILE=%d "
        "-D LI_STRIDE_IN=%d "
        "-D LI_STRIDE_WEIGHT=%d ",
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
        qkv_dim,
        eps,
        li_lws_out,
        li_lws_token,
        li_tpt,
        li_opt,
        li_tile,
        li_stride_in,
        li_stride_weight
    );

    err = clBuildProgram(ctx.program, 1, &ctx.device, build_options, NULL, NULL);
    build_error(ctx.program, ctx.device, err); CHECK_ERROR(err);

    for (int i = 0; i < 152; i++) {
        if (networks[i].data == NULL) continue;
        if (networks[i].size <= 0) continue;

        ctx.d_networks[i] = clCreateBuffer(ctx.context,
            CL_MEM_READ_ONLY,
            sizeof(float) * networks[i].size,
            NULL,
            &err);

        CHECK_ERROR(err);

        err = clEnqueueWriteBuffer(ctx.q_transfer,
            ctx.d_networks[i],
            CL_FALSE, 0,
            sizeof(float) * networks[i].size,
            networks[i].data,
            0, NULL, &ctx.evt_transfer[i]);
        CHECK_ERROR(err);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create Kernels

    ctx.k_patch_embed  = clCreateKernel(ctx.program, "linear_conv2d", &err); CHECK_ERROR(err);
    ctx.k_linear       = clCreateKernel(ctx.program, "linear_layer", &err); CHECK_ERROR(err);
    ctx.k_linear_gelu  = clCreateKernel(ctx.program, "linear_layer", &err); CHECK_ERROR(err);
    ctx.k_attn_score   = clCreateKernel(ctx.program, "attn_score", &err); CHECK_ERROR(err);
    ctx.k_softmax      = clCreateKernel(ctx.program, "softmax", &err); CHECK_ERROR(err);
    ctx.k_attn_context = clCreateKernel(ctx.program, "attn_context", &err); CHECK_ERROR(err);
    ctx.k_layernorm    = clCreateKernel(ctx.program, "layer_norm", &err); CHECK_ERROR(err);
    ctx.k_add          = clCreateKernel(ctx.program, "add", &err); CHECK_ERROR(err);
    ctx.k_pos_emb      = clCreateKernel(ctx.program, "pos_embedding", &err); CHECK_ERROR(err);
    ctx.k_extract_cls  = clCreateKernel(ctx.program, "extract_cls", &err); CHECK_ERROR(err);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Set Initial Args

    err = clSetKernelArg(ctx.k_linear, 7, sizeof(int), &(int){0}); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear_gelu, 7, sizeof(int), &(int){1}); CHECK_ERROR(err);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create Buffers

    ctx.d_img[0] = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, sizeof(float) * batch_size * in_chans * img_size * img_size, NULL, &err); CHECK_ERROR(err);
	ctx.d_img[1] = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, sizeof(float) * batch_size * in_chans * img_size * img_size, NULL, &err); CHECK_ERROR(err);

    ctx.d_patch = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * embed_dim * num_patches, NULL, &err); CHECK_ERROR(err);
    ctx.d_input_embed = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * embed_dim * tokens, NULL, &err); CHECK_ERROR(err);

    ctx.d_hidden[0] = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * enc_size, NULL, &err); CHECK_ERROR(err);
    ctx.d_hidden[1] = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * enc_size, NULL, &err); CHECK_ERROR(err);

    ctx.d_qkv = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens * qkv_dim, NULL, &err); CHECK_ERROR(err);
    ctx.d_context_vec = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens * embed_dim, NULL, &err); CHECK_ERROR(err);
    ctx.d_attn_map = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens * num_heads * tokens, NULL, &err); CHECK_ERROR(err);
    ctx.d_mlp_tmp = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_tokens * hidden_dim, NULL, &err); CHECK_ERROR(err);
    ctx.d_enc_tmp = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * tokens * embed_dim, NULL, &err); CHECK_ERROR(err);

    ctx.d_cls_tokens = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, sizeof(float) * batch_size * embed_dim, NULL, &err); CHECK_ERROR(err);

    ctx.d_logits[0] = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * num_classes, NULL, &err); CHECK_ERROR(err);
	ctx.d_logits[1] = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * num_classes, NULL, &err); CHECK_ERROR(err);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Set work sizes

    set_size_2d(ctx.lws_linear, li_lws_out, li_lws_token);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ctx.evt_ptr = NULL;

    free(kernel_source);
}

static const float image_bytes = sizeof(float) * in_chans * img_size * img_size;

static inline void wait_transfer(int index) {
    if (ctx.evt_transfer[index] == NULL) return;

    cl_int err = clEnqueueBarrierWithWaitList(ctx.q_compute, 1, &ctx.evt_transfer[index], NULL); CHECK_ERROR(err);

    clReleaseEvent(ctx.evt_transfer[index]);
    ctx.evt_transfer[index] = NULL;
}

void ViT_seq_sb(ImageData* image, Network* networks, float** probabilities) {
    cl_int err;

    init_kernel(networks);

    int current_batch_size;
    int steps = 0;

    cl_event evt_input = NULL;
    cl_event evt_done[2] = { NULL, NULL };

    size_t probs_bytes = sizeof(float) * num_classes * image->n;

    cl_mem d_results = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, probs_bytes, NULL, &err); CHECK_ERROR(err);

    float* probs = (float*)clEnqueueMapBuffer(ctx.q_compute, d_results, CL_TRUE, CL_MAP_READ, 0, probs_bytes, 0, NULL, NULL, &err); CHECK_ERROR(err);

    for (int i = 0; i < image->n; i += batch_size) {
        steps = (i / batch_size) % 2;

        if (evt_done[steps] != NULL) { // double buffering
            clWaitForEvents(1, &evt_done[steps]);

            clReleaseEvent(evt_done[steps]);
            evt_done[steps] = NULL;
        }

        current_batch_size = (image->n - i) < batch_size ? (image->n - i) : batch_size;

        for (int j = 0; j < current_batch_size; j++) {
            cl_event* ptr = (j < current_batch_size - 1) ? NULL : &evt_input;

            err = clEnqueueWriteBuffer(ctx.q_input, ctx.d_img[steps], CL_FALSE, image_bytes * j,
                image_bytes, image[i + j].data, 0, NULL, ptr);
            CHECK_ERROR(err);
        }

        clEnqueueBarrierWithWaitList(ctx.q_compute, 1, &evt_input, NULL);

        clReleaseEvent(evt_input);
        evt_input = NULL;

        // patch_embbeding
        wait_transfer(2);
		linear_layer(ctx.k_patch_embed, ctx.d_img[steps], ctx.d_patch, batch_size * num_patches, in_chans * patch_size * patch_size, embed_dim, ctx.d_networks[1], ctx.d_networks[2]);

        wait_transfer(3);
        pos_embedding(ctx.d_patch, ctx.d_networks[0], ctx.d_networks[3], ctx.d_input_embed);

        wait_transfer(15);
        Encoder(ctx.d_input_embed, ctx.d_hidden[0],
            ctx.d_networks[4], ctx.d_networks[5], ctx.d_networks[6], ctx.d_networks[7],
            ctx.d_networks[8], ctx.d_networks[9], ctx.d_networks[10], ctx.d_networks[11],
            ctx.d_networks[12], ctx.d_networks[13], ctx.d_networks[14], ctx.d_networks[15]);

		wait_transfer(27);
        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            ctx.d_networks[16], ctx.d_networks[17], ctx.d_networks[18], ctx.d_networks[19],
            ctx.d_networks[20], ctx.d_networks[21], ctx.d_networks[22], ctx.d_networks[23],
            ctx.d_networks[24], ctx.d_networks[25], ctx.d_networks[26], ctx.d_networks[27]);

		wait_transfer(39);
        Encoder(ctx.d_hidden[1], ctx.d_hidden[0],
            ctx.d_networks[28], ctx.d_networks[29], ctx.d_networks[30], ctx.d_networks[31],
            ctx.d_networks[32], ctx.d_networks[33], ctx.d_networks[34], ctx.d_networks[35],
            ctx.d_networks[36], ctx.d_networks[37], ctx.d_networks[38], ctx.d_networks[39]);

		wait_transfer(51);
        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            ctx.d_networks[40], ctx.d_networks[41], ctx.d_networks[42], ctx.d_networks[43],
            ctx.d_networks[44], ctx.d_networks[45], ctx.d_networks[46], ctx.d_networks[47],
            ctx.d_networks[48], ctx.d_networks[49], ctx.d_networks[50], ctx.d_networks[51]);

		wait_transfer(63);
        Encoder(ctx.d_hidden[1], ctx.d_hidden[0],
            ctx.d_networks[52], ctx.d_networks[53], ctx.d_networks[54], ctx.d_networks[55],
            ctx.d_networks[56], ctx.d_networks[57], ctx.d_networks[58], ctx.d_networks[59],
            ctx.d_networks[60], ctx.d_networks[61], ctx.d_networks[62], ctx.d_networks[63]);

		wait_transfer(75);
        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            ctx.d_networks[64], ctx.d_networks[65], ctx.d_networks[66], ctx.d_networks[67],
            ctx.d_networks[68], ctx.d_networks[69], ctx.d_networks[70], ctx.d_networks[71],
            ctx.d_networks[72], ctx.d_networks[73], ctx.d_networks[74], ctx.d_networks[75]);

		wait_transfer(87);
        Encoder(ctx.d_hidden[1], ctx.d_hidden[0],
            ctx.d_networks[76], ctx.d_networks[77], ctx.d_networks[78], ctx.d_networks[79],
            ctx.d_networks[80], ctx.d_networks[81], ctx.d_networks[82], ctx.d_networks[83],
            ctx.d_networks[84], ctx.d_networks[85], ctx.d_networks[86], ctx.d_networks[87]);

		wait_transfer(99);
        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            ctx.d_networks[88], ctx.d_networks[89], ctx.d_networks[90], ctx.d_networks[91],
            ctx.d_networks[92], ctx.d_networks[93], ctx.d_networks[94], ctx.d_networks[95],
            ctx.d_networks[96], ctx.d_networks[97], ctx.d_networks[98], ctx.d_networks[99]);

		wait_transfer(111);
        Encoder(ctx.d_hidden[1], ctx.d_hidden[0],
            ctx.d_networks[100], ctx.d_networks[101], ctx.d_networks[102], ctx.d_networks[103],
            ctx.d_networks[104], ctx.d_networks[105], ctx.d_networks[106], ctx.d_networks[107],
            ctx.d_networks[108], ctx.d_networks[109], ctx.d_networks[110], ctx.d_networks[111]);

		wait_transfer(123);
        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            ctx.d_networks[112], ctx.d_networks[113], ctx.d_networks[114], ctx.d_networks[115],
            ctx.d_networks[116], ctx.d_networks[117], ctx.d_networks[118], ctx.d_networks[119],
            ctx.d_networks[120], ctx.d_networks[121], ctx.d_networks[122], ctx.d_networks[123]);

		wait_transfer(135);
        Encoder(ctx.d_hidden[1], ctx.d_hidden[0],
            ctx.d_networks[124], ctx.d_networks[125], ctx.d_networks[126], ctx.d_networks[127],
            ctx.d_networks[128], ctx.d_networks[129], ctx.d_networks[130], ctx.d_networks[131],
            ctx.d_networks[132], ctx.d_networks[133], ctx.d_networks[134], ctx.d_networks[135]);

		wait_transfer(147);
        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            ctx.d_networks[136], ctx.d_networks[137], ctx.d_networks[138], ctx.d_networks[139],
            ctx.d_networks[140], ctx.d_networks[141], ctx.d_networks[142], ctx.d_networks[143],
            ctx.d_networks[144], ctx.d_networks[145], ctx.d_networks[146], ctx.d_networks[147]);

		wait_transfer(149);
        layer_norm(ctx.d_hidden[1], ctx.d_hidden[0], ctx.d_networks[148], ctx.d_networks[149]);

        err = clSetKernelArg(ctx.k_extract_cls, 0, sizeof(cl_mem), &ctx.d_hidden[0]); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.k_extract_cls, 1, sizeof(cl_mem), &ctx.d_cls_tokens); CHECK_ERROR(err);

        size_t gws_extract_cls[2] = {
            (size_t)batch_size,
            (size_t)embed_dim
        };

        err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_extract_cls, 2, NULL, gws_extract_cls, NULL, 0, NULL, NULL); CHECK_ERROR(err);

		wait_transfer(151);
        linear_layer(ctx.k_linear, ctx.d_cls_tokens, ctx.d_logits[steps], batch_size, embed_dim, num_classes, ctx.d_networks[150], ctx.d_networks[151]);

        // Softmax

        int classes = num_classes;
        err = clSetKernelArg(ctx.k_softmax, 0, sizeof(cl_mem), &ctx.d_logits[steps]); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.k_softmax, 1, sizeof(int), &classes); CHECK_ERROR(err);

        size_t lws_softmax = 256;
        size_t gws_softmax = (size_t)current_batch_size * lws_softmax;

        err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_softmax, 1, NULL, &gws_softmax, &lws_softmax, 0, NULL, &evt_done[steps]); CHECK_ERROR(err);
		float* probs_ptr = &probs[i * num_classes];
        size_t output_bytes = sizeof(float) * num_classes * current_batch_size;

		clEnqueueBarrierWithWaitList(ctx.q_output, 1, &evt_done[steps], NULL);
		err = clEnqueueReadBuffer(ctx.q_output, ctx.d_logits[steps], CL_FALSE, 0, output_bytes, probs_ptr, 0, NULL, NULL); CHECK_ERROR(err);

        // break; // for test purpose, process only one batch
    }

	for (int s = 0; s < 2; s++) {
        if (evt_done[s]) {
            clWaitForEvents(1, &evt_done[s]);

            clReleaseEvent(evt_done[s]);
            evt_done[s] = NULL;
        }
    }
    clFinish(ctx.q_output);
    for (int k = 0; k < image->n; k++) {
        memcpy(probabilities[k], &probs[k * num_classes], sizeof(float) * num_classes);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    clReleaseMemObject(d_results);

    release_kernel();
}

static void release_kernel() {
    clReleaseMemObject(ctx.d_qkv);
    clReleaseMemObject(ctx.d_context_vec);
    clReleaseMemObject(ctx.d_attn_map);

    clReleaseMemObject(ctx.d_mlp_tmp);

    clReleaseMemObject(ctx.d_img[0]);
	clReleaseMemObject(ctx.d_img[1]);
    clReleaseMemObject(ctx.d_patch);
    clReleaseMemObject(ctx.d_input_embed);
    clReleaseMemObject(ctx.d_cls_tokens);

    clReleaseMemObject(ctx.d_logits[0]);
	clReleaseMemObject(ctx.d_logits[1]);

    clReleaseMemObject(ctx.d_hidden[0]);
	clReleaseMemObject(ctx.d_hidden[1]);

    clReleaseKernel(ctx.k_patch_embed);
    clReleaseKernel(ctx.k_linear);
    clReleaseKernel(ctx.k_attn_score);
    clReleaseKernel(ctx.k_softmax);
    clReleaseKernel(ctx.k_attn_context);
    clReleaseKernel(ctx.k_layernorm);
    clReleaseKernel(ctx.k_add);
    clReleaseKernel(ctx.k_pos_emb);
    clReleaseKernel(ctx.k_extract_cls);

    clReleaseProgram(ctx.program);

    clReleaseCommandQueue(ctx.q_input);
    clReleaseCommandQueue(ctx.q_compute);
    clReleaseCommandQueue(ctx.q_transfer);

    clReleaseContext(ctx.context);
}


// utility functions
/////////////////////////////////////////////////////////////////////////////

static void set_size_1d(size_t* source, int l) {
    source[0] = l;
}

static void set_size_2d(size_t* source, int h, int w) {
    source[0] = h;
    source[1] = w;
}

static void set_size_3d(size_t* source, int d, int h, int w) {
    source[0] = d;
    source[1] = h;
    source[2] = w;
}

static void padding_size(size_t* source, const size_t* ref, int dim) {
    for (cl_uint i = 0; i < dim; ++i) {
        source[i] = (source[i] + ref[i] - 1) / ref[i] * ref[i];
    }
}