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
#define batch_size 4
#define dfl_ls 256 // default local size

#define output_size img_size / patch_size
#define num_patches output_size * output_size
#define tokens (num_patches + 1)
#define total_tokens tokens * batch_size

#define head_dim embed_dim / num_heads
#define qkv_dim embed_dim * 3
#define hidden_dim (int)(embed_dim * mlp_ratio)

#define enc_size tokens * embed_dim

 #define PROFILE_MODE

typedef struct __cl_context {
    cl_platform_id platform;
    cl_device_id   device;
    cl_context     context;

    cl_command_queue q_input;
    cl_command_queue q_compute;

    cl_program program;

    cl_kernel k_patch_embed;
    size_t    gws_patch[3];
    size_t    lws_patch[3];

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

    cl_mem d_hidden[2];

    cl_mem d_img;
    cl_mem d_patch;
    cl_mem d_input_embed;

    cl_mem d_cls_tokens;
    cl_mem d_logits;

    cl_mem d_qkv;
    cl_mem d_attn_map;
    cl_mem d_context_vec;

    cl_mem d_mlp_tmp;
    cl_mem d_enc_tmp;

    cl_event  evt_profile;
    cl_event* evt_ptr;
} CLContext;

static CLContext ctx = { 0 };

static void linear_layer(cl_mem, cl_mem, int, int, int, Network, Network);
static void add(cl_mem, cl_mem, cl_mem, int);

static void init_kernel(Network*);
static void release_kernel();

static void set_size_1d(size_t*, int);
static void set_size_2d(size_t*, int, int);
static void set_size_3d(size_t*, int, int, int);
static void padding_size(size_t*, const size_t*, int);

////////////////////////////////////// ViT function //////////////////////////////////////

// input : (3, 224, 224)
// output : (768, 14, 14)
static void Conv2d(cl_mem input, cl_mem output, Network weight, Network bias) {
    cl_int err;

    err = clSetKernelArg(ctx.k_patch_embed, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_patch_embed, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_patch_embed, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_patch_embed, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_patch_embed, 3, NULL,
        ctx.gws_patch, ctx.lws_patch, 0, NULL, ctx.evt_ptr);
    CHECK_ERROR(err);

#ifdef PROFILE_MODE
    profile_event(*ctx.evt_ptr, "Conv2d");
#endif
}

static void layer_norm(cl_mem input, cl_mem ouput, Network weight, Network bias) {
    cl_int err;

    err = clSetKernelArg(ctx.k_layernorm, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_layernorm, 1, sizeof(cl_mem), &ouput); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_layernorm, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_layernorm, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);

    size_t gws_layernorm = (size_t)total_tokens;

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_layernorm, 1, NULL, &gws_layernorm, NULL, 0, NULL, ctx.evt_ptr); CHECK_ERROR(err);
#ifdef PROFILE_MODE
    profile_event(*ctx.evt_ptr, "LayerNorm");
#endif
}

static void multihead_attn(cl_mem input, cl_mem output,
    Network in_weight, Network in_bias, Network out_weight, Network out_bias) {

    cl_int err;
    linear_layer(input, ctx.d_qkv, total_tokens, embed_dim, qkv_dim, in_weight, in_bias);

    err = clSetKernelArg(ctx.k_attn_score, 0, sizeof(cl_mem), &ctx.d_qkv); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_attn_score, 1, sizeof(cl_mem), &ctx.d_attn_map); CHECK_ERROR(err);

    size_t j_threads = (tokens + 3) / 4;
    size_t lws_attn[3] = { 32, 1, 1};

    size_t gws_attn_score[3] = {
        (size_t)tokens,
		(size_t)j_threads,
        (size_t)batch_size * num_heads
	};
	padding_size(gws_attn_score, lws_attn, 3);

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_attn_score, 3, NULL, gws_attn_score, lws_attn, 0, NULL, ctx.evt_ptr); CHECK_ERROR(err);
#ifdef PROFILE_MODE
    profile_event(*ctx.evt_ptr, "Attention Score");
#endif

    // Softmax

    int token_size = tokens;
    err = clSetKernelArg(ctx.k_softmax, 0, sizeof(cl_mem), &ctx.d_attn_map); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_softmax, 1, sizeof(int), &token_size); CHECK_ERROR(err);

    size_t gws_softmax = (size_t)tokens * batch_size * num_heads;

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_softmax, 1, NULL, &gws_softmax, NULL, 0, NULL, ctx.evt_ptr); CHECK_ERROR(err);
#ifdef PROFILE_MODE
    profile_event(*ctx.evt_ptr, "Softmax");
#endif

    // Context Vector
    err = clSetKernelArg(ctx.k_attn_context, 0, sizeof(cl_mem), &ctx.d_attn_map); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_attn_context, 1, sizeof(cl_mem), &ctx.d_qkv); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_attn_context, 2, sizeof(cl_mem), &ctx.d_context_vec); CHECK_ERROR(err);

    size_t gws[3] = {
        (size_t)tokens,
        (size_t)head_dim / 4,
        (size_t)batch_size * num_heads
    };

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_attn_context, 3, NULL, gws, NULL, 0, NULL, ctx.evt_ptr); CHECK_ERROR(err);
#ifdef PROFILE_MODE
    profile_event(*ctx.evt_ptr, "Context Vector");
#endif

    linear_layer(ctx.d_context_vec, output, total_tokens, embed_dim, embed_dim, out_weight, out_bias);
}

static void linear_layer(cl_mem input, cl_mem output, int token_size, int in_features, int out_features, Network weight, Network bias) {
    cl_int err;

    err = clSetKernelArg(ctx.k_linear, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear, 4, sizeof(int), &token_size); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear, 5, sizeof(int), &in_features); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear, 6, sizeof(int), &out_features); CHECK_ERROR(err);

    size_t gws[2] = { token_size, out_features };

    padding_size(gws, ctx.lws_linear, 2);

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_linear, 2, NULL, gws, ctx.lws_linear, 0, NULL, ctx.evt_ptr); CHECK_ERROR(err);
#ifdef PROFILE_MODE
    profile_event(*ctx.evt_ptr, "Linear Layer");
#endif
}

static void linear_gelu_layer(cl_mem input, cl_mem output, int token_size, int in_features, int out_features, Network weight, Network bias) {
    cl_int err;

    err = clSetKernelArg(ctx.k_linear_gelu, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear_gelu, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear_gelu, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear_gelu, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear_gelu, 4, sizeof(int), &token_size); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear_gelu, 5, sizeof(int), &in_features); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_linear_gelu, 6, sizeof(int), &out_features); CHECK_ERROR(err);

    size_t gws[2] = { token_size, out_features };

    padding_size(gws, ctx.lws_linear, 2);

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_linear_gelu, 2, NULL, gws, ctx.lws_linear, 0, NULL, ctx.evt_ptr); CHECK_ERROR(err);
#ifdef PROFILE_MODE
    profile_event(*ctx.evt_ptr, "Linear-GELU Layer");
#endif
}

static void mlp_block(cl_mem input, cl_mem output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias) {
    linear_gelu_layer(input, ctx.d_mlp_tmp, total_tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias);
    linear_layer(ctx.d_mlp_tmp, output, total_tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias);
}

static void Encoder(cl_mem input, cl_mem output,
    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b) {

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

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_add, 1, NULL, &gws, NULL, 0, NULL, ctx.evt_ptr); CHECK_ERROR(err);
#ifdef PROFILE_MODE
    profile_event(*ctx.evt_ptr, "Add");
#endif
}

static void pos_embedding(cl_mem input, cl_mem cls, cl_mem pos, cl_mem output) {
    cl_int err;

    err = clSetKernelArg(ctx.k_pos_emb, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_pos_emb, 1, sizeof(cl_mem), &cls); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_pos_emb, 2, sizeof(cl_mem), &pos); CHECK_ERROR(err);
    err = clSetKernelArg(ctx.k_pos_emb, 3, sizeof(cl_mem), &output); CHECK_ERROR(err);

    size_t gws[3] = {
        (size_t)embed_dim,
        (size_t)tokens,
        (size_t)batch_size
    };

    err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_pos_emb, 3, NULL, gws, NULL, 0, NULL, ctx.evt_ptr);
    CHECK_ERROR(err);

#ifdef PROFILE_MODE
    profile_event(*ctx.evt_ptr, "Prepare Input");
#endif
}

static void init_kernel(Network* networks) {
    cl_int err;

    err = clGetPlatformIDs(1, &ctx.platform, NULL); CHECK_ERROR(err);
    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, NULL); CHECK_ERROR(err);

    ctx.context = clCreateContext(NULL, 1, &ctx.device, NULL, NULL, &err); CHECK_ERROR(err);

    cl_queue_properties props[] = {
#ifdef PROFILE_MODE
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,
#endif
        0
    };

    ctx.q_input = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);
    ctx.q_compute = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);

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
        "-D QKV_DIM=%d "
        "-D DFL_LS=%d ",
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
        dfl_ls
    );

    err = clBuildProgram(ctx.program, 1, &ctx.device, build_options, NULL, NULL);
    build_error(ctx.program, ctx.device, err); CHECK_ERROR(err);

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

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create Kernels

    ctx.k_patch_embed = clCreateKernel(ctx.program, "patch_embedding", &err); CHECK_ERROR(err);
    ctx.k_linear = clCreateKernel(ctx.program, "linear", &err); CHECK_ERROR(err);
    ctx.k_linear_gelu = clCreateKernel(ctx.program, "linear_gelu", &err); CHECK_ERROR(err);

    ctx.k_attn_score = clCreateKernel(ctx.program, "attn_score", &err); CHECK_ERROR(err);
    ctx.k_softmax = clCreateKernel(ctx.program, "softmax", &err); CHECK_ERROR(err);
    ctx.k_attn_context = clCreateKernel(ctx.program, "attn_context", &err); CHECK_ERROR(err);
    ctx.k_layernorm = clCreateKernel(ctx.program, "layer_norm", &err); CHECK_ERROR(err);
    ctx.k_add = clCreateKernel(ctx.program, "add", &err); CHECK_ERROR(err);
    ctx.k_pos_emb = clCreateKernel(ctx.program, "pos_embedding", &err); CHECK_ERROR(err);
    ctx.k_extract_cls = clCreateKernel(ctx.program, "extract_cls", &err); CHECK_ERROR(err);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create Buffers

    ctx.d_img = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY, sizeof(float) * batch_size * in_chans * img_size * img_size, NULL, &err); CHECK_ERROR(err);
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
    ctx.d_logits = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * num_classes, NULL, &err); CHECK_ERROR(err);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Set work sizes

    set_size_3d(ctx.gws_patch, embed_dim, num_patches, batch_size);
    set_size_3d(ctx.lws_patch, 4, 4, 4);
    padding_size(ctx.gws_patch, ctx.lws_patch, 3);

    set_size_2d(ctx.lws_linear, 4, 64);

#ifdef PROFILE_MODE
    ctx.evt_ptr = &ctx.evt_profile;
#else
    ctx.evt_ptr = NULL;
#endif

    free(kernel_source);
}

static const float image_bytes = sizeof(float) * in_chans * img_size * img_size;

void ViT_seq_ZZani(ImageData* image, Network* networks, float** probabilities) {
    cl_int err;

    init_kernel(networks);

    int current_batch_size;
    int steps = 0;

    cl_event evt_input = NULL;
    cl_event evt_done[2] = { NULL, NULL };

    init_profiler();

    for (int i = 0; i < image->n; i += batch_size) {
        steps = i % 2;

        if (evt_done[steps] != NULL) { // double buffering
            clWaitForEvents(1, &evt_done[steps]);

            clReleaseEvent(evt_done[steps]);
            evt_done[steps] = NULL;
        }

        printf("Processing image %d/%d\n", i + 1, image->n);
        current_batch_size = (image->n - i) < batch_size ? (image->n - i) : batch_size;

        for (int j = 0; j < current_batch_size; j++) {
            cl_event* ptr = (j < current_batch_size - 1) ? NULL : &evt_input;

            err = clEnqueueWriteBuffer(ctx.q_input, ctx.d_img, CL_FALSE, image_bytes * j,
                image_bytes, image[i + j].data, 0, NULL, ptr);
            CHECK_ERROR(err);
        }

        clEnqueueBarrierWithWaitList(ctx.q_compute, 1, &evt_input, NULL);

        clReleaseEvent(evt_input);
        evt_input = NULL;

        Conv2d(ctx.d_img, ctx.d_patch, networks[1], networks[2]);

        pos_embedding(ctx.d_patch, networks[0].buffer, networks[3].buffer, ctx.d_input_embed);

        Encoder(ctx.d_input_embed, ctx.d_hidden[0],
            networks[4], networks[5], networks[6], networks[7],
            networks[8], networks[9], networks[10], networks[11],
            networks[12], networks[13], networks[14], networks[15]);

        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            networks[16], networks[17], networks[18], networks[19],
            networks[20], networks[21], networks[22], networks[23],
            networks[24], networks[25], networks[26], networks[27]);

        Encoder(ctx.d_hidden[1], ctx.d_hidden[0],
            networks[28], networks[29], networks[30], networks[31],
            networks[32], networks[33], networks[34], networks[35],
            networks[36], networks[37], networks[38], networks[39]);

        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            networks[40], networks[41], networks[42], networks[43],
            networks[44], networks[45], networks[46], networks[47],
            networks[48], networks[49], networks[50], networks[51]);

        Encoder(ctx.d_hidden[1], ctx.d_hidden[0],
            networks[52], networks[53], networks[54], networks[55],
            networks[56], networks[57], networks[58], networks[59],
            networks[60], networks[61], networks[62], networks[63]);

        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            networks[64], networks[65], networks[66], networks[67],
            networks[68], networks[69], networks[70], networks[71],
            networks[72], networks[73], networks[74], networks[75]);

        Encoder(ctx.d_hidden[1], ctx.d_hidden[0],
            networks[76], networks[77], networks[78], networks[79],
            networks[80], networks[81], networks[82], networks[83],
            networks[84], networks[85], networks[86], networks[87]);

        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            networks[88], networks[89], networks[90], networks[91],
            networks[92], networks[93], networks[94], networks[95],
            networks[96], networks[97], networks[98], networks[99]);

        Encoder(ctx.d_hidden[1], ctx.d_hidden[0],
            networks[100], networks[101], networks[102], networks[103],
            networks[104], networks[105], networks[106], networks[107],
            networks[108], networks[109], networks[110], networks[111]);

        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            networks[112], networks[113], networks[114], networks[115],
            networks[116], networks[117], networks[118], networks[119],
            networks[120], networks[121], networks[122], networks[123]);

        Encoder(ctx.d_hidden[1], ctx.d_hidden[0],
            networks[124], networks[125], networks[126], networks[127],
            networks[128], networks[129], networks[130], networks[131],
            networks[132], networks[133], networks[134], networks[135]);

        Encoder(ctx.d_hidden[0], ctx.d_hidden[1],
            networks[136], networks[137], networks[138], networks[139],
            networks[140], networks[141], networks[142], networks[143],
            networks[144], networks[145], networks[146], networks[147]);

        layer_norm(ctx.d_hidden[1], ctx.d_hidden[0], networks[148], networks[149]);

        err = clSetKernelArg(ctx.k_extract_cls, 0, sizeof(cl_mem), &ctx.d_hidden[0]); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.k_extract_cls, 1, sizeof(cl_mem), &ctx.d_cls_tokens); CHECK_ERROR(err);

        size_t gws_extract_cls[2] = {
            (size_t)batch_size,
            (size_t)embed_dim
        };

        err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_extract_cls, 2, NULL, gws_extract_cls, NULL, 0, NULL, ctx.evt_ptr); CHECK_ERROR(err);
#ifdef PROFILE_MODE
        profile_event(*ctx.evt_ptr, "Extract CLS Token");
#endif

        linear_layer(ctx.d_cls_tokens, ctx.d_logits, batch_size, embed_dim, num_classes, networks[150], networks[151]);

        int classes = num_classes;
        err = clSetKernelArg(ctx.k_softmax, 0, sizeof(cl_mem), &ctx.d_logits); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.k_softmax, 1, sizeof(int), &classes); CHECK_ERROR(err);

        size_t global_size_softmax = current_batch_size;

        err = clEnqueueNDRangeKernel(ctx.q_compute, ctx.k_softmax, 1, NULL, &global_size_softmax, NULL, 0, NULL, ctx.evt_ptr); CHECK_ERROR(err);
#ifdef PROFILE_MODE
        profile_event(*ctx.evt_ptr, "Output Softmax");
#endif

        for (int b = 0; b < current_batch_size; b++) {
            cl_event* ptr = (b < current_batch_size - 1) ? NULL : &evt_done[steps];

            err = clEnqueueReadBuffer(ctx.q_compute, ctx.d_logits, CL_FALSE, sizeof(float) * num_classes * b, sizeof(float) * num_classes,
                probabilities[i + b], 0, NULL, ptr); CHECK_ERROR(err);
        }

        // break; // for test purpose, process only one batch
    }

    if (evt_done[steps]) {
        clWaitForEvents(1, &evt_done[steps]);
        clReleaseEvent(evt_done[steps]);
        evt_done[steps] = NULL;
    }

#ifdef PROFILE_MODE
    clFinish(ctx.q_compute);
    print_profiler_stats();
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    release_kernel();
}

static void release_kernel() {
    clReleaseMemObject(ctx.d_qkv);
    clReleaseMemObject(ctx.d_context_vec);
    clReleaseMemObject(ctx.d_attn_map);

    clReleaseMemObject(ctx.d_mlp_tmp);

    clReleaseMemObject(ctx.d_img);
    clReleaseMemObject(ctx.d_patch);
    clReleaseMemObject(ctx.d_input_embed);
    clReleaseMemObject(ctx.d_cls_tokens);
    clReleaseMemObject(ctx.d_logits);

    for (int i = 0; i < 2; i++) {
        clReleaseMemObject(ctx.d_hidden[i]);
    }

    clReleaseKernel(ctx.k_patch_embed);
    clReleaseKernel(ctx.k_linear);
    // clReleaseKernel(ctx.gelu_kernel);
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