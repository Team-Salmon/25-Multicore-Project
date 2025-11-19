//#pragma warning(disable : 4996)
//#include <stdio.h>
//#include <stdlib.h>
//#include <stdint.h>
//#include <string.h>
//#include <math.h>
//#include <time.h>
//#include "Network.h"
//#include "debug.h"
//#include "ViT_seq.h"
//#define img_size 224
//#define patch_size 16
//#define in_chans 3
//#define num_classes 1000
//#define embed_dim 768
//#define depth 12
//#define num_heads 12
//#define mlp_ratio 4.0
//#define dropout 0.0
//#define attn_dropout 0.0
//#define drop_path_rate 0.0
//#define eps 1e-6
//
//// custom defines
//#define batch_size 4
//
//#define output_size img_size / patch_size
//#define num_patches output_size * output_size
//#define tokens (num_patches + 1)
//#define total_tokens tokens * batch_size
//
//#define head_dim embed_dim / num_heads
//#define qkv_dim embed_dim * 3
//#define hidden_dim (int)(embed_dim * mlp_ratio)
//
//#define enc_size tokens * embed_dim
//
//typedef struct __cl_context {
//    cl_platform_id platform;
//    cl_device_id device;
//    cl_context context;
//
//    cl_command_queue input_queue;
//	cl_command_queue compute_queue;
//
//    cl_program program;
//
//	cl_kernel conv2d_kernel;
//    cl_kernel linear_kernel;
//    cl_kernel gelu_kernel;
//    cl_kernel score_kernel;
//    cl_kernel softmax_kernel;
//	cl_kernel context_kernel;
//    cl_kernel normalize_kernel;
//    cl_kernel add_kernel;
//	cl_kernel prepare_kernel;
//	cl_kernel extract_cls_kernel;
//} CLContext;
//
//static CLContext ctx = { 0 };
//
//static void linear_layer(cl_mem input, cl_mem output, int token_size, int in_features, int out_features, Network weight, Network bias);
//static void add(cl_mem a, cl_mem b, cl_mem output, int size);
//
//////////////////////////////////////// ViT function //////////////////////////////////////
//
//// input : (3, 224, 224)
//// output : (768, 14, 14)
//
//static void Conv2d(cl_mem* input, cl_mem* output, Network weight, Network bias) {
//    cl_int err;
//
//    err = clSetKernelArg(ctx.conv2d_kernel, 0, sizeof(cl_mem), &input); CHECK_ERROR(err);
//    err = clSetKernelArg(ctx.conv2d_kernel, 1, sizeof(cl_mem), &output); CHECK_ERROR(err);
//    err = clSetKernelArg(ctx.conv2d_kernel, 2, sizeof(cl_mem), &weight.buffer); CHECK_ERROR(err);
//    err = clSetKernelArg(ctx.conv2d_kernel, 3, sizeof(cl_mem), &bias.buffer); CHECK_ERROR(err);
//
//    size_t global_work_size[3] = {
//        (size_t)embed_dim,
//        (size_t)num_patches,
//        (size_t)batch_size
//    };
//
//    err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.conv2d_kernel, 3, NULL, global_work_size, NULL, 0, NULL, NULL); CHECK_ERROR(err);
//}
//
//// input : (768, 14, 14)
//// output : (num_patches, embed_dim) -> (196, 768)
//
//static void flatten_transpose(float* input, float* output) {
//    int output_size = img_size / patch_size;
//    int num_patches = output_size * output_size;
//
//    // 각 공간 위치(oh, ow)를 하나의 패치로 취급하여 patch index 계산
//    for (int oh = 0; oh < output_size; oh++) {
//        for (int ow = 0; ow < output_size; ow++) {
//            int patch_idx = oh * output_size + ow;
//            for (int oc = 0; oc < embed_dim; oc++) {
//                // 기존 입력은 (oc, oh, ow)
//                int idx_input = (oc * output_size + oh) * output_size + ow;
//                // 원하는 출력은 (patch_idx, oc)
//                int idx_output = patch_idx * embed_dim + oc;
//                output[idx_output] = input[idx_input];
//                //printf("%f ",output[idx_output]);
//            }
//        }
//    }
//}
//
//static void class_token(float* patch_tokens, float* final_tokens, Network cls_tk) {
//    // 이미지의 패치 수 계산: output_size = img_size / patch_size, num_patches = output_size^2
//    int output_size = img_size / patch_size;
//    int num_patches = output_size * output_size;
//
//    // 1. 첫 번째 토큰에 class token 복사 (networks[0].data에 저장됨, embed_dim 길이)
//    for (int j = 0; j < embed_dim; j++) {
//        final_tokens[j] = cls_tk.data[j];
//    }
//
//    // 2. 이후 patch_tokens를 이어붙임
//   // final_tokens의 인덱스 embed_dim부터, patch_tokens 전체(embed_dim * num_patches) 복사
//    memcpy(final_tokens + embed_dim, patch_tokens, sizeof(float) * embed_dim * num_patches);
//
//    int total_tokens = num_patches + 1; // class token + patch tokens
//    for (int i = 0; i < total_tokens * embed_dim; i++) {
//        //("%f ", final_tokens[i]);
//    }
//    //printf("\n");
//}
//
//static void pos_emb(float* input, float* output, Network pos_emb) {
//    // output_size: 한 변의 패치 수, num_patches: 전체 패치 수, total_tokens: class token + patch tokens
//    int output_size = img_size / patch_size;
//    int num_patches = output_size * output_size;
//    int total_tokens = num_patches + 1;
//    int total_elements = total_tokens * embed_dim;
//    for (int i = 0; i < total_elements; i++) {
//        output[i] = input[i] + pos_emb.data[i];
//    }
//}
//
//static void layer_norm(float* input, float* output, Network weight, Network bias) {
//    int token = ((img_size / patch_size) * (img_size / patch_size)) + 1;
//
//    for (int t = 0; t < token; t++) {
//        float sum = 0.0, sum_sq = 0.0;
//        for (int i = 0; i < embed_dim; i++) {
//            float val = input[t * embed_dim + i];
//            sum += val;
//            sum_sq += val * val;
//        }
//        float mean = sum / embed_dim;
//        float var = sum_sq / embed_dim - mean * mean;
//        float inv_std = 1.0f / sqrtf(var + eps);
//        for (int i = 0; i < embed_dim; i++) {
//            int idx = t * embed_dim + i;
//            output[idx] = (input[idx] - mean) * inv_std * weight.data[i] + bias.data[i];
//        }
//    }
//}
//
//static void multihead_attn(float* input, float* output,
//    Network in_weight, Network in_bias, Network out_weight, Network out_bias) {
//
//    int head_dim = embed_dim / num_heads, tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
//
//    /*Allocate Q, K, V : tokens * dim*/
//    int Q_dim = 0, K_dim = embed_dim, V_dim = embed_dim * 2;
//    float* Q = (float*)malloc(sizeof(float) * tokens * embed_dim);
//    float* K = (float*)malloc(sizeof(float) * tokens * embed_dim);
//    float* V = (float*)malloc(sizeof(float) * tokens * embed_dim);
//
//    /*Q, K, V 구하기*/
//    for (int t = 0; t < tokens; t++) {
//        float sum_q, sum_k, sum_v;
//        for (int i = 0; i < embed_dim; i++) {
//            sum_q = in_bias.data[Q_dim + i], sum_k = in_bias.data[K_dim + i], sum_v = in_bias.data[V_dim + i];
//            for (int j = 0; j < embed_dim; j++) {
//                sum_q += input[t * embed_dim + j] * in_weight.data[(Q_dim + i) * embed_dim + j];
//                sum_k += input[t * embed_dim + j] * in_weight.data[(K_dim + i) * embed_dim + j];
//                sum_v += input[t * embed_dim + j] * in_weight.data[(V_dim + i) * embed_dim + j];
//            }
//            Q[t * embed_dim + i] = sum_q;
//            K[t * embed_dim + i] = sum_k;
//            V[t * embed_dim + i] = sum_v;
//        }
//    }
//    int print_tokens = tokens < 5 ? tokens : 5;
//    int print_dims = embed_dim < 10 ? embed_dim : 10;
//
//    /*Attn 결과를 저장할 버퍼*/
//    float* attn_output = (float*)malloc(sizeof(float) * tokens * embed_dim);
//    for (int i = 0; i < tokens * embed_dim; i++) attn_output[i] = 0.0f;
//
//    /*head별로 attn 수행*/
//    for (int h = 0; h < num_heads; h++) {
//        int head_offset = h * head_dim;
//
//        // attn_score 저장 공간
//        float* scores = (float*)malloc(sizeof(float) * tokens * tokens);
//        float* scores_tmp = (float*)malloc(sizeof(float) * tokens * tokens);
//
//
//        // 각 head에 대해 scaled-dot attn
//        for (int i = 0; i < tokens; i++) {
//            for (int j = 0; j < tokens; j++) {
//                float score = 0.0f;
//                for (int d = 0; d < head_dim; d++) {
//                    float q = Q[i * embed_dim + head_offset + d];
//                    float k = K[j * embed_dim + head_offset + d];
//                    score += q * k;
//                }
//                scores[i * tokens + j] = score / sqrtf((float)head_dim);
//            }
//        }
//
//        // softmax 적용
//        for (int i = 0; i < tokens; i++) {
//            float max_val = scores[i * tokens];
//            for (int j = 1; j < tokens; j++) {
//                if (scores[i * tokens + j] > max_val) max_val = scores[i * tokens + j];
//            }
//            float sum_exp = 0.0f;
//            for (int j = 0; j < tokens; j++) {
//                scores[i * tokens + j] = expf(scores[i * tokens + j] - max_val);
//                sum_exp += scores[i * tokens + j];
//            }
//            for (int j = 0; j < tokens; j++) {
//                scores[i * tokens + j] /= sum_exp;
//            }
//        }
//
//        // scores와 V를 곱해 head output 계산
//        float* head_out = (float*)malloc(sizeof(float) * tokens * head_dim);
//        for (int i = 0; i < tokens; i++) {
//            for (int d = 0; d < head_dim; d++) {
//                float sum = 0.0f;
//                for (int j = 0; j < tokens; j++) {
//                    sum += scores[i * tokens + j] * V[j * embed_dim + head_offset + d];
//                }
//                head_out[i * head_dim + d] = sum;
//            }
//        }
//
//        // head_out를 attn_output의 해당 부분에 복사
//        for (int i = 0; i < tokens; i++) {
//            for (int d = 0; d < head_dim; d++) {
//                attn_output[i * embed_dim + head_offset + d] = head_out[i * head_dim + d];
//            }
//        }
//
//        free(scores);
//        free(head_out);
//    }
//
//    free(Q); free(K); free(V);
//
//    // 최종 선형 프로젝션
//    for (int t = 0; t < tokens; t++) {
//        for (int i = 0; i < embed_dim; i++) {
//            float sum = out_bias.data[i];
//            for (int j = 0; j < embed_dim; j++) {
//                sum += attn_output[t * embed_dim + j] * out_weight.data[i * embed_dim + j];
//            }
//            output[t * embed_dim + i] = sum;
//        }
//    }
//    free(attn_output);
//}
//
//static float gelu(float x) {
//    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
//}
//static void gelu_activation(float* input, float* output, int size) {
//    for (int i = 0; i < size; i++) {
//        output[i] = gelu(input[i]);
//    }
//}
//
//static void linear_layer(float* input, float* output, int tokens, int in_features, int out_features, Network weight, Network bias) {
//    for (int t = 0; t < tokens; t++) {
//        for (int o = 0; o < out_features; o++) {
//            float sum = bias.data[o];
//            for (int i = 0; i < in_features; i++) {
//                sum += input[t * in_features + i] * weight.data[o * in_features + i];
//            }
//            output[t * out_features + o] = sum;
//        }
//    }
//}
//static void mlp_block(float* input, float* output, Network fc1_weight, Network fc1_bias, Network fc2_weight, Network fc2_bias) {
//    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1; //197
//    int Embed_dim = embed_dim; //768
//    int hidden_dim = ((int)(embed_dim * mlp_ratio)); //3072
//
//
//
//    float* fc1_out = (float*)malloc(sizeof(float) * tokens * hidden_dim);
//
//    linear_layer(input, fc1_out, tokens, embed_dim, hidden_dim, fc1_weight, fc1_bias);
//    // GELU 활성화
//    for (int i = 0; i < tokens * hidden_dim; i++) {
//        fc1_out[i] = gelu(fc1_out[i]);
//    }
//    // fc2: (tokens, in_dim)
//    linear_layer(fc1_out, output, tokens, hidden_dim, embed_dim, fc2_weight, fc2_bias);
//    free(fc1_out);
//}
//
//////////////////////////////////////// Encoder Architecture //////////////////////////////////////
//static void Encoder(float* input, float* output,
//    Network ln1_w, Network ln1_b, Network attn_w, Network attn_b, Network attn_out_w, Network attn_out_b,
//    Network ln2_w, Network ln2_b, Network mlp1_w, Network mlp1_b, Network mlp2_w, Network mlp2_b) {
//    int tokens = ((img_size / patch_size) * (img_size / patch_size)) + 1;
//    float* ln1_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
//    float* attn_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
//    float* residual = (float*)malloc(sizeof(float) * tokens * embed_dim);
//    float* ln2_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
//    float* mlp_out = (float*)malloc(sizeof(float) * tokens * embed_dim);
//
//    /*LN1*/
//    layer_norm(input, ln1_out, ln1_w, ln1_b);
//
//    /*Attn*/
//    multihead_attn(ln1_out, attn_out, attn_w, attn_b, attn_out_w, attn_out_b);
//
//    /*Residual1*/
//    for (int i = 0; i < tokens * embed_dim; i++) {
//        residual[i] = input[i] + attn_out[i];
//    }
//
//    /*LN2*/
//    layer_norm(residual, ln2_out, ln2_w, ln2_b);
//
//    /*MLP*/
//    mlp_block(ln2_out, mlp_out, mlp1_w, mlp1_b, mlp2_w, mlp2_b);
//
//    /*Residual2*/
//    for (int i = 0; i < tokens * embed_dim; i++) {
//        output[i] = residual[i] + mlp_out[i];
//    }
//
//    free(ln1_out); free(attn_out); free(residual); free(ln2_out); free(mlp_out);
//}
//
//static void Softmax(float* logits, float* probabilities, int length) {
//    // 수치 안정성을 위한 최대값 계산
//    float max_val = logits[0];
//    for (int i = 1; i < length; i++) {
//        if (logits[i] > max_val) {
//            max_val = logits[i];
//        }
//    }
//
//    // 각 원소에 대해 exp(logit - max_val)을 계산하고 합산
//    float sum_exp = 0.0f;
//    for (int i = 0; i < length; i++) {
//        probabilities[i] = expf(logits[i] - max_val);
//        sum_exp += probabilities[i];
//    }
//
//    // 확률값으로 정규화
//    for (int i = 0; i < length; i++) {
//        probabilities[i] /= sum_exp;
//    }
//}
//
/////////////////////////////////////// layer별 size  //////////////////////////////////////
//static const int size[] = {
//    embed_dim * (img_size / patch_size) * (img_size / patch_size), // conv2D
//    embed_dim * (img_size / patch_size) * (img_size / patch_size), // flatten and transpose
//    embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1), // class token
//    embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1) // position embedding
//};
//
//static const int enc_size = embed_dim * ((img_size / patch_size) * (img_size / patch_size) + 1);
//
//void ViT_seq_ZZani(ImageData* image, Network* networks, float** probabilities) {
//    cl_int err;
//
//    err = clGetPlatformIDs(1, &ctx.platform, NULL);
//    CHECK_ERROR(err);
//
//    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, NULL);
//    CHECK_ERROR(err);
//
//    ctx.context = clCreateContext(NULL, 1, &ctx.device, NULL, NULL, &err);
//    CHECK_ERROR(err);
//
//    ctx.queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, 0, &err);
//    CHECK_ERROR(err);
//
//    size_t kernel_source_size;
//    char* kernel_source = get_source_code("kernel_ZZani.cl", &kernel_source_size);
//    cl_program program = clCreateProgramWithSource(ctx.context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
//    CHECK_ERROR(err);
//
//    err = clBuildProgram(program, 1, &ctx.device, "", NULL, NULL);
//    build_error(program, ctx.device, err);
//    CHECK_ERROR(err);
//
//    // below : kernel creation, buffer allocation, data transfer, kernel execution, result retrieval, cleanup //////////////
//
//    int token_size = ((img_size / patch_size) * (img_size / patch_size) + 1); // 197
//    float* layer[4];
//    float* enc_layer[12];
//    float* enc_output;
//    int hidden_dim = ((int)(embed_dim * mlp_ratio)); // 3072
//
//    // printf("%d %d = %d\n", token_size, hidden_dim, token_size * hidden_dim);
//
//    for (int i = 0; i < 4; i++) {
//        layer[i] = (float*)malloc(sizeof(float) * size[i]);
//    }
//    for (int i = 0; i < 12; i++) {
//        enc_layer[i] = (float*)malloc(sizeof(float) * enc_size);
//    }
//    enc_output = (float*)malloc(sizeof(float) * enc_size);
//
//    clock_t start, end;
//    double cpu_time_used;
//
//    for (int i = 0; i < image->n; i++) {
//        clock_t total_start = clock();
//        printf("Processing image %d/%d\n", i + 1, image->n);
//
//        /*patch embedding*/
//        start = clock();
//        Conv2d(image[i].data, layer[0], networks[1], networks[2]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Conv2d: %f seconds\n", cpu_time_used);
//
//        /*flatten and transpose*/
//        start = clock();
//        flatten_transpose(layer[0], layer[1]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - flatten_transpose: %f seconds\n", cpu_time_used);
//
//        /*prepend class token*/
//        start = clock();
//        class_token(layer[1], layer[2], networks[0]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - class_token: %f seconds\n", cpu_time_used);
//
//        /*position embedding*/
//        start = clock();
//        pos_emb(layer[2], layer[3], networks[3]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - pos_emb: %f seconds\n", cpu_time_used);
//        
//        /*Encoder - 12 Layers*/
//        start = clock();
//        Encoder(layer[3], enc_layer[0],
//            networks[4], networks[5], networks[6], networks[7],
//            networks[8], networks[9], networks[10], networks[11],
//            networks[12], networks[13], networks[14], networks[15]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Encoder 1: %f seconds\n", cpu_time_used);
//
//        start = clock();
//        Encoder(enc_layer[0], enc_layer[1],
//            networks[16], networks[17], networks[18], networks[19],
//            networks[20], networks[21], networks[22], networks[23],
//            networks[24], networks[25], networks[26], networks[27]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Encoder 2: %f seconds\n", cpu_time_used);
//
//        start = clock();
//        Encoder(enc_layer[1], enc_layer[2],
//            networks[28], networks[29], networks[30], networks[31],
//            networks[32], networks[33], networks[34], networks[35],
//            networks[36], networks[37], networks[38], networks[39]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Encoder 3: %f seconds\n", cpu_time_used);
//
//        start = clock();
//        Encoder(enc_layer[2], enc_layer[3],
//            networks[40], networks[41], networks[42], networks[43],
//            networks[44], networks[45], networks[46], networks[47],
//            networks[48], networks[49], networks[50], networks[51]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Encoder 4: %f seconds\n", cpu_time_used);
//
//        start = clock();
//        Encoder(enc_layer[3], enc_layer[4],
//            networks[52], networks[53], networks[54], networks[55],
//            networks[56], networks[57], networks[58], networks[59],
//            networks[60], networks[61], networks[62], networks[63]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Encoder 5: %f seconds\n", cpu_time_used);
//
//        start = clock();
//        Encoder(enc_layer[4], enc_layer[5],
//            networks[64], networks[65], networks[66], networks[67],
//            networks[68], networks[69], networks[70], networks[71],
//            networks[72], networks[73], networks[74], networks[75]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Encoder 6: %f seconds\n", cpu_time_used);
//
//        start = clock();
//        Encoder(enc_layer[5], enc_layer[6],
//            networks[76], networks[77], networks[78], networks[79],
//            networks[80], networks[81], networks[82], networks[83],
//            networks[84], networks[85], networks[86], networks[87]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Encoder 7: %f seconds\n", cpu_time_used);
//
//        start = clock();
//        Encoder(enc_layer[6], enc_layer[7],
//            networks[88], networks[89], networks[90], networks[91],
//            networks[92], networks[93], networks[94], networks[95],
//            networks[96], networks[97], networks[98], networks[99]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Encoder 8: %f seconds\n", cpu_time_used);
//
//        start = clock();
//        Encoder(enc_layer[7], enc_layer[8],
//            networks[100], networks[101], networks[102], networks[103],
//            networks[104], networks[105], networks[106], networks[107],
//            networks[108], networks[109], networks[110], networks[111]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Encoder 9: %f seconds\n", cpu_time_used);
//
//        start = clock();
//        Encoder(enc_layer[8], enc_layer[9],
//            networks[112], networks[113], networks[114], networks[115],
//            networks[116], networks[117], networks[118], networks[119],
//            networks[120], networks[121], networks[122], networks[123]);
//        end = clock();
//        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//        printf("  - Encoder 10: %f seconds\n", cpu_time_used);
//
//        start = clock();
//        Encoder(enc_layer[9], enc_layer[10],
//            networks[124], networks[125], networks[126], networks[127],
//            networks[128], networks[129], networks[130], networks[131],
//            networks[132], networks[133], networks[134], networks[135]);
//		end = clock();
//		cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//		printf("  - Encoder 11: %f seconds\n", cpu_time_used);
//
//		start = clock();
//        Encoder(enc_layer[10], enc_layer[11],
//            networks[136], networks[137], networks[138], networks[139],
//            networks[140], networks[141], networks[142], networks[143],
//            networks[144], networks[145], networks[146], networks[147]);
//		end = clock();
//		cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//		printf("  - Encoder 12: %f seconds\n", cpu_time_used);
//
//		start = clock();
//        layer_norm(enc_layer[11], enc_output, networks[148], networks[149]);
//		end = clock();
//		cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//		printf("  - Final Layer Norm: %f seconds\n", cpu_time_used);
//
//        /* Token */
//        float* cls_token = (float*)malloc(sizeof(float) * embed_dim);
//        float* cls_output = (float*)malloc(sizeof(float) * num_classes);
//		start = clock();
//        memcpy(cls_token, enc_output, sizeof(float) * embed_dim);
//		end = clock();
//		cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//		printf("  - Copy Class Token: %f seconds\n", cpu_time_used);
//
//		start = clock();
//        linear_layer(cls_token, cls_output, 1, embed_dim, num_classes, networks[150], networks[151]);
//		end = clock();
//		cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//		printf("  - Classification Head: %f seconds\n", cpu_time_used);
//
//		start = clock();
//        /* Softmax */
//        Softmax(cls_output, probabilities[i], num_classes);
//		end = clock();
//		cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
//		printf("  - Softmax: %f seconds\n", cpu_time_used);
//
//    }
//
//    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//    free(kernel_source);
//    clReleaseCommandQueue(ctx.queue);
//    clReleaseContext(ctx.context);
//    clReleaseProgram(program);
//}
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

    cl_mem attn_out = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_token * embed_dim, NULL, &err); CHECK_ERROR(err);
    cl_mem scores = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * total_token * tokens, NULL, &err); CHECK_ERROR(err);

    /*head���� attn ����*/
    for (int h = 0; h < num_heads; h++) {
        int head_offset = h * head_dim;

        // Attention Score ���
        err = clSetKernelArg(ctx.score_kernel, 0, sizeof(cl_mem), &QKV); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 1, sizeof(cl_mem), &scores); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.score_kernel, 2, sizeof(int), &head_offset); CHECK_ERROR(err);

        size_t global_size_score[3] = {
            (size_t)tokens,
            (size_t)tokens,
            (size_t)batch_size
        };

        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.score_kernel, 3, NULL, global_size_score, NULL, 0, NULL, NULL); CHECK_ERROR(err);

        // Softmax ���
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

static void initialize_kernel(Network* networks) {
    cl_int err;

    err = clGetPlatformIDs(1, &ctx.platform, NULL); CHECK_ERROR(err);
    err = clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, NULL); CHECK_ERROR(err);

    ctx.context = clCreateContext(NULL, 1, &ctx.device, NULL, NULL, &err); CHECK_ERROR(err);

    cl_queue_properties props[] = {
        /*CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,*/
        0
    };

    ctx.input_queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);
    ctx.compute_queue = clCreateCommandQueueWithProperties(ctx.context, ctx.device, props, &err); CHECK_ERROR(err);

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

void ViT_seq_sb(ImageData* image, Network* networks, float** probabilities) {
    cl_int err;

    initialize_kernel(networks);

    cl_mem enc_buf[12];
    for (int i = 0; i < 12; i++) {
        enc_buf[i] = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(float) * batch_size * enc_size, NULL, &err); CHECK_ERROR(err);
    }

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
        Conv2d(input, buf, networks[1], networks[2]);

        // class token                              ��
        // class token prepended + patch tokens     ��- in one kernel
        // positional encoding                      ��
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

        size_t global_size_extract_cls[2] = {
            (size_t)batch_size,
            (size_t)embed_dim
        };

        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.extract_cls_kernel, 2, NULL, global_size_extract_cls, NULL, 0, NULL, NULL); CHECK_ERROR(err);

        linear_layer(cls_tokens, cls_output, batch_size, embed_dim, num_classes, networks[150], networks[151]);

        int classes = num_classes;
        err = clSetKernelArg(ctx.softmax_kernel, 0, sizeof(cl_mem), &cls_output); CHECK_ERROR(err);
        err = clSetKernelArg(ctx.softmax_kernel, 1, sizeof(int), &classes); CHECK_ERROR(err);

        size_t global_size_softmax = current_batch_size;
        err = clEnqueueNDRangeKernel(ctx.compute_queue, ctx.softmax_kernel, 1, NULL, &global_size_softmax, NULL, 0, NULL, NULL); CHECK_ERROR(err);

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