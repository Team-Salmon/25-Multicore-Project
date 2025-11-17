#include "Network.h"

#ifndef _ViT_seq_H
#define _ViT_seq_H

#include <CL/cl.h>

char* get_source_code(const char* file_name, size_t* len);
void build_error(cl_program program, cl_device_id device, cl_int err);

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

void ViT_seq(ImageData* image, Network* networks, float** prb);

void ViT_seq_sb(ImageData* image, Network* networks, float** prb);
void ViT_seq_seoh(ImageData* image, Network* networks, float** prb);
void ViT_seq_ZZani(ImageData* image, Network* networks, float** prb);

#endif#pragma once
