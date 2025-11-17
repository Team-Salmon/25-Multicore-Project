#include "Network.h"


#ifndef _ViT_seq_H
#define _ViT_seq_H

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

void ViT_seq(ImageData* image, Network* networks, float** prb);

void ViT_seq_opencl(ImageData * image, Network * networks, float** prb);

#endif#pragma once
