#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include "renderAlongWithPixel.h"

__global__ void doit() {

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    printf("hello from %d\n", index);
    return ;
}
