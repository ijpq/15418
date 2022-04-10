//---------- a.cu ----------
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "b.h"
#include "a.h"


__global__ void foo (void) {
  printf("calling from kernel foo: %d\n", threadIdx.x);
  bar();
}

void warperFoo() {
    printf("calling from warperFoo\n");
    dim3 gdim(1);
    dim3 bdim(4);
    foo<<<gdim, bdim>>>();
    cudaDeviceSynchronize();
}
