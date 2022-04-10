#include "b.h"
#include <stdio.h>

__device__ int g[N];

__device__ void bar (void)
{
    printf("calling from kernel bar: %d \n", threadIdx.x);   
    g[threadIdx.x]++;
      
}
