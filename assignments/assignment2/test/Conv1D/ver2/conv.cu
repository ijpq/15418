#define SRC_SIZE 1000
#define THREADS_PER_BLOCK 256
#define DST_SIZE (SRC_SIZE - 2)
#include <iostream>
#include <time.h>
#include <cstring>

using namespace std;
__global__ void conv(float *src, float *dst, int dst_N) {

    // mapping threads to dst data
    int thd = threadIdx.x + blockDim.x * blockIdx.x;
    if (thd >= dst_N) return;

    // whether it is final remainder block
    int bx = THREADS_PER_BLOCK;
    if (thd + THREADS_PER_BLOCK > dst_N) {
        bx = dst_N % bx;
    }
    __shared__ float shared_mem_per_block[THREADS_PER_BLOCK+2];
    if (threadIdx.x < 2) {
        shared_mem_per_block[threadIdx.x + bx] = src[thd + bx];
    }
    shared_mem_per_block[threadIdx.x] = src[thd];

    __syncthreads();

    dst[thd] = (shared_mem_per_block[threadIdx.x] + \
            shared_mem_per_block[threadIdx.x+1] + \
            shared_mem_per_block[threadIdx.x+2]) / 3.f;
    return ;
}

void serial(float *src, float *dst, const int &dst_N) {
    for (int i = 0; i < dst_N; i++) {
        dst[i] = (src[i] + src[i+1] + src[i+2]) / 3.f;
    }
    return ;
}

void output_dst(float *dst, size_t size) {
    for (auto i =0; i < size; i++) {
        cout << dst[i] << ",";
    }
    cout << endl;
    return ;
}

int checkResult(float *cpuResult, float *gpuResult, size_t size) {
    for (int i =0 ; i < size; i++) {
        if (cpuResult[i] != gpuResult[i]) {
            cout << cpuResult[i] << "!=" << gpuResult[i] << endl;
            return 0;
        }
    }
    return 1;
}

int main(void) {
    
    size_t src_alloc = sizeof(float) * SRC_SIZE;
    size_t dst_alloc = sizeof(float) * DST_SIZE;
    float *src = (float *)malloc(src_alloc);
    float *dst = (float *)malloc(dst_alloc);
    memset(src, 0, src_alloc);
    
    for (int i =0; i < SRC_SIZE; i++) {
        src[i] = i+0.1f;
    }
    clock_t start = clock();
    serial(src, dst, DST_SIZE);
    cout << ((double)(clock()-start)) / CLOCKS_PER_SEC << endl;
    cout << "from cpu" << endl;
    // output_dst(dst, DST_SIZE);
    float *cpures = (float *)malloc(dst_alloc);
    memcpy(cpures, dst, dst_alloc);

    float *d_src = nullptr;
    float *d_dst = nullptr;
    cudaMalloc(&d_src, src_alloc);
    cudaMalloc(&d_dst, dst_alloc);
    cudaMemcpy(d_src, src, src_alloc, cudaMemcpyHostToDevice);

    memset(dst, 0, dst_alloc);
    int num_threads = THREADS_PER_BLOCK;
    int num_blocks = (DST_SIZE + num_threads -1) / num_threads; 
    dim3 blocks(num_blocks);
    dim3 threads(num_threads);
    start = clock();
    conv<<<blocks, threads>>>(d_src, d_dst, DST_SIZE);
    cudaDeviceSynchronize();
    cout << ((double)(clock()-start)) / CLOCKS_PER_SEC << endl;

    float *gpures = (float *)malloc(dst_alloc);
    cudaMemcpy(gpures, d_dst, dst_alloc, cudaMemcpyDeviceToHost);
    cout << "from gpu" << endl;
    // output_dst(gpures, DST_SIZE);
    if (checkResult(cpures, gpures, DST_SIZE)) {
        cout << "passed" << endl;
    }
    else {
        cout << "failed" << endl;
    }
    
    return 0;
}
