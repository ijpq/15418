#define SRC_SIZE 1000
#define DST_SIZE (SRC_SIZE - 2)
#include <iostream>
#include <time.h>

using namespace std;
__global__ void conv(float *src, float *dst, int dst_N) {

    // mapping threads to dst data
    int thd = threadIdx.x + blockDim.x * blockIdx.x;
    if (thd >= dst_N) return ; 
    dst[thd] = (src[thd] + src[thd+1] + src[thd+2]) / 3.f;
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
    output_dst(dst, DST_SIZE);

    float *d_src = nullptr;
    float *d_dst = nullptr;
    cudaMalloc(&d_src, src_alloc);
    cudaMalloc(&d_dst, dst_alloc);
    cudaMemcpy(d_src, src, src_alloc, cudaMemcpyHostToDevice);

    memset(dst, 0, dst_alloc);
    int num_threads = 256;
    int num_blocks = (DST_SIZE + num_threads -1) / num_threads; 
    dim3 blocks(num_blocks);
    dim3 threads(num_threads);
    start = clock();
    conv<<<blocks, threads>>>(d_src, d_dst, DST_SIZE);
    cudaDeviceSynchronize();
    cout << ((double)(clock()-start)) / CLOCKS_PER_SEC << endl;
    cudaMemcpy(dst, d_dst, dst_alloc, cudaMemcpyDeviceToHost);
    cout << "from gpu" << endl;
    output_dst(dst, DST_SIZE);
    
    return 0;
}
