#include <stdio.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess) {\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s\n",error, cudaGetErrorString(error));\
        exit(-1);\
    }\
}

extern float toBW(int bytes, float sec);

__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

void
saxpyCuda(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    int totalBytes = sizeof(float) * 3 * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* device_x;
    float* device_y;
    float* device_result;

    //
    // TODO: allocate device memory buffers on the GPU using
    // cudaMalloc.  The started code issues warnings on build because
    // these buffers are used in the call to saxpy_kernel below
    // without being initialized.
    //

    //float *dev_ptr;
    //cudaError_t malloc_error = cudaMalloc(&dev_ptr, totalBytes);
    //CHECK(malloc_error);
    CHECK(cudaMalloc(&device_x, N*sizeof(float)));
    CHECK(cudaMalloc(&device_y, N*sizeof(float)));
    CHECK(cudaMalloc(&device_result, N*sizeof(float)));

    //device_x = dev_ptr;
    //device_y = dev_ptr + N;
    //device_result = device_y + N;

    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();
    clock_t entire_timer = clock();
    //
    // TODO: copy input arrays to the GPU using cudaMemcpy
    //
    cudaError_t memcpy_state = cudaMemcpy(device_x, xarray, N*sizeof(float), cudaMemcpyHostToDevice);
    CHECK(memcpy_state);
    memcpy_state = cudaMemcpy(device_y, yarray, N*sizeof(float), cudaMemcpyHostToDevice);
    CHECK(memcpy_state);
    memcpy_state = cudaMemcpy(device_result, resultarray, N*sizeof(float), cudaMemcpyHostToDevice);
    CHECK(memcpy_state);

    //
    // TODO: insert time here to begin timing only the kernel
    //
    clock_t kernel_timer = clock();

    // run saxpy_kernel on the GPU
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);

    //
    // TODO: insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaThreadSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    CHECK(cudaDeviceSynchronize());
    printf("kernel time takes %lf ms\n", 1000*(double)(clock()-kernel_timer)/CLOCKS_PER_SEC);


    //
    // TODO: copy result from GPU using cudaMemcpy
    //
    memcpy_state = cudaMemcpy(resultarray, device_result, N*sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(memcpy_state);

    // end timing after result has been copied back into host memory.
    // The time elapsed between startTime and endTime is the total
    // time to copy data to the GPU, run the kernel, and copy the
    // result back to the CPU
    double endTime = CycleTimer::currentSeconds();
    printf("entire process time takes %lf ms\n", 1000*(double)(clock()-entire_timer)/CLOCKS_PER_SEC);

    CHECK(cudaPeekAtLastError());
    //cudaError_t errCode = cudaPeekAtLastError();
    //if (errCode != cudaSuccess) {
    //    fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    //}

    double overallDuration = endTime - startTime;
    printf("Overall time: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

    //
    // TODO free memory buffers on the GPU
    //
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
