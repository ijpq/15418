#include <cuda.h>
#include <cuda_runtime.h>

#define THREADSPERBLOCK 1024

static inline int nextPow2_2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__device__ int scan_warp(volatile int *ptr, const unsigned int idx) {
    /* return exclusive scan resutls, but set value inclusive*/
    const unsigned int lane = idx & 31;
    if (lane >= 1) ptr[idx] += ptr[idx-1];
    if (lane >= 2) ptr[idx] += ptr[idx-2];
    if (lane >= 4) ptr[idx] += ptr[idx-4];
    if (lane >= 8) ptr[idx] += ptr[idx-8];
    if (lane >= 16) ptr[idx] += ptr[idx-16];
    return (lane > 0) ? ptr[idx-1] : 0;
}

__device__ int inclusive_scan_warp(volatile int *ptr, const unsigned int idx) {
    /* return inclusive scan resutls and set value inclusive*/
    const unsigned int lane = idx & 31;
    if (lane >= 1) ptr[idx] += ptr[idx-1];
    if (lane >= 2) ptr[idx] += ptr[idx-2];
    if (lane >= 4) ptr[idx] += ptr[idx-4];
    if (lane >= 8) ptr[idx] += ptr[idx-8];
    if (lane >= 16) ptr[idx] += ptr[idx-16];
    return ptr[idx];
}

__device__ void scan_block(volatile int *ptr, int idx) {
    /* 
       given a pointer to  block address and threadidx, return the exclusive
       scan result of that index.
    */

    // ptr is different from each thread block.
    const unsigned int lane = idx & 31; // mod 32
    
    // (0~31) belongs to warp0, (32~63) belongs to warp1.
    const unsigned int warp_id = idx >> 5; 

    // get exclusive scan resutls
    int val = scan_warp(ptr, idx);
    __syncthreads();


    // set inclusive scan base to mem addr.
    if (lane == 31) {
        ptr[warp_id] = ptr[idx];
    }
    __syncthreads();

    if (warp_id == 0) {
        scan_warp(ptr, idx);
    }
    __syncthreads();

    if (warp_id > 0) {
        val = val + ptr[warp_id - 1];
    }
    __syncthreads();

    ptr[idx] = val;
    return ;
}

__device__ void inclusive_scan_block(volatile int *ptr, int idx) {
    /* 
       the only difference from scan_block is write ptr inclusive results.
    */

    // ptr is different from each thread block.
    const unsigned int lane = idx & 31; // mod 32
    
    // (0~31) belongs to warp0, (32~63) belongs to warp1.
    const unsigned int warp_id = idx >> 5; 

    // get inclusive scan resutls
    int val = inclusive_scan_warp(ptr, idx);
    __syncthreads();


    // set inclusive scan base to mem addr.
    if (lane == 31) {
        ptr[warp_id] = ptr[idx];
    }
    __syncthreads();

    // base inclusive scan
    if (warp_id == 0) {
        inclusive_scan_warp(ptr, idx);
    }
    __syncthreads();

    if (warp_id > 0) {
        val = val + ptr[warp_id - 1];
    }
    __syncthreads();

    // correct value settle within each block. but base value between blocks
    // has not been added to other block
    ptr[idx] = val;
    return ;
}


// NOTE: 把各block都写成inclusive是为了存base，但是这样的话，后面在给个元素+base的时候，这些值也都是inclusive了就不对了。
// 不能到stage3再去读取前一个位置的元素值，因为是并行的代码，如果想要去读取前一个元素的值，就会产生数据依赖，需要串行了，不科学

__global__ void calculate_block_base(int *device_data, int *base) {
    /*
       save block accumulated summation to base pointer via add exlcusive result of last element
       to last element, but device_data save exclusive scan resutls.
    */
    int *block_ptr = device_data + blockIdx.x * blockDim.x;
    // save last element value of each block
    base[blockIdx.x] = block_ptr[THREADSPERBLOCK-1];
    // get exclusive scan results
    scan_block(block_ptr, threadIdx.x);
    // add exclusive scan result of last index to the base, which is inclusive.
    if (threadIdx.x == 0) {
        base[blockIdx.x] += block_ptr[THREADSPERBLOCK-1];
    }

    return ;
}

__device__ void add_base_to_latter_blocks(int *block_ptr, int idx, int base) {
    block_ptr[idx] += base;
    return ;
}

__global__ void sync_global_base(int *base, int nr_nested_blocks) {
    /* 
       accumulated each nested block base, add last element of nested block to latter block base
        FIXME: sequential impl exactly, how to improve it?
    */
    for (int blockindex = 1; blockindex < nr_nested_blocks; blockindex++) {
        int *block_ptr = base + blockindex * blockDim.x;
        add_base_to_latter_blocks(block_ptr, threadIdx.x, base[blockindex*blockDim.x-1]);
    }
}

__global__ void calculate_nested_block_base(int *base) {
    /* 
       execute inclusive scan in each nested block within base pointer which 
       have no impact at device_data.
       */
    int *block_ptr = base + blockIdx.x * blockDim.x;
    inclusive_scan_block(block_ptr, threadIdx.x);
}

__device__ void add_base_to_values(int *ptr, int base) {
    *(ptr) += base;
}
__global__ void sync_global_values(int *device_data, int *base) {
    if (blockIdx.x > 0) {
        add_base_to_values(device_data+blockIdx.x*blockDim.x+threadIdx.x, base[blockIdx.x-1]);
    }
}

void exclusive_scan1(int *device_data, int length) {
    /*
       this functions is a implement style introduced by course video.
       caused NlogN work and logN span.
   */

    int valid_size = nextPow2_2(length);
    int num_blocks = (valid_size + THREADSPERBLOCK -1) / THREADSPERBLOCK;
    
    int *base;
    cudaMalloc((void **)&base, sizeof(int) * num_blocks);
    cudaMemset(base, 0, sizeof(int) * num_blocks);
    calculate_block_base<<<num_blocks, THREADSPERBLOCK>>>(device_data, base);
    cudaDeviceSynchronize();

    int nr_nested_blocks = (num_blocks + THREADSPERBLOCK -1) / THREADSPERBLOCK;
    calculate_nested_block_base<<<nr_nested_blocks, THREADSPERBLOCK>>>(base);
    cudaDeviceSynchronize();
    
    sync_global_base<<<1, THREADSPERBLOCK>>>(base, nr_nested_blocks);
    cudaDeviceSynchronize();

    sync_global_values<<<num_blocks, THREADSPERBLOCK>>>(device_data, base);
    
    cudaFree(base);

    return ;
}
