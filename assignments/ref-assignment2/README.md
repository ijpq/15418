CMU 15418 Assignment 2: A Simple CUDA Renderer
=========================================================================

Please refer to the [course website](http://15418.courses.cs.cmu.edu/spring2017/article/4) for instructions.


====
1. CUDA warmup Saxpy 
  * 只测量cuda kernel的运行时间
  * 关于cuda的时间度量函数: 对于cpu上的主函数线程来说，CUDA线程是异步的。所以需要在调用完cuda kernel之后，调用cudaThreadSynchronize()函数。这个函数会在cuda kernel执行完后reutrn.

2. CUDA warmup2 parallel prefix-sum
  * 总体目标：实现查找重复数的函数find_repeats(). array[] = {1,2,2,1,1,1,3,5,3,3}; output = {1,3,4,8}.
  * 第一步：实现并行版本的exclusive prefix-sum. array[] = {1,4,6,8,2}; output = {0,1,5,11,19}
  * 递归版本的exclusive prefix-sum如下,但栈消耗太大了。
  * ![image](https://user-images.githubusercontent.com/44460962/151551623-b010c30d-ae73-4470-91c6-a6796833e030.png)
  * 迭代版本的exclusive prefix-sum如下,同时可以借助示例图理解。
  * ![image](https://user-images.githubusercontent.com/44460962/151551790-f07ff2b2-8858-4815-b86e-00437794a661.png)
  * ![image](https://user-images.githubusercontent.com/44460962/151551805-2473ed07-c289-4677-9187-399add47f8c7.png)
  * 完全可以模仿这个迭代版本的算法在scan.cu中实现第一步的目标,即exclusive_scan()函数.
  * NOTE: 实例图中假设数组长度是 $2^N$，在CUDA的kernel func实现时，alloc要对数组长度进行向上(下一个 $2^N$)取整，最终再copy N长内存回host.
  * iterative_exclusive_scan impl.
```cpp
//
// Created by ke tang on 2021/11/20.
//

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cwchar>
#define N 17
#define CONSTANT 1
using namespace std;

void iterative_scan_stage1(int *arr, int size) {
    for (int step =1; step < N/2; step*=2) {
        int iter = 2*step-1;
        for (; iter < N; iter+=step*2 ) {
            arr[iter] += arr[iter-step];
        }
    }
    return ;
}

void iterative_scan_stage2(int *arr, int size) {
    int valid_size = size;
    if (size & 1) {
        valid_size = size - 1;
    }
    arr[valid_size-1] = 0;
    for (int step=N/2; step > 0; step /=2) {
        int iter = valid_size -1;

        // sum
        for (; iter > 0; iter -= 2*step) {
            // save
            int temp = arr[iter];

            arr[iter] += arr[iter-step];

            // write to
            arr[iter-step] = temp;
        }

    }
    return ;
}

void print_out(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d,", arr[i]);
    }
    printf("\n");
    return ;
}

void fill_out(int *arr, int size) {
    for (int i =0; i < size; i ++) {
        arr[i] = CONSTANT;
    }
    return ;
}
int main() {
    int *arr = (int *)malloc(sizeof(int) * N);
    fill_out(arr, N);
    print_out(arr, N);
    iterative_scan_stage1(arr, N);
    print_out(arr, N);
    iterative_scan_stage2(arr, N);
    if (N & 1) {
        arr[N-1] += arr[N-2];
    }
    print_out(arr, N);

    return 0;
}
```

