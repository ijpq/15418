# prob1

1.使用双核并行计算mandelbrot，一个core计算图的上半部分，另一个core计算下半部分。这种解决方案称为spatial decomposition

2.进一步扩展为使用2,4,8,16线程来分割图像计算。需要修改workerThreadStart。
处理器有8个core，每个core支持2个超线程。每张图有900行，所以对于16线程，横向分割不能整除。画一个图来对比线程并行和非并行的加速比，纵坐标是加速比，横坐标是使用的core数。

**回答**：是否这个加速比是线性的？为什么？

3.为了验证是否线性的问题，测一下所有线程cost的时间，通过在workerThreadStart的前后加入时间测量函数。

4.修改mapping的代码，将加速比提高到8倍。

# prob2

prob2需要类模板、函数模板、模板继承的知识，可以温习一下

---

function.cpp中的clampedExpSerial实现了非并行的快速幂算法，不了解快速幂的话可以看一下，因为最终要实现一个向量化的快速幂。prob2的任务是实现一些基础的向量操作，可以在SIMD上操作，需要使用CMU418intrin.h中的一些函数。
function.cpp中有一个向量版本的abs操作，演示了一些关于向量化的操作，可以从中了解一下如何模拟向量化，（以后使用ispc库的时候，就知道底层大概怎么实现的了）但存在一些问题，不能应对所有的输入。

```cpp
void absVector(float* values, float* output, int N) {
    /*
    template <typename T>
    struct __cmu418_vec {
      T value[VECTOR_WIDTH];
    };
    */
    __cmu418_vec_float x; 
    __cmu418_vec_float result;
  
  	/*
  	void _cmu418_vset_float(__cmu418_vec_float &vecResult, float value, __cmu418_mask &mask);
  	__cmu418_vec_float _cmu418_vset_float(float value);
  	*/
    __cmu418_vec_float zero = _cmu418_vset_float(0.f);
  
  	//struct __cmu418_mask : __cmu418_vec<bool> {};
    __cmu418_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i=0; i<N; i+=VECTOR_WIDTH) {

	// All ones
	maskAll = _cmu418_init_ones();

	// All zeros
	maskIsNegative = _cmu418_init_ones(0);

	// Load vector of values from contiguous memory addresses
	_cmu418_vload_float(x, values+i, maskAll);               // x = values[i];

	// Set mask according to predicate
	_cmu418_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

	// Execute instruction using mask ("if" clause)
	_cmu418_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

	// Inverse maskIsNegative to generate "else" mask
	maskIsNotNegative = _cmu418_mask_not(maskIsNegative);     // } else {

	// Execute instruction ("else" clause)
	_cmu418_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

	// Write results back to memory
	_cmu418_vstore_float(output+i, result, maskAll);
    }
}
```

提示

1. 每一个向量操作都需要有一个mask，mask中的0（false）表示操作不会被写入，反之亦然。
2. _cmu418_cntbits函数很有用
3. 需要考虑循环长度不能整除向量宽度的情况，测试一下./vrun -s 3, 可以看一下_cmu418_init_ones函数，有用
4. ./vrun -l 是打印向量操作的log

**需要做的事**:

1. 实现向量化版本的clampedExpSerial,就是function.cpp中的clampedExpVector().
2. 运行./vrun -s 10000 在向量宽度分别为[2,4,8,16,32]的配置下，记录利用率。
   * 随着向量宽度变化，利用率的变化如何？
   * 解释向量宽度和利用率之间的关系
   * 解释向量指令事如何随着向量宽度变化的

3. 加分作业 TODO...

# prob3
在prob1中，每一个线程调用一个core。在prob3中，使用ISPC来实现。在mandelbrot图计算这个例子中，每个像素点的计算都是独立的过程，基于这些前提条件，ISPC编译器负责构建程序，来尽可能高效地调用CPU的并行资源。

## part1
ispc看起来像cpp,但实际有区别。不同于C的是，ISPC程序的多个进行并行的运行CPU中的SIMD指令，并行的进程数取决于编译器。因此，从C的代码中调用ISPC
的API就像产生了一组并行的ispc进程。这些并发的进程都运行完以后，再返回到C的代码中。

下面这个例子：

![image](https://user-images.githubusercontent.com/44460962/143202634-22c86dea-9131-458d-a9ca-03ae430adfc7.png)

ispc提供的功能可以减少对于如何split任务的思考
比如ispc提供的foreach，其中所有的迭代都是独立的，上一张图描述描述了如何将工作split给不同的worker，下面这张图这种写法只是表述执行啥

![image](https://user-images.githubusercontent.com/44460962/143202978-59dfdfb7-fc19-4534-ae45-1ae7a7204100.png)

可以看看ispc的文档，进一步了解使用方法。

**需要做的事**

编译运行mandelbrot.ispc，ispc编译器配置为使用8bit宽的向量指令。你期望你的cpu能达到的最大加速比是多少？为什么实际效果比这个预估的加速比低？

**提醒**

ispc编译器在一个core上使用simd指令，这种并行方式与prog1是不同的，prog1是并发了多个线程，每个线程使用一个core。
