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

