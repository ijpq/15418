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

在prob1中，每一个线程调用一个core。在prob3中，使用ISPC来实现。在mandelbrot图计算这个例子中，每个像素点的计算都是独立的过程，基于这些前提条件，ISPC编译器负责构建程序，来尽可能高效地调用CPU的并行资源。**problem3中程序的正确性可以保证，但效率上存在问题，你需要做的是fix这个问题，以获得比串行计算高20%的加速比**

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

编译运行mandelbrot.ispc，ispc编译器配置为使用8bit宽的向量指令。你期望你的cpu能达到的最大加速比是多少？为什么实际效果比这个预估的加速比低？（考虑一下你正在进行的计算的特点；图片的哪一部分对于SIMD指令是最难的；对比一下不同的图view1~6，可能会验证你的假设）

**提醒**

ispc编译器map的一个gang的实例是在一个core上使用simd指令，这种并行方式与prog1是不同的，prog1是并发了多个线程，每个线程使用一个core。

## part2

foreach利用的是simd，而ispc提供的launch利用的是多核，具体通过称为task的东西，类似线程。每一个task定义了一种计算，这部分计算由一个gang通过SIMD的方式来完成。正如函数mandelbrot_ispc_task()，每一个task计算全图的一部分。与foreach相似的是，task也是可以按任意顺序执行的。

所以launch和foreach都非固定顺序执行的

**需要做的事**

* 运行mandelbrot_ispc并设定不同的tasks，对于图1来说，不同的tasks数能获得什么加速比？与foreach这种不适用tasks的实现方法，现在的实现方法的加速比更高了还是如何？
* 最简单的提高加速比的方式是增加tasks数量，但是如何确定应该创建多少个tasks？对于你的任务来说，多少的tasks是最好的？只是修改代码中的mandelbrot_ispc_withtasks()函数，你应该能获得20-22倍的加速比(注意处理图片的高不能被整除的情况)
* extra credict

## 其他问题

为什么需要launch和foreach两种机制？为什么不能通过foreach直接实现在多核上实现SIMD?

A: 需要看lectures



# prob4

prob4实现的是开方运算，计算两千万(2e7)个0~3之间的数字的平方根。对于一个数字s,使用迭代牛顿法去计算$1/x^2=s$,求得$x \approx \sqrt{1 / s}$,再给x乘上s得到s的平方根。下图展示了0~3之间数字收敛到精确解所需要的迭代次数。

![image-20211212172435417](https://tva1.sinaimg.cn/large/008i3skNly1gxb6coq6mvj30ic0a1aab.jpg)

**需要做的事**

* build run sqrt。对比ispc不用task和用了task的加速比之间的差异。由SIMD带来的加速比是多少？由多核带来的加速比是多少？
* 更改data.cpp中的initGood()，用来生成数据以获得相对更高的加速比。描述：为什么这样的数据（在带task和不带task时）会获得相比串行实现最大的加速比？使用`--data g`来测试不同输入数据的加速比结果。做了这种更改后，是否能提高SIMD的加速效果？是否能提高多核的加速效果？解释一下为什么？
* 更改data.cpp中的initBad()，相比不带task的SIMD实现，将会生成数据导致非常低的加速比。解释一下为什么相比串行实现，这种输入数据会导致SIMD（带task和不带task）获得相对低的加速比？使用`--data b`来进行测试。这种更改的输入数据能否提高多核的加速比？为什么？

**注意**：当运行good输入数据版本时，多核执行的收益是什么？你可能会发现这个加速比非常高，**这是intel 超线程技术的效果**

## writeup

实现时，badinit的理解是增加了串行的迭代次数，goodinit的减少了串行的迭代次数。

拟牛顿法求解有两种形式，从代数上是等价的，但数值上可能存在误差，分别如下：

### method1



![image-20211213163243208](https://tva1.sinaimg.cn/large/008i3skNly1gxcah0rlnrj30ge0ba3zh.jpg)

统计迭代次数结果如下：

![iter_counts](https://tva1.sinaimg.cn/large/008i3skNly1gxftxvrx1wj30rs0go0tx.jpg)

### method2

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gxgo3g02zwj30xo0nago1.jpg" alt="image-20211217112620978" style="zoom:50%;" />

统计迭代次数结果如下：

![image-20211216210406197](https://tva1.sinaimg.cn/large/008i3skNly1gxfz6aidxbj31c80u0q68.jpg)

---

### 在method1的设定下:

bad init选择0.001f, good init选择1.001f;

bad init导致的串行计算时间增加不明显,只比随机初始化增加了3.5%.



![image-20211216180940468](https://tva1.sinaimg.cn/large/008i3skNly1gxfu4sdmiyj30fk07mjs9.jpg)

如果我们把bad init降低为0.00001f，则会出现求解错误，这应该是计算误差导致的

![image-20211216181423025](https://tva1.sinaimg.cn/large/008i3skNly1gxfu9oy9z2j30af08ljsk.jpg)

修改计算方法为只使用一个fabs，仍然出现计算错误

![image-20211219191920589](https://tva1.sinaimg.cn/large/008i3skNly1gxjd08qcg3j30ge09x0tg.jpg)

**估计的原因是，两种迭代实现方法存在细微的计算误差。使用method1的方法时，不是因为abs导致的误差**



### 在method2的设定下:

badinit 3.0f -0.00001f; goodinit 1.0f+0.00001f;

可以看到，badinit时串行所需要的iter增加了很多。但是通过ispc，可以获得更好的加速效果。goodinit时，所需要的迭代数本来就少，通过ispc获得的加速比就不显著了。

![image-20211216210905832](https://tva1.sinaimg.cn/large/008i3skNly1gxfzbhf12oj315q0ewdim.jpg)

