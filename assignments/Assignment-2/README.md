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
  * 递归版本的exclusive prefix-sum在15-210讲过,但栈消耗太大了。
  * 迭代版本的exclusive prefix-sum,Work是O(N), Span是O(logN)，同时可以借助示例图理解。
  * 完全可以模仿这个迭代版本的算法在scan.cu中实现第一步的目标,即exclusive_scan()函数.
  * 课程中还给出了另一种实现方式，先计算每一个以warp为单位的局部scan,再将base传给非第一个warp。block之间同理。
  * NOTE: 实例图中假设数组长度是 $2^N$，在CUDA的kernel func实现时，alloc要对数组长度进行向上(下一个 $2^N$)取整，最终再copy N长内存回host.
  * iterative_exclusive_scan impl.


