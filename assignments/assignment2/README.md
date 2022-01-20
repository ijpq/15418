CMU 15418 Assignment 2: A Simple CUDA Renderer
=========================================================================

Please refer to the [course website](http://15418.courses.cs.cmu.edu/spring2017/article/4) for instructions.


====
1. CUDA warmup Saxpy 
  * 只测量cuda kernel的运行时间
  * 关于cuda的时间度量函数: 对于cpu上的主函数线程来说，CUDA线程是异步的。所以需要在调用完cuda kernel之后，调用cudaThreadSynchronize()函数。这个函数会在cuda kernel执行完后reutrn.
