# Basic Concept

* processor: contains multiple cores. 现在一般pc都是单processor

* core:

  ![image-20211125105332551](https://tva1.sinaimg.cn/large/008i3skNly1gwr7ikf5r2j30gj0cg0td.jpg)

  比如在risc-V指令集中，就是一个取指令，一个alu,一个register file,一个data mem，如下

  ![image-20211125105520709](https://tva1.sinaimg.cn/large/008i3skNly1gwr7kf2yy7j30kn0cqwg4.jpg)

  <center>图2</center>

* super scalar。

  就是流水线技术，比如这个汇编代码中，计算xi的立方，所以`mul r1, r0, r0`这条指令取完指令进入寄存器访问阶段的时候，`mul r1, r1, r0`这条指令就可以开始decode了。不过在这个具体的例子中，r1寄存器之间存在依赖关系，所以产生了数据冒险，可能通过forwarding解决

![image-20211125105629703](https://tva1.sinaimg.cn/large/008i3skNly1gwr7lljb8tj30gx0ch0tr.jpg)

* multi core

  多核设计中，我们计算xi的立方用一个core，计算xj的立方用另一个core，每个core都可以使用流水线技术来指令级并行元素的计算。这种设计的好处是我们不仅可以在计算具体每一个元素时使用指令级并行加速，而且整个数组元素之间也可以并行。

​	![image-20211125105811710](https://tva1.sinaimg.cn/large/008i3skNly1gwr7ndb5rij30gi09kdgf.jpg)



* SIMD

![image-20211125110325938](https://tva1.sinaimg.cn/large/008i3skNly1gwr7stetpjj30gi0cgwfi.jpg)

我们的图2，增加alu的数量，增加reg的宽度，效果如下

![image-20211125110550718](https://tva1.sinaimg.cn/large/008i3skNly1gwr7vbs52cj30fw0cxwgb.jpg)

多线程和超线程：

![image-20211125113919956](https://tva1.sinaimg.cn/large/008i3skNgy1gwr8uajzilj30gx0c7dhf.jpg)

> **Temporal multithreading** is one of the two main forms of [multithreading](https://en.wikipedia.org/wiki/Multithreading_(computer_architecture)) that can be implemented on computer processor hardware, the other being [simultaneous multithreading](https://en.wikipedia.org/wiki/Simultaneous_multithreading). The distinguishing difference between the two forms is the maximum number of concurrent [threads](https://en.wikipedia.org/wiki/Thread_(computer_science)) that can execute in any given [pipeline](https://en.wikipedia.org/wiki/Pipeline_(computing)) stage in a given [cycle](https://en.wikipedia.org/wiki/Instruction_cycle). In temporal multithreading the number is one, while in simultaneous multithreading the number is greater than one. Some authors use the term **super-threading** synonymously

temporal multithreading和hyperthreading(simultaneous)在指令流水线上的区别只是，进来的之间属于不同的线程。而为了避免冒险的forwarding技术是不需要区分线程的?

![image-20211125115529066](https://tva1.sinaimg.cn/large/008i3skNgy1gwr9azaw0jj30ev05a3yt.jpg)

所以这个图说的是，对于流水线技术仍不能提高的那部分latency（比如某些指令存在冒险，无法通过forwarding等解决了），需要切换线程，来进一步减少latency

基础概念总结：

![image-20211125111204075](https://tva1.sinaimg.cn/large/008i3skNly1gwr81t1kauj30hh0d3wgd.jpg)

# mem

* stalls: 访问data mem太慢，alu停止

![image-20211125112155217](https://tva1.sinaimg.cn/large/008i3skNly1gwr8c1z1wmj30h80bvq3w.jpg)

角度1: cache,内存的数据已经放到cache里面，不会去访问mem，所以减少了alu stall的latency

![image-20211125112345825](https://tva1.sinaimg.cn/large/008i3skNly1gwr8dzadjdj30h80c7dgl.jpg)

角度2: prefetch，cpu提前把后面需要访问内存的数据提前load到cache里面去，等执行到后面的时候，数据已经在cache里面了。

![image-20211125112449519](https://tva1.sinaimg.cn/large/008i3skNly1gwr8f2oahtj30gy0c8t9v.jpg)

角度3:多线程

![image-20211125112620787](https://tva1.sinaimg.cn/large/008i3skNly1gwr8gnoioij30hn0d6dhc.jpg)

# gpu结构

![image-20211125115822968](https://tva1.sinaimg.cn/large/008i3skNly1gwr9dznopvj30gz0c0myh.jpg)

对比AVX(256)/SSE系列(128)，AVXfloat32的vectorwidth是8,GTX480是32.

![image-20211125135626095](https://tva1.sinaimg.cn/large/008i3skNly1gwrcsu5k1bj30y20ijgpg.jpg)

# CUDA

* starter code

  ```cpp
  // Kernel definition
  __global__ void VecAdd(float* A, float* B, float* C)
  {
      int i = threadIdx.x;
      C[i] = A[i] + B[i];
  }
  
  int main()
  {
      ...
      // Kernel invocation with N threads
      VecAdd<<<2, N>>>(A, B, C);
      ...
  }
  ```

  执行N次，每次有一个单独的线程执行

* thread Hierarchy

  1. thread, block, grid: thread 和block都是3维索引。这样设计的原因是我们处理的数据一般是N维的，所以这样方便直接使用SIMD的思路去配合数据解决问题。
  2. kernel的一个block中threads上限是1k,这是因为一个block的线程都分配到一个core上，mem资源有限，所以不建议超过1k

  3. 线程通信(__syncthreads() etc)
  4. example1 vecAdd

  ```cpp
  // Kernel definition
  __global__ void MatAdd(float A[N][N], float B[N][N],
  float C[N][N])
  {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      if (i < N && j < N)
          C[i][j] = A[i][j] + B[i][j];
  }
  
  int main()
  {
      ...
      // Kernel invocation
      dim3 threadsPerBlock(16, 16);
      dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
      MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
      ...
  } 
  ```

  * id到数据索引的映射

* mem hierarchy

  ![image-20211125162543193](https://tva1.sinaimg.cn/large/008i3skNly1gwrh45qpn2j30cl0egabr.jpg)

  ![image-20211126140731351](https://tva1.sinaimg.cn/large/008i3skNgy1gwsiqpwgtcj30dg0ecmy8.jpg)

  * global mem:类似DRAM
  * texture mem:
  * constant mem:只读的，所有核函数可见，warp中所有线程访问相同地址
  * local mem:过大的结构体，从寄存器溢出
  * shared mem:相当于host的cache
  * reg:局部变量

