# ispc相关概念

## gang

是一定数量的程序实例，类似CUDA的warp概念。

![image-20211219211602060](https://tva1.sinaimg.cn/large/008i3skNly1gxjgdmlvlhj30ca03ndfs.jpg)

例如上面这个程序，它描述了一个gang中每个程序实例执行的计算方法。一个gang中的程序实例在相同的硬件线程及context中执行，ispc不创建线程，不执行context切换。实际上，ispc是将这些程序实例map到cpu的SIMD上去并行执行。也就是说，一条执行处理多个数据，每个数据对应一个程序实例。

gang中的程序实例数量很少，一般是普通SIMD宽度的2到4倍（因为一组被调度到SIMD上的实例执行完，需要切换到另一组，或者没执行完就切换了，为了隐藏latency等等原因）

## uniform和varing

声明为uniform的变量表示在gang中所有的程序实例都共享这个变量。而声明为varing的变量，对于gang中的每一个程序实例都是存储在不同位置的。

将变量声明为uniform的一个优势是，减少内存占用需求。更重要的收益是，这使得编译器产生更好的flow control(比如SIMD遇到if-else时)。当uniform变量遇到判断条件时，编译器能够意识到所有的程序实例都将执行相同的判断分支，减少分支分化带来的开销。

## tasking

ispc提供了异步函数机制，通过launch。使用launch调用一个ispc函数时，执行过程时异步的，这个函数可能被立即执行或被调度到另一个core上执行。

如果一个函数launch了多个task，不能确保task之间的执行顺序。

## sync

通过sync调用的task，需要同步执行。sync要求一个task等待其他task都执行完成后，才能结束。
