## assignment1
Pthreads and ISPC implement data-parallel in CPU.

## assignment2
saxpy cuda impl

exclusive scan cuda impl

cuda renderer 

## assignment3
// TODO


## ENV

### CPU
Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz

```cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l``` = 1 主板CPU个数

```cat /proc/cpuinfo| grep "cpu cores"| uniq``` = 6 也是processor数




CPU Specifications from intel.com

根据intel信息，每个core不支持超线程，每个core只能一个线程
Total Cores

6

Total Threads

6

### GPU
RTX2070Ti