# 15418

# ENV

型号：Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz

物理CPU个数：```cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l``` = 1 主板CPU个数

每个物理CPU的core数：```cat /proc/cpuinfo| grep "cpu cores"| uniq``` = 6 也是processor数

根据intel信息，每个core不支持超线程，每个core只能一个线程

CPU Specifications

Total Cores

6

Total Threads

6
