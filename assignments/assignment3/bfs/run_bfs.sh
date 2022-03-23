#!/bin/bash

#PBS -l select=1:ncpus=272 -lplace=excl
export OMP_NUM_THREADS=68;
export KMP_AFFINITY=granularity=fine,compact,1,0;

$BFS_GRADER_PATH /export/shared/cmu-15418/asst3_graphs/
