make 
qsub -v BFS_GRADER_PATH=`readlink -f ./bfs_grader` -q cmu-15418 ./run_bfs.sh
