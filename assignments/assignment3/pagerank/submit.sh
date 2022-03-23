make 
qsub -v PR_GRADER_PATH=`readlink -f ./pr_grader` -q cmu-15418 ./run_pr.sh
