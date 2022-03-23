# # Get a list of all currently submitted jobs and delete from queue
# PREV_JOBS=$(qstat -u `whoami` | grep iam-pbs | tail -n +2 | cut -d ' ' -f 1)
# qdel $PREV_JOBS

qsub -v PR_PATH=`readlink -f ./pr_dist`,GRAPH_TYPE=$1 -q cmu-15418 /export/shared/cmu-15418/pr_job.sh 
