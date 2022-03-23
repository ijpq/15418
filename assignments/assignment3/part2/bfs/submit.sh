# # Get a list of all currently submitted jobs and delete from queue
# PREV_JOBS=$(qstat -u `whoami` | grep iam-pbs | tail -n +2 | cut -d ' ' -f 1)
# qdel $PREV_JOBS

qsub -v BFS_PATH=`readlink -f ./bfs_dist`,GRAPH_TYPE=$1 -q cmu-15418 /export/shared/cmu-15418/bfs_job.sh 
