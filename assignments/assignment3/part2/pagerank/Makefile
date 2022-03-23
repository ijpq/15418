main: main.cpp page_rank_ref.o ../include/graph_dist.h
	mpic++ -O3 -std=c++11 -fopenmp -Wall -g -I../include/ -o pr_dist page_rank.cpp main.cpp page_rank_ref.o graph_dist_ref.o

intel: main.cpp intel_page_rank_ref.o ../include/graph_dist.h
	mpicc -cc=icc -O3 -std=c++11 -fopenmp -Wall -g -I../include/ -o pr_dist page_rank.cpp main.cpp intel_page_rank_ref.o graph_dist_ref_intel.o

clean:
	rm -rf pr_dist *~ *.*~
