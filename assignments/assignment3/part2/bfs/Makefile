main: bfs.cpp main.cpp bfs_ref.o ../include/graph_dist.h
	mpic++ -O3 -std=c++11 -fopenmp -Wall -g -I../include/ -o bfs_dist bfs.cpp main.cpp graph_dist_ref.o bfs_ref.o

intel: bfs.cpp main.cpp bfs_ref_intel.o ../include/graph_dist.h
	mpicc -cc=icc -O3 -std=c++11 -fopenmp -Wall -g -I../include/ -o bfs_dist bfs.cpp main.cpp graph_dist_ref_intel.o bfs_ref_intel.o

clean:
	rm -rf bfs_dist
