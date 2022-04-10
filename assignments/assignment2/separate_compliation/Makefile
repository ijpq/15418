.PHONY: clean
all:
	nvcc --gpu-architecture=sm_75 -dc a.cu b.cu
	nvcc --gpu-architecture=sm_75 -dlink a.o b.o -o link.o
	g++ -m64 -Wall -c main.cpp -o main.o
	g++ -m64 -Wall a.o b.o link.o main.o -lcudart -L/data/cuda/cuda-10.2/cuda/lib64/
	./a.out

clean:
	rm -rf *.o a.out
