.PHONY: clean
clean:
	rm -rf *.o a.out
all:
	nvcc --gpu-architecture=sm_75 --device-c a.cu b.cu
	nvcc --gpu-architecture=sm_75 --device-link a.o b.o --output-file link.o
	g++ -m64 -Wall a.o b.o link.o main.cpp -lcudart -L/data/cuda/cuda-10.2/cuda/lib64/
	./a.out

