EXECUTABLE := render
LDFLAGS=-L/data/cuda/cuda-10.2/cuda/lib64/ -lcudart
CU_FILES   := cudaRenderer.cu renderAlongWithPixel.cu
CU_DEPS    :=
CC_FILES   := main.cpp display.cpp benchmark.cpp refRenderer.cpp \
              noise.cpp ppm.cpp sceneLoader.cpp 
LOGS	   := logs

all: $(EXECUTABLE)

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       :=
FRAMEWORKS :=

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_75 -ccbin /usr/bin/gcc
LIBS += GL glut cudart

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/display.o $(OBJDIR)/benchmark.o $(OBJDIR)/refRenderer.o \
	 $(OBJDIR)/renderAlongWithPixel.o \
     $(OBJDIR)/cudaRenderer.o $(OBJDIR)/noise.o $(OBJDIR)/ppm.o $(OBJDIR)/sceneLoader.o \

CUOBJS=$(OBJDIR)/renderAlongWithPixel.o $(OBJDIR)/cudaRenderer.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) *.ppm

check:	default
		./checker.pl

export: $(EXFILES)
	cp -p $(EXFILES) $(STARTER)

link.o:
	$(NVCC) $(NVCCFLAGS) $(CUOBJS) -dlink -o $(OBJDIR)/link.o

$(EXECUTABLE): dirs $(OBJS) link.o
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(OBJDIR)/link.o $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS) -lcudadevrt




$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -dc -o $@ 
