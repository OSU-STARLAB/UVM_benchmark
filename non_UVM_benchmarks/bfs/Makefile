CUDA_INSTALL_PATH ?= /usr/local/cuda-10.2/
GCC_VER = -5

CXX := /usr/bin/g++
CC := /usr/bin/gcc
CCPATH := gcc

NVCC  := $(CUDA_INSTALL_PATH)bin/nvcc -ccbin $(CCPATH)
LINK := $(NVCC)

INCLUDES := -I. -I$(CUDA_INSTALL_PATH)include
# Libraries
LIB_CUDA := -L/usr/lib/nvidia-current -lcuda

# Options
NVCCOPTIONS := -std=c++11  -Wno-deprecated-gpu-targets
CXXOPTIONS = -std=c++11 -O2

COMMONFLAGS += $(INCLUDES) 
NVCCFLAGS += $(NVCCOPTIONS)  $(COMMONFLAGS) 
CXXFLAGS += $(COMMONFLAGS) $(CXXOPTIONS)
CFLAGS += $(COMMONFLAGS)
TARGET = main
OBJS =  graph.c.o bfsCPU.c.o main.cu.o bfsCUDA.cu.o
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)



$(TARGET):  $(OBJS) 
	$(LINKLINE)

%.cu.o: %.cu 
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.c.o: %.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(TARGET) *.o 
