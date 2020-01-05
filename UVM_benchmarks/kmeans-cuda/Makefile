include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

all: kmeans_seq kmeans_cuda kmeans_standard 

kmeans_seq:
	g++ -std=c++11 kmeansSequential.cpp -o kmeans_sequential

kmeans_cuda:
	$(CC) -std=c++11 kmeans_cuda.cu -o kmeans_cuda

kmeans_standard:
	$(CC) -std=c++11 standard_kmeans.cu -o kmeans_standard

clean:
	rm kmeans_cuda kmeans_standard kmeans_sequential
