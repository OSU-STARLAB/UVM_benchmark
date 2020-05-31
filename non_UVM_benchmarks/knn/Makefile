NVCCCFLAGS =  -std=c++11 -lcuda -O0
NVCC = nvcc

knn: knn_cuda.cu 
	$(NVCC) ${NVCCCFLAGS} -o $@ $<

clean:
	rm -rf knn