include ../../common/make.config


CC := $(CUDA_DIR)/bin/nvcc

all:
	$(CC) -lcuda -lcublas *.cu -o CNN

run:
	./CNN
clean:
	rm -f CNN
