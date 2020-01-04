# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n
include ../../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include
LIB := $(CUDA_DIR)/lib
# link objects(binaries) together
myocyte.out:		main.o
	$(CC)	main.o \
				-I$(INCLUDE) \
				-L$(LIB) \
				-lm -lcuda -lcudart \
                                -o myocyte.out

# compile main function file into object (binary)
main.o:	define.c \
				main.cu \
				work.cu \
				solver.cu \
				embedded_fehlberg_7_8.cu \
				master.cu \
				kernel.cu \
				kernel_ecc.cu \
				kernel_cam.cu \
				kernel_fin.cu \
				work_2.cu \
				solver_2.cu \
				embedded_fehlberg_7_8_2.cu \
				kernel_2.cu \
				kernel_ecc_2.cu \
				kernel_cam_2.cu \
				kernel_fin_2.cu \
				file.c \
				timer.c
	$(CC)	main.cu \
				-c -O3 -g

# delete all object files
clean:
	rm *.o *.out output.txt
