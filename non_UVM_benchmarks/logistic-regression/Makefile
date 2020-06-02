

################################ Macros #################################

SHELL = /bin/sh
CC = g++
# Enable debug options
# CFLAGS = -g -Wall -std=c++11
# Enable best optimization options
CFLAGS =   -std=c++11
OBJECTS = Helper.o ArffImporter.o

# Enable Nvidia gpu
NVCC = nvcc
NVCCCFLAGS = -arch=sm_60 -std=c++11 -use_fast_math -lcublas# -lcublas_device -rdc=true -lcudadevrt

################################ Compile ################################

gpu_exec: ${OBJECTS} LogisticRegression_CuBLAS.cu
	$(NVCC) ${NVCCCFLAGS} -o $@ ${OBJECTS} LogisticRegression_CuBLAS.cu

Helper.o: Helper.c Helper.h BasicDataStructures.h
	$(CC) ${CFLAGS} -c Helper.c

ArffImporter.o: ArffImporter.cu ArffImporter.cuh BasicDataStructures.h Helper.h
	$(NVCC) ${CFLAGS} -c ArffImporter.cu

# exec: ${OBJECTS} LogisticRegression_Serial.c
# 	$(CC) ${CFLAGS} -o $@ ${OBJECTS} LogisticRegression_Serial.c

# Helper.o: Helper.c Helper.h BasicDataStructures.h
# 	$(CC) ${CFLAGS} -c Helper.c

# ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.h
# 	$(CC) ${CFLAGS} -c ArffImporter.cpp

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch *exec*
