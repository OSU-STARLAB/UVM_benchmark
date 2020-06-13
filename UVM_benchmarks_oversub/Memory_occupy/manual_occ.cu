#include <iostream>
#include <math.h>
#include <stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <assert.h>
int main(int argc, char *argv[])
{
    float *MB_ptr;
    int size_char_format = atoi(argv[1]);

    printf("size is  %d MB\n", size_char_format ); 
    cudaMalloc((void **) &MB_ptr, sizeof(float) * 1024*256 * size_char_format);
    sleep(900);
    cudaFree(MB_ptr);
    return 0;

}

