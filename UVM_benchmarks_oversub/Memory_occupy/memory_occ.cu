#include <iostream>
#include <math.h>
#include <stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <assert.h>

// #define GB 8
// #define MB 1
// #define KB 1

int main(int argc, char *argv[])
{
    unsigned GB,MB,KB;
    char *endptr;

    fprintf(stdout, "Input %d argument\n",argc);
    assert(argc == 3);
    // float size_char_format = atoi(argv[1]);
    float size = strtof(argv[1], &endptr);
    if (*endptr != '\0')
        printf("%s is not convertible to float\n", argv[1]);
    else
        printf("size is  %f\n", size ); //0.000000    
    int over_sub_char_format = atoi(argv[2]);
    // float size = (float) size_char_format;
    unsigned over_sub = (unsigned) over_sub_char_format;

    if (over_sub == 1)
    {
        // printf("size is %f\n", size);
        fprintf(stdout, "It is 110%% oversubscription mode\n");
        float occupy_mem = 11  - size   +  size * 1.0/10.0 -0.1 ;
        GB = (unsigned) occupy_mem;
        float temp_MB = (occupy_mem -floor(occupy_mem)) *1024.0;
        MB = (unsigned) temp_MB;
        float temp_KB = (temp_MB - floor(temp_MB)) *1024.0;
        KB = (unsigned) temp_KB;
    }
    else if (over_sub == 2)
    {
        printf("size is %f\n", size);
        fprintf(stdout, "It is 125%% oversubscription mode\n");
        float occupy_mem = 11  - size   +  size * 1.0/4.0 - 0.1  ; 
        GB = (unsigned) occupy_mem;
        float temp_MB = (occupy_mem -floor(occupy_mem)) *1024.0;
        MB = (unsigned) temp_MB;
        float temp_KB = (temp_MB - floor(temp_MB)) *1024.0;
        KB = (unsigned) temp_KB;
    }else{
        assert(0);
    }
    
    if (GB == 0)
    {
        assert(0);
    }
    if(MB == 0)
    {
        assert(0);
    }
    if(KB == 0)
    {
        assert(0);
    }
    
    fprintf(stdout, "the calculated size is %u GB, %u MB,%u KB \n",GB,MB,KB);
    

    float *GB_ptr;
    float *MB_ptr;
    float *KB_ptr;
    
    cudaMalloc((void **) &GB_ptr, sizeof(float) * 1024*1024*256 * GB );
    cudaMalloc((void **) &MB_ptr, sizeof(float) * 1024*256 * MB);
    cudaMalloc((void **) &KB_ptr, sizeof(float) * 256 * KB);
    
    sleep(900);
    sleep(900);
    sleep(900);
    sleep(900);
    sleep(900);
    

    cudaFree(GB_ptr);
    cudaFree(MB_ptr);
    cudaFree(KB_ptr);
    return 0;
}