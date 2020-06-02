/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "../../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define N 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int i, j;
	
	for (i = 0; i < N; i++)
	{
		tmp[i] = 0;
		y[i] = 0;
		for (j = 0; j < N; j++)
		{
			tmp[i] = A[i*N + j] * x[j] + tmp[i];
			y[i] = B[i*N + j] * x[j] + y[i];
		}
		
		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}


void init(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* A_gpu, DATA_TYPE* x_gpu)
{
  	int i, j;

 	for (i = 0; i < N; i++)
    {
		x[i] = ((DATA_TYPE) i) / N;
		x_gpu[i] = ((DATA_TYPE) i) / N;
      	
		for (j = 0; j < N; j++) 
		{
			A[i*N + j] = ((DATA_TYPE) i*j) / N;
			A_gpu[i*N + j] = ((DATA_TYPE) i*j) / N;
		}
    }
}


void compareResults(DATA_TYPE* y, DATA_TYPE* y_outputFromGpu)
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<(N); i++) 
	{
		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gesummv_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j = 0; j < N; j++)
		{	
			tmp[i] += a[i * N + j] * x[j];
			y[i] += b[i * N + j] * x[j];
		}
		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}

void gesummvCuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* x_gpu, DATA_TYPE* y_gpu, DATA_TYPE* tmp_gpu)
{
	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStream_t stream3;
	cudaStream_t stream4;
	cudaStream_t stream5;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);
	cudaStreamCreate(&stream5);

	#ifdef PREF
	double t_start, t_end;		
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), 1);
	cudaMemPrefetchAsync(A_gpu,N*N*sizeof(DATA_TYPE), GPU_DEVICE, stream1 );
	cudaMemPrefetchAsync(B_gpu,N*N*sizeof(DATA_TYPE), GPU_DEVICE, stream2 );
	cudaMemPrefetchAsync(x_gpu,N*sizeof(DATA_TYPE), GPU_DEVICE, stream3 );
	cudaMemPrefetchAsync(y_gpu,N*sizeof(DATA_TYPE), GPU_DEVICE, stream4 );
	cudaMemPrefetchAsync(tmp_gpu,N*sizeof(DATA_TYPE), GPU_DEVICE, stream5 );
	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);
	cudaStreamSynchronize(stream3);
	cudaStreamSynchronize(stream4);
	cudaStreamSynchronize(stream5);
	t_start = rtclock();
	for (int i = 0; i < 1; i++){
	gesummv_kernel<<< grid, block, 0 , stream5>>>(A_gpu,B_gpu,x_gpu, y_gpu, tmp_gpu);
	cudaDeviceSynchronize();
	}
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	#else
	double t_start, t_end;		
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), 1);
	t_start = rtclock();
	for (int i = 0; i < 1; i++){
	gesummv_kernel<<< grid, block>>>(A_gpu,B_gpu,x_gpu, y_gpu, tmp_gpu);
	cudaDeviceSynchronize();
	}
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	#endif
}


int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* x;  
	DATA_TYPE* y;
	DATA_TYPE* tmp;
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;
	
	A = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE)); 
	y = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

	cudaMallocManaged((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMallocManaged((void **)&B_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMallocManaged((void **)&x_gpu, sizeof(DATA_TYPE) * N);
	cudaMallocManaged((void **)&y_gpu, sizeof(DATA_TYPE) * N);
	cudaMallocManaged((void **)&tmp_gpu, sizeof(DATA_TYPE) * N);

	init(A, x, A_gpu, x_gpu);
	
	GPU_argv_init();
	gesummvCuda(A_gpu, B_gpu, x_gpu, y_gpu, tmp_gpu);
	
	t_start = rtclock();
	gesummv(A, B, x, y, tmp);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(y, y_gpu);

	free(A);
	free(B);  
	free(x);  
	free(y);
	free(tmp);
	cudaFree(A_gpu);
	cudaFree(B_gpu);  
	cudaFree(x_gpu);  
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);

	return 0;
}

