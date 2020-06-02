

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

#define ITERATIONS 1

////////////////////////////////////////////////////////////////////////////////

extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern "C"
int setup(int argc, char** argv);

extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	setup(argc, argv);
}


extern "C"
void bpnn_train_cuda(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  
  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
  int m = 0;
  float *input_hidden_cuda;
  float *input_cuda;
  float *output_hidden_cuda;
  float *partial_sum;
  float *hidden_partial_sum;
  float *hidden_delta_cuda;
  float *input_prev_weights_cuda;
  float sum;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  num_blocks = in / 16;  
  dim3  grid( 1 , num_blocks);
  dim3  threads(16 , 16);


  cudaMallocManaged(&input_cuda, (in + 1) * sizeof(float));
  cudaMallocManaged(&output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMallocManaged(&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  cudaMallocManaged(&hidden_partial_sum, num_blocks * WIDTH * sizeof(float));
  cudaMallocManaged(&input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));
  memcpy(input_cuda,net->input_units, (in + 1)  *sizeof(float));
 
  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {	
   for (int j = 0; j <= hid; j++) {
	  input_hidden_cuda[m] = net->input_weights[k][j];
	  input_prev_weights_cuda[m] = net-> input_prev_weights[k][j];
	  m++;
    }
  }

  



#ifdef PREF
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
cudaMemPrefetchAsync(input_cuda,(in + 1) * sizeof(float), 0, stream1 );
cudaMemPrefetchAsync(output_hidden_cuda,(hid + 1) * sizeof(float), 0, stream2 );
cudaMemPrefetchAsync(input_hidden_cuda,(in + 1) * (hid + 1) * sizeof(float), 0, stream3 );
cudaMemPrefetchAsync(hidden_partial_sum,num_blocks * WIDTH * sizeof(float), 0, stream4 );
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
cudaStreamSynchronize(stream3);
cudaStreamSynchronize(stream4);
#endif


 
#ifdef PREF
printf("Performing GPU computation\n");  
for(int i = 0; i < ITERATIONS; i ++){
  bpnn_layerforward_CUDA<<< grid, threads, 0, stream5 >>>(input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid);
 
  cudaDeviceSynchronize();
}
#else
printf("Performing GPU computation\n");  
for(int i = 0; i < ITERATIONS; i ++){
  bpnn_layerforward_CUDA<<< grid, threads >>>(input_cuda,
	                                          output_hidden_cuda,
											  input_hidden_cuda,
											  hidden_partial_sum,
											  in,
											  hid);
 
  cudaDeviceSynchronize();
}

#endif

  
  
  // cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);
     
  for (int j = 1; j <= hid; j++) {
    sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {	
      sum += hidden_partial_sum[k * hid + j-1] ;
    }
	sum += net->input_weights[0][j];
	net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }


  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);





  cudaMallocManaged( &hidden_delta_cuda, (hid + 1) * sizeof(float));
  memcpy(hidden_delta_cuda, net->hidden_delta,(hid + 1)* sizeof(float));


  #ifdef PREF
  cudaMemPrefetchAsync(input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float), 0, stream1 );
  cudaMemPrefetchAsync(hidden_delta_cuda,(hid + 1) * sizeof(float), 0, stream2 );
  // cudaMemPrefetchAsync(input_cuda,(in + 1) * sizeof(float), 0, stream3 );
  // cudaMemPrefetchAsync(input_hidden_cuda,(in + 1) * (hid + 1) * sizeof(float), 0, stream4 );
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  // cudaStreamSynchronize(stream3);
  // cudaStreamSynchronize(stream4);
    for(int i = 0; i < ITERATIONS; i ++){
      bpnn_adjust_weights_cuda<<< grid, threads, 0, stream5 >>>(hidden_delta_cuda,  
                            hid, 
                            input_cuda, 
                            in,
                            input_hidden_cuda, 
                            input_prev_weights_cuda
                            );
      cudaDeviceSynchronize();
                          }

  #else

    for(int i = 0; i < ITERATIONS; i ++){
    bpnn_adjust_weights_cuda<<< grid, threads >>>(hidden_delta_cuda,  
                          hid, 
                          input_cuda, 
                          in,
                          input_hidden_cuda, 
                          input_prev_weights_cuda
                          );
    cudaDeviceSynchronize();
                        }
  #endif 
          
  memcpy(net->input_units, input_cuda, (in + 1)  * sizeof(float));
    
  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);

  
  
  

}
