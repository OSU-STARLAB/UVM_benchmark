/*
by Qin Yu, Apr 2019
*/

#include <fstream>
using namespace std;

#include <cuda.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
                                  blockIdx.y*gridDim.x+blockIdx.x,\
                                  threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
                                  __VA_ARGS__)  // Idiom, not used, put here for convenient debugging.

__global__ void kernel_minibatch(int *iters, float *alpha, float *sigma,
                                 float *K, int *y, int l, int C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  int can_stop = 0;
  float delta_ = 0;
  __shared__ float delta;
  float last_alpha_j, last_alpha;

  int counter = 0;
  while (true) {
    counter++;
    last_alpha_j = alpha[j];
    for (int i = 0; i < l; i++) {
      if (j == i) {
        last_alpha = alpha[i];
        delta = 1 / K[i * l + i] * (1 - y[i] * sigma[i]);
        alpha[i] += delta;
        if (alpha[i] < 0) {
          alpha[i] = 0;
          delta = 0 - last_alpha;
        }
        if (alpha[i] > C) {
          alpha[i] = C;
          delta = C - last_alpha;
        }
      }
      __syncthreads();
      sigma[j] += delta * y[i] * K[i * l + j];
    }
    can_stop = 0;
    delta_ = alpha[j] - last_alpha_j;
    if (-0.0001f < delta_ && delta_ < 0.0001f) can_stop = 1;
    // CUPRINTF("%d, %9.6f, %9.6f, %9.6f, %d\n", counter, alpha[j],
    // last_alpha_j, delta_, can_stop);
    if (__syncthreads_and(can_stop) > 0) {
      if (j == 1) {
        // CUPRINTF("iters = %d\n", counter);
        iters[0] = counter;
      }
      break;
    }
  }
}

extern "C" __global__ void kernel_minibatch_g(int *iters, float *alpha,
                                              float *sigma, float *K, int *y,
                                              int *d, int ddim, float *delta,
                                              int l, int C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < l) {
    cg::grid_group grid = cg::this_grid();
    // if (j == l-1) CUPRINTF("l = %d, C = %d\n", l, C);

    int can_break = 0;
    int can_stop = 0;
    float delta_ = 0;
    float last_alpha_j, last_alpha;

    int counter = 0;
    while (true) {
      // for (int counter = 0; counter < 2000; counter++) {
      counter++;
      if (threadIdx.x == 1) d[blockIdx.x] = 0;
      last_alpha_j = alpha[j];
      for (uint32_t i = 0; i < l; i++) {
        if (j == i) {
          // if (threadIdx.x == i) {  // This was a big big bug
          last_alpha = alpha[i];
          delta[0] = 1 / K[i * l + i] * (1 - y[i] * sigma[i]);
          alpha[i] += delta[0];
          // alpha[i] += delta;
          if (alpha[i] < 0) {
            alpha[i] = 0;
            delta[0] = 0 - last_alpha;
          }
          if (alpha[i] > C) {
            alpha[i] = C;
            delta[0] = C - last_alpha;
          }
        }
        cg::sync(grid);
        sigma[j] += delta[0] * y[i] * K[i * l + j];
      }
      can_stop = 0;
      delta_ = alpha[j] - last_alpha_j;
      if (-0.0001f < delta_ && delta_ < 0.0001f) can_stop = 1;
      if (__syncthreads_and(can_stop) > 0)
        if (threadIdx.x == 1) d[blockIdx.x] = 1;
      cg::sync(grid);
      can_break = 0;
      for (int i = 0; i < ddim; i++) {
        can_break += d[i];
      }
      // if (j == 1) CUPRINTF("iters = %d\n", counter);
      if (can_break == ddim) {
        if (j == 1) {
          // CUPRINTF("iters = %d\n", counter);
          iters[0] = counter;
        }
        // cg::sync(grid);
        break;
      }
    }
  }
}

// Helper function for using CUDA to update sigma in parallel:
cudaError_t kernel_minibatch_wrapper(int *iters, float *alpha, float *sigma,
                                     float *K, int *y, int l, int C) {
  int *dev_iters = 0;
  float *dev_alpha = 0;
  float *dev_sigma = 0;
  float *dev_K = 0;
  int *dev_y = 0;
  int *dev_block_done = 0;
  float *dev_delta = 0;

  const int block_dim_max = 1024;
  int block_dimension = block_dim_max;
  int grid_dimension = (l - 1) / block_dim_max + 1;
  dim3 block(block_dimension);
  dim3 grid(grid_dimension);

  void *args[10] = {
      &dev_iters,      &dev_alpha,      &dev_sigma, &dev_K, &dev_y,
      &dev_block_done, &grid_dimension, &dev_delta, &l,     &C};

  cudaError_t cudaStatus;

  // Allocate GPU buffers for all vectors:
  cudaStatus = cudaMalloc(&dev_iters, sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error1;
  }
  cudaStatus = cudaMalloc(&dev_alpha, l * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error2;
  }
  cudaStatus = cudaMalloc(&dev_sigma, l * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error3;
  }
  cudaStatus = cudaMalloc(&dev_K, l * l * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error4;
  }
  cudaStatus = cudaMalloc(&dev_y, l * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error5;
  }
  cudaStatus =
      cudaMalloc(&dev_block_done, grid_dimension * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error5;
  }
  cudaStatus = cudaMalloc(&dev_delta, 1 * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error5;
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus =
      cudaMemcpy(dev_K, K, l * l * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error5;
  }
  cudaStatus = cudaMemcpy(dev_y, y, l * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error5;
  }

  // printf("READY TO CALL KERNEL\n");
  cudaStatus =
      cudaLaunchCooperativeKernel((void *)kernel_minibatch_g, grid, block, args,
                                  sizeof(float), cudaStream_t(0));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "kernel_minibatch_g launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    goto Error5;
  }

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "kernel_minibatch_g launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    goto Error5;
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "addKernel!\n",
            cudaStatus);
    goto Error5;
  }

  // Copy output vector from GPU buffer to host memory.
  cudaStatus =
      cudaMemcpy(iters, dev_iters, sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error5;
  }

  cudaStatus =
      cudaMemcpy(alpha, dev_alpha, l * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error5;
  }

  cudaStatus =
      cudaMemcpy(sigma, dev_sigma, l * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error5;
  }
  // Error6:
  cudaFree(dev_block_done);
  cudaFree(dev_delta);
Error5:
  cudaFree(dev_y);
Error4:
  cudaFree(dev_K);
Error3:
  cudaFree(dev_sigma);
Error2:
  cudaFree(dev_alpha);
Error1:
  cudaFree(dev_iters);
  // Error0:
  return cudaStatus;
}

cudaError_t kernel_minibatch_block_wrapper(int *iters, float *alpha,
                                           float *sigma, float *K, int *y,
                                           int l, int C) {
  int *dev_iters = 0;
  float *dev_alpha = 0;
  float *dev_sigma = 0;
  float *dev_K = 0;
  int *dev_y = 0;

  dim3 grid(1);
  dim3 block(l);
  void *args[7] = {&dev_iters, &dev_alpha, &dev_sigma, &dev_K, &dev_y, &l, &C};

  cudaError_t cudaStatus;

  // Allocate GPU buffers for all vectors:
  cudaStatus = cudaMalloc(&dev_iters, sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error1;
  }
  cudaStatus = cudaMalloc(&dev_alpha, l * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error2;
  }
  cudaStatus = cudaMalloc(&dev_sigma, l * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error3;
  }
  cudaStatus = cudaMalloc(&dev_K, l * l * sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error4;
  }
  cudaStatus = cudaMalloc(&dev_y, l * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error5;
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus =
      cudaMemcpy(dev_K, K, l * l * sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error5;
  }
  cudaStatus = cudaMemcpy(dev_y, y, l * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error5;
  }

  // printf("READY TO CALL KERNEL\n");
  cudaStatus = cudaLaunchKernel((void *)kernel_minibatch, grid, block, args,
                                sizeof(float), cudaStream_t(0));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "kernel_minibatch launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    goto Error5;
  }

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "kernel_minibatch_g launch failed: %s\n",
            cudaGetErrorString(cudaStatus));
    goto Error5;
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize returned error code %d after launching "
            "addKernel!\n",
            cudaStatus);
    goto Error5;
  }

  // Copy output vector from GPU buffer to host memory.
  cudaStatus =
      cudaMemcpy(iters, dev_iters, sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error5;
  }

  cudaStatus =
      cudaMemcpy(alpha, dev_alpha, l * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error5;
  }

  cudaStatus =
      cudaMemcpy(sigma, dev_sigma, l * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error5;
  }
Error5:
  cudaFree(dev_y);
Error4:
  cudaFree(dev_K);
Error3:
  cudaFree(dev_sigma);
Error2:
  cudaFree(dev_alpha);
Error1:
  cudaFree(dev_iters);
  // Error0:
  return cudaStatus;
}
