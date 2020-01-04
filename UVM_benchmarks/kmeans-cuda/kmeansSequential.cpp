#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <time.h>
#include <string>

double cpu_time_used;
#define I(row, col, ncols) (row * ncols + col)

void get_dst(float *dst, float *x, float *y,
             float *mu_x, float *mu_y, int n, int k){
  int i, j;
  for(i = 0; i < n; ++i){
    for(j = 0; j < k; ++j){
      dst[I(i, j, k)] = (x[i] - mu_x[j]) * (x[i] - mu_x[j]);
      dst[I(i, j, k)] += (y[i] - mu_y[j]) * (y[i] - mu_y[j]);
    }
  }
}

void regroup(int *group, float *dst, int n, int k){
  int i, j;
  float min_dst;

  for(i = 0; i < n; ++i){
    min_dst = dst[I(i, 0, k)];
    group[i] = 1;
    for(j = 1; j < k; ++j){
      if(dst[I(i, j, k)] < min_dst){
        min_dst = dst[I(i, j, k)];
        group[i] = j + 1;
      }
    }
  }
}

void clear(float *sum_x, float *sum_y, int *nx, int *ny, int k){
  int j;

  for(j = 0; j < k; ++ j){
    sum_x[j] = 0;
    sum_y[j] = 0;
    nx[j] = 0;
    ny[j] = 0;
  }
}

void recenter_step1(float *sum_x, float *sum_y, int *nx, int *ny,
                    float *x, float *y, int *group, int n, int k){
  int i, j;
  for(j = 0; j < k; ++j){
    for(i = 0; i < n; ++i){
      if(group[i] == (j + 1)){
        sum_x[j] += x[i];
        sum_y[j] += y[i];
        nx[j]++;
        ny[j]++;
      }
    }
  }
}

void recenter_step2(float *mu_x, float *mu_y, float *sum_x,
                    float *sum_y, int *nx, int *ny, int k){
  int j;
  for(j = 0; j < k; ++j){
    mu_x[j] = sum_x[j]/nx[j];
    mu_y[j] = sum_y[j]/ny[j];
  }
}

void kmeans(int nreps, int n, int k,
            float *x, float *y, float *mu_x, float *mu_y,
            int *group, int *nx, int *ny,
            float *sum_x, float *sum_y, float *dst){
  int i;
  for(i = 0; i < nreps; ++i){
    get_dst(dst, x, y, mu_x, mu_y, n, k);
    regroup(group, dst, n, k);
    clear(sum_x, sum_y, nx, ny, k);
    recenter_step1(sum_x, sum_y, nx, ny, x, y, group, n, k);
    recenter_step2(mu_x, mu_y, sum_x, sum_y, nx, ny, k);
   }
}

void read_data(float **x, float **y, float **mu_x, float **mu_y, int *n, int *k,char* argv);
void print_results(int *group, float *mu_x, float *mu_y, int n, int k,char* argv);

int main(int argc,char * argv[]){

  int n; /* number of points */
  int k; /* number of clusters */
  int *group, *nx, *ny;
  float *x = NULL, *y = NULL, *mu_x = NULL, *mu_y = NULL;
  float *sum_x, *sum_y, *dst;

  /* read data from files */
  read_data(&x, &y, &mu_x, &mu_y, &n, &k,argv[2]);

  /* allocate memory */
  group = (int*) malloc(n*sizeof(int));
  nx = (int*) malloc(k*sizeof(float));
  ny = (int*) malloc(k*sizeof(float));
  sum_x = (float*) malloc(k*sizeof(float));
  sum_y = (float*) malloc(k*sizeof(float));
  dst = (float*) malloc(n*k*sizeof(float));



  const auto start = std::chrono::high_resolution_clock::now();
  /* perform kmeans */
  kmeans(10, n, k, x, y, mu_x, mu_y, group, nx, ny, sum_x, sum_y, dst);


  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
  std::cerr << "Sequential Time taken " << duration.count() << "s" << " for "<<argv[3]<<" points." << std::endl;  
cpu_time_used = duration.count();
  /* print results and clean up */
  print_results(group, mu_x, mu_y, n, k,argv[3]);

  free(x);
  free(y);
  free(mu_x);
  free(mu_y);
  free(group);
  free(nx);
  free(ny);
  free(sum_x);
  free(sum_y);
  free(dst);

  return 0;
}

void read_data(float **x, float **y, float **mu_x, float **mu_y, int *n, int *k,char* arg){
  FILE *fp;
  char buf[64];

  *n = 0;
  fp = fopen(arg, "r");
  while(fgets(buf, 64, fp) != NULL){
    *n += 1;
    *x = (float*) realloc(*x, (*n)*sizeof(float));
    *y = (float*) realloc(*y, (*n)*sizeof(float));
    std::istringstream line_stream(buf);
    float x1,y1;
    line_stream >> x1 >> y1;
    (*x)[*n - 1] = x1;
    (*y)[*n - 1] = y1;
  }
  fclose(fp);

  *k = 0;
  fp = fopen("input/initCoord.txt", "r");
  while(fgets(buf, 64, fp) != NULL){
    *k += 1;
    *mu_x = (float*) realloc(*mu_x, (*k)*sizeof(float));
    *mu_y = (float*) realloc(*mu_y, (*k)*sizeof(float));
    std::istringstream line_stream(buf);
    float x1,y1;
    line_stream >> x1 >> y1;
    (*mu_x)[*k - 1] = x1;
    (*mu_y)[*k - 1] = x1;
  }
  fclose(fp);
}

void print_results(int *group, float *mu_x, float *mu_y, int n, int k,char* arg){
  FILE *fp;
  int i;
  std::string str(arg),str1,str2;
  str = "output/sequential/" + str;

  str1 = str + "_group_members.txt";
  fp = fopen(str1.c_str(), "w");
  for(i = 0; i < n; ++i){
    fprintf(fp, "%d\n", group[i]);
  }
  fclose(fp);
  
  str2 = str + "_centroids.txt";
  fp = fopen(str2.c_str(), "w");
  for(i = 0; i < k; ++i){
    fprintf(fp, "%0.3f %0.3f\n", mu_x[i], mu_y[i]);
  }

  fp = fopen("Sequentialtimes.txt", "a");
  fprintf(fp, "%0.6f\n", cpu_time_used);
  fclose(fp);

}
