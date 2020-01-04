#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
     
clock_t t;
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

void read_data(float **x, float **y, float **mu_x, float **mu_y, int *n, int *k,string strsize);
void print_results(int *group, float *mu_x, float *mu_y, int n, int k,string strsize);

int main(){
  string sizearr=["500","1000","5000","10000","50000","100000","500000"];
  int iterator=0;

for(iterator=0;iterator<7;iterator++){

  int n; /* number of points */
  int k; /* number of clusters */
  int *group, *nx, *ny;
  float *x = NULL, *y = NULL, *mu_x = NULL, *mu_y = NULL;
  float *sum_x, *sum_y, *dst;

  /* read data from files */
  read_data(&x, &y, &mu_x, &mu_y, &n, &k,sizearr[iterator]);

  /* allocate memory */
  group = (int*) malloc(n*sizeof(int));
  nx = (int*) malloc(k*sizeof(float));
  ny = (int*) malloc(k*sizeof(float));
  sum_x = (float*) malloc(k*sizeof(float));
  sum_y = (float*) malloc(k*sizeof(float));
  dst = (float*) malloc(n*k*sizeof(float));
     
  t = clock();

  /* perform kmeans */
  kmeans(10, n, k, x, y, mu_x, mu_y, group, nx, ny, sum_x, sum_y, dst);
  
  t = clock() - t;
  cpu_time_used = ((double)t)/CLOCKS_PER_SEC;
  printf("Sequential Time taken = %lf\n",cpu_time_used);
  
  /* print results and clean up */
  print_results(group, mu_x, mu_y, n, k);

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
}

  return 0;
}

void read_data(float **x, float **y, float **mu_x, float **mu_y, int *n, int *k,string strsize){
  FILE *fp;
  char buf[64];
  int i;

  *n = 0;
  fp = fopen(strcat(strcat("input/x_coordinates_",strsize),".txt"), "r");
  while(fgets(buf, 64, fp) != NULL){
    *n += 1;
    *x = (float*) realloc(*x, (*n)*sizeof(float));
    (*x)[*n - 1] = atof(buf);
  }
  fclose(fp);

  i = 0;
  fp = fopen(strcat(strcat("input/y_coordinates_",strsize),".txt"), "r");
  while(fgets(buf, 64, fp) != NULL){
    i += 1;
    *y = (float*) realloc(*y, i*sizeof(float));
    (*y)[i - 1] = atof(buf);
  }
  fclose(fp);
  
  if(i != *n){
    printf("ERROR: x.txt and y.txt must have same number of rows\n.");
    printf("That includes whitespace.\n");
    exit(EXIT_FAILURE);
  }
  
  *k = 0;
  fp = fopen("input/initialCluster_x_coordinates.txt", "r");
  while(fgets(buf, 64, fp) != NULL){
    *k += 1;
    *mu_x = (float*) realloc(*mu_x, (*k)*sizeof(float));
    (*mu_x)[*k - 1] = atof(buf);
  }
  fclose(fp);
  
  i = 0;
  fp = fopen("input/initialCluster_y_coordinates.txt", "r");
  while(fgets(buf, 64, fp) != NULL){
    i += 1;
    *mu_y = (float*) realloc(*mu_y, i*sizeof(float));
    (*mu_y)[i - 1] = atof(buf);
  }
  fclose(fp);
  
  if(i != *k){
    printf("ERROR: mu_x.txt and mu_y.txt must have same number of rows\n.");
    printf("That includes whitespace.\n");
    exit(EXIT_FAILURE);
  }
}

void print_results(int *group, float *mu_x, float *mu_y, int n, int k,string strsize){
  FILE *fp;
  int i;

  fp = fopen(strcat(strcat("output/sequential/cluster_members_",strsize),".txt"), "w");
  for(i = 0; i < n; ++i)
    fprintf(fp, "%d\n", group[i]);
  fclose(fp);
  
  fp = fopen(strcat(strcat("output/sequential/finalCluster_x_coordinates_",strsize),".txt"), "w");
  for(i = 0; i < k; ++i)
    fprintf(fp, "%0.3f\n", mu_x[i]);
  fclose(fp);
  
  fp = fopen(strcat(strcat("output/sequential/finalCluster_y_coordinates_",strsize),".txt"), "w");
  for(i = 0; i < k; ++i)
    fprintf(fp, "%0.3f\n", mu_y[i]);
  fclose(fp);
}
