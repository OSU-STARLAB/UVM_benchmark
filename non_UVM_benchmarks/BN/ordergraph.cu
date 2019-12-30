#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
//#include <cutil.h>
// includes CUDA
#include <cuda_runtime.h>
// includes, kernels
#include "ordergraph_kernel.cu"
;

const int HIGHEST = 3;
int taskperthr = 1;
int sizepernode;
int ITER = 100;

// global var
float preScore = -99999999999;
float score = 0.0;
float maxScore[HIGHEST] = {-999999999};
bool orders[NODE_N][NODE_N];
bool preOrders[NODE_N][NODE_N];
bool preGraph[NODE_N][NODE_N];
bool bestGraph[HIGHEST][NODE_N][NODE_N];
bool graph[NODE_N][NODE_N];
// float prior[NODE_N][NODE_N];
float *localscore, *D_localscore, *D_Score, *scores;
float *LG;
bool *D_parent;
int *D_resP, *parents;

void initial();  // initial orders and data
int genOrders(); // swap
int ConCore();   // discard new order or not
bool getparent(int *bit, int *pre, int posN, int *parent, int *parN,
               int time);    // get every possible set of parents for a node
void incr(int *bit, int n);  // binary code increases 1 each time
void incrS(int *bit, int n); // STATE_N code increases 1 each time
bool getState(
    int parN, int *state,
    int time); // get every possible combination of state for a parent set
float logGamma(int N); // log and gamma
float findBestGraph();
void genScore();
int convert(int *parent, int parN);
void sortGraph();
void swap(int a, int b);
void Pre_logGamma();
int findindex(int *arr, int size);
int C(int n, int a);

FILE *fpout;

int main() {

  /*
      for(i=0;i<NODE_N;i++){
              for(j=0;j<NODE_N;j++)
              prior[i][j]=0.5;
      }
  */
  int i, j, c = 0, tmp, a, b;
  float tmpd;
  fpout = fopen(name, "w");

  clock_t start, finish, total = 0, pre1, pre2;
  cudaDeviceSynchronize();

  printf("NODE_N=%d\nInitialization...\n", NODE_N);
  pre1 = clock();

  srand(time(NULL));
  initial();
  genScore();
  pre2 = clock();
  printf("OK, begin to generate orders.\n");

  i = 0;
  while (i != ITER) {

    start = clock();

    i++;
    score = 0;

    for (a = 0; a < NODE_N; a++) {
      for (j = 0; j < NODE_N; j++) {
        orders[a][j] = preOrders[a][j];
      }
    }

    tmp = rand() % 6;
    for (j = 0; j < tmp; j++)
      genOrders();

    score = findBestGraph();

    finish = clock();
    total += finish - start;

    ConCore();

    // store the top HIGHEST highest orders
    if (c < HIGHEST) {
      tmp = 1;
      for (j = 0; j < c; j++) {
        if (maxScore[j] == preScore) {
          tmp = 0;
        }
      }
      if (tmp != 0) {
        maxScore[c] = preScore;
        for (a = 0; a < NODE_N; a++) {
          for (b = 0; b < NODE_N; b++) {
            bestGraph[c][a][b] = preGraph[a][b];
          }
        }
        c++;
      }

    } else if (c == HIGHEST) {
      sortGraph();
      c++;
    } else {

      tmp = 1;
      for (j = 0; j < HIGHEST; j++) {
        if (maxScore[j] == preScore) {
          tmp = 0;
          break;
        }
      }
      if (tmp != 0 && preScore > maxScore[HIGHEST - 1]) {
        maxScore[HIGHEST - 1] = preScore;
        for (a = 0; a < NODE_N; a++) {
          for (b = 0; b < NODE_N; b++) {
            bestGraph[HIGHEST - 1][a][b] = preGraph[a][b];
          }
        }
        b = HIGHEST - 1;
        for (a = HIGHEST - 2; a >= 0; a--) {
          if (maxScore[b] > maxScore[a]) {
            swap(a, b);
            tmpd = maxScore[a];
            maxScore[a] = maxScore[b];
            maxScore[b] = tmpd;
            b = a;
          }
        }
      }
    }

  } // endwhile

  cudaFreeHost(localscore);
  cudaFree(D_localscore);
  cudaFree(D_parent);

  cudaFreeHost(scores);
  cudaFreeHost(parents);
  cudaFree(D_Score);
  cudaFree(D_resP);

  /*
          for(j=0;j<HIGHEST;j++){

                          fprintf(fpout,"score:%f\n",maxScore[j]);
                          fprintf(fpout,"Best Graph:\n");
                          for(int a=0;a<NODE_N;a++){
                                  for(int b=0;b<NODE_N;b++){
                                          fprintf(fpout,"%d
     ",bestGraph[j][a][b]);
                                  }
                                  fprintf(fpout,"%\n");
                          }

                          fprintf(fpout,"--------------------------------------------------------------------\n");
                  }*/

  fprintf(fpout, "Duration per interation is %f seconds.\n",
          ((float)total / ITER) / CLOCKS_PER_SEC);
  fprintf(fpout, "Total duration is %f seconds.\n",
          (float)(pre2 - pre1 + total) / CLOCKS_PER_SEC);
  fprintf(fpout, "Preprocessing duration is %f seconds.\n",
          (float)(pre2 - pre1) / CLOCKS_PER_SEC);

  printf("Duration per interation is %f seconds.\n",
         ((float)total / ITER) / CLOCKS_PER_SEC);
  printf("Total duration is %f seconds.\n",
         (float)(pre2 - pre1 + total) / CLOCKS_PER_SEC);
  printf("Preprocessing duration is %f seconds.\n",
         (float)(pre2 - pre1) / CLOCKS_PER_SEC);

  return 0;
}

void sortGraph() {
  float max = -99999999999999;
  int maxi, i, j;
  float tmp;

  for (j = 0; j < HIGHEST - 1; j++) {
    max = maxScore[j];
    maxi = j;
    for (i = j + 1; i < HIGHEST; i++) {
      if (maxScore[i] > max) {
        max = maxScore[i];
        maxi = i;
      }
    }

    swap(j, maxi);
    tmp = maxScore[j];
    maxScore[j] = max;
    maxScore[maxi] = tmp;
  }
}

void swap(int a, int b) {
  int i, j;
  bool tmp;

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++) {

      tmp = bestGraph[a][i][j];
      bestGraph[a][i][j] = bestGraph[b][i][j];
      bestGraph[b][i][j] = tmp;
    }
  }
}

void initial() {
  int i, j, tmp, a, b, r;
  bool tmpd;
  tmp = 1;
  for (i = 1; i <= 4; i++) {
    tmp += C(NODE_N - 1, i);
  }
  sizepernode = tmp;
  tmp *= NODE_N;

  cudaMallocHost((void **)&localscore, tmp * sizeof(float));

  for (i = 0; i < tmp; i++)
    localscore[i] = 0;

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++)
      orders[i][j] = 0;
  }
  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < i; j++)
      orders[i][j] = 1;
  }
  r = rand() % 10000;
  for (i = 0; i < r; i++) {
    a = rand() % NODE_N;
    b = rand() % NODE_N;
    for (j = 0; j < NODE_N; j++) {
      tmpd = orders[j][a];
      orders[j][a] = orders[j][b];
      orders[j][b] = tmpd;
    }

    for (j = 0; j < NODE_N; j++) {
      tmpd = orders[a][j];
      orders[a][j] = orders[b][j];
      orders[b][j] = tmpd;
    }
  }

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++) {
      preOrders[i][j] = orders[i][j];
    }
  }
}

// generate ramdom order
int genOrders() {

  int a, b, j;
  bool tmp;
  a = rand() % NODE_N;
  b = rand() % NODE_N;

  for (j = 0; j < NODE_N; j++) {
    tmp = orders[a][j];
    orders[a][j] = orders[b][j];
    orders[b][j] = tmp;
  }
  for (j = 0; j < NODE_N; j++) {
    tmp = orders[j][a];
    orders[j][a] = orders[j][b];
    orders[j][b] = tmp;
  }

  return 1;
}

// decide leave or discard an order
int ConCore() {
  int i, j;
  float tmp;
  tmp = log((rand() % 100000) / 100000.0);
  if (tmp < (score - preScore)) {

    for (i = 0; i < NODE_N; i++) {
      for (j = 0; j < NODE_N; j++) {
        preOrders[i][j] = orders[i][j];
        preGraph[i][j] = graph[i][j];
      }
    }
    preScore = score;

    return 1;
  }

  return 0;
}

void genScore() {
  int *D_data;
  float *D_LG;
  dim3 grid(sizepernode / 256 + 1, 1, 1);
  dim3 threads(256, 1, 1);

  Pre_logGamma();
  // cudaPrintfInit();
  cudaMalloc((void **)&D_data, NODE_N * DATA_N * sizeof(int));
  cudaMalloc((void **)&D_localscore, NODE_N * sizepernode * sizeof(float));
  cudaMalloc((void **)&D_LG, (DATA_N + 2) * sizeof(float));
  cudaMemset(D_localscore, 0.0, NODE_N * sizepernode * sizeof(float));
  cudaMemcpy(D_data, data, NODE_N * DATA_N * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(D_LG, LG, (DATA_N + 2) * sizeof(float), cudaMemcpyHostToDevice);
  genScoreKernel<<<grid, threads>>>(sizepernode, D_localscore, D_data, D_LG);
  cudaDeviceSynchronize();
  cudaMemcpy(localscore, D_localscore, NODE_N * sizepernode * sizeof(float),
             cudaMemcpyDeviceToHost);

  // cudaPrintfDisplay(stdout, true);
  // cudaPrintfEnd();

  cudaFreeHost(LG);
  cudaFree(D_LG);
  cudaFree(D_data);

  cudaMallocHost((void **)&scores,
                 (sizepernode / (256 * taskperthr) + 1) * sizeof(float));
  cudaMallocHost((void **)&parents,
                 (sizepernode / (256 * taskperthr) + 1) * 4 * sizeof(int));
  cudaMalloc((void **)&D_Score,
             (sizepernode / (256 * taskperthr) + 1) * sizeof(float));
  cudaMalloc((void **)&D_parent, NODE_N * sizeof(bool));
  cudaMalloc((void **)&D_resP,
             (sizepernode / (256 * taskperthr) + 1) * 4 * sizeof(int));
}

int convert(int *parent, int parN) {
  int i, j, w = 1, tmp = 0;
  j = 0;
  for (i = 0; parN > 0 && i <= parent[parN - 1]; i++) {
    if (parent[j] == i) {
      j++;
      tmp += w;
    }
    w *= 2;
  }

  return tmp;
}

void Pre_logGamma() {

  cudaMallocHost((void **)&LG, (DATA_N + 2) * sizeof(float));

  LG[1] = log(1.0);
  float i;
  for (i = 2; i <= DATA_N + 1; i++) {
    LG[(int)i] = LG[(int)i - 1] + log((float)i);
  }
}

void incr(int *bit, int n) {

  bit[n]++;
  if (bit[n] >= 2) {
    bit[n] = 0;
    incr(bit, n + 1);
  }

  return;
}

void incrS(int *bit, int n) {

  bit[n]++;
  if (bit[n] >= STATE_N) {
    bit[n] = 0;
    incr(bit, n + 1);
  }

  return;
}

bool getState(int parN, int *state, int time) {
  int j = 1;

  j = pow(STATE_N, (float)parN) - 1;

  if (time > j)
    return false;

  if (time >= 1)
    incrS(state, 0);

  return true;
}

bool getparent(int *bit, int *pre, int posN, int *parent, int *parN, int time) {
  int i, j = 1;

  *parN = 0;
  if (time == 0)
    return true;

  for (i = 0; i < posN; i++) {
    j = j * 2;
  }
  j--;

  if (time > j)
    return false;

  incr(bit, 0);

  for (i = 0; i < posN; i++) {
    if (bit[i] == 1) {
      parent[(*parN)++] = pre[i];
    }
  }

  return true;
}

float findBestGraph() {
  float bestls = -99999999;
  int bestparent[5];
  int bestpN, total;
  int node, index;
  int pre[NODE_N] = {0};
  int parent[NODE_N] = {0};
  int posN = 0, i, j, parN, tmp, k, l;
  float ls = -99999999999, score = 0;
  int blocknum;

  for (i = 0; i < NODE_N; i++)
    for (j = 0; j < NODE_N; j++)
      graph[i][j] = 0;

  for (node = 0; node < NODE_N; node++) {

    bestls = -99999999;
    posN = 0;

    for (i = 0; i < NODE_N; i++) {
      if (orders[node][i] == 1) {
        pre[posN++] = i;
      }
    }

    if (posN >= 0) {
      total = C(posN, 4) + C(posN, 3) + C(posN, 2) + posN + 1;
      taskperthr = 1;
      blocknum = total / (256 * taskperthr) + 1;

      cudaMemset(D_resP, 0, blocknum * 4 * sizeof(int));
      cudaMemset(D_Score, -999999.0, blocknum * sizeof(float));
      cudaMemcpy(D_parent, orders[node], NODE_N * sizeof(bool),
                 cudaMemcpyHostToDevice);

      computeKernel<<<blocknum, 256, 256 * sizeof(float)>>>(
          taskperthr, sizepernode, D_localscore, D_parent, node, total, D_Score,
          D_resP);
      cudaDeviceSynchronize();
      cudaMemcpy(parents, D_resP, blocknum * 4 * sizeof(int),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(scores, D_Score, blocknum * sizeof(float),
                 cudaMemcpyDeviceToHost);

      for (i = 0; i < blocknum; i++) {

        if (scores[i] > bestls) {

          bestls = scores[i];

          parN = 0;
          for (tmp = 0; tmp < 4; tmp++) {
            if (parents[i * 4 + tmp] < 0)
              break;

            bestparent[tmp] = parents[i * 4 + tmp];

            parN++;
          }

          bestpN = parN;
        }
      }
    } else {
      if (posN >= 4) {
        for (i = 0; i < posN; i++) {
          for (j = i + 1; j < posN; j++) {
            for (k = j + 1; k < posN; k++) {
              for (l = k + 1; l < posN; l++) {
                parN = 4;
                if (pre[i] > node)
                  parent[1] = pre[i];
                else
                  parent[1] = pre[i] + 1;
                if (pre[j] > node)
                  parent[2] = pre[j];
                else
                  parent[2] = pre[j] + 1;
                if (pre[k] > node)
                  parent[3] = pre[k];
                else
                  parent[3] = pre[k] + 1;
                if (pre[l] > node)
                  parent[4] = pre[l];
                else
                  parent[4] = pre[l] + 1;

                index = findindex(parent, parN);
                index += sizepernode * node;
                ls = localscore[index];

                if (ls > bestls) {
                  bestls = ls;
                  bestpN = parN;
                  for (tmp = 0; tmp < parN; tmp++)
                    bestparent[tmp] = parent[tmp + 1];
                }
              }
            }
          }
        }
      }

      if (posN >= 3) {
        for (i = 0; i < posN; i++) {
          for (j = i + 1; j < posN; j++) {
            for (k = j + 1; k < posN; k++) {

              parN = 3;
              if (pre[i] > node)
                parent[1] = pre[i];
              else
                parent[1] = pre[i] + 1;
              if (pre[j] > node)
                parent[2] = pre[j];
              else
                parent[2] = pre[j] + 1;
              if (pre[k] > node)
                parent[3] = pre[k];
              else
                parent[3] = pre[k] + 1;

              index = findindex(parent, parN);
              index += sizepernode * node;
              ls = localscore[index];

              if (ls > bestls) {
                bestls = ls;
                bestpN = parN;
                for (tmp = 0; tmp < parN; tmp++)
                  bestparent[tmp] = parent[tmp + 1];
              }
            }
          }
        }
      }

      if (posN >= 2) {
        for (i = 0; i < posN; i++) {
          for (j = i + 1; j < posN; j++) {

            parN = 2;
            if (pre[i] > node)
              parent[1] = pre[i];
            else
              parent[1] = pre[i] + 1;
            if (pre[j] > node)
              parent[2] = pre[j];
            else
              parent[2] = pre[j] + 1;

            index = findindex(parent, parN);
            index += sizepernode * node;
            ls = localscore[index];

            if (ls > bestls) {
              bestls = ls;
              bestpN = parN;
              for (tmp = 0; tmp < parN; tmp++)
                bestparent[tmp] = parent[tmp + 1];
            }
          }
        }
      }

      if (posN >= 1) {
        for (i = 0; i < posN; i++) {

          parN = 1;
          if (pre[i] > node)
            parent[1] = pre[i];
          else
            parent[1] = pre[i] + 1;

          index = findindex(parent, parN);
          index += sizepernode * node;
          ls = localscore[index];

          if (ls > bestls) {
            bestls = ls;
            bestpN = parN;
            for (tmp = 0; tmp < parN; tmp++)
              bestparent[tmp] = parent[tmp + 1];
          }
        }
      }

      parN = 0;
      index = sizepernode * node;

      ls = localscore[index];

      if (ls > bestls) {
        bestls = ls;
        bestpN = 0;
      }
    }
    if (bestls > -99999999) {

      for (i = 0; i < bestpN; i++) {
        if (bestparent[i] < node)
          graph[node][bestparent[i] - 1] = 1;
        else
          graph[node][bestparent[i]] = 1;
      }
      score += bestls;
    }
  }

  return score;
}

int findindex(int *arr, int size) { // reminder: arr[0] has to be 0 && size ==
                                    // array size-1 && index start from 0
  int i, j, index = 0;

  for (i = 1; i < size; i++) {
    index += C(NODE_N - 1, i);
  }

  for (i = 1; i <= size - 1; i++) {
    for (j = arr[i - 1] + 1; j <= arr[i] - 1; j++) {
      index += C(NODE_N - 1 - j, size - i);
    }
  }

  index += arr[size] - arr[size - 1];

  return index;
}

int C(int n, int a) {
  int i, res = 1, atmp = a;

  for (i = 0; i < atmp; i++) {
    res *= n;
    n--;
  }

  for (i = 0; i < atmp; i++) {
    res /= a;
    a--;
  }

  return res;
}
