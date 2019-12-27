#ifndef _BFSCUDA_H_
#define _BFSCUDA_H_

__global__ void simpleBfs(int N, int level, int *d_adjacencyList, int *d_edgesOffset,
               int *d_edgesSize, int *d_distance, int *d_parent, int *changed);

__global__ void queueBfs(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
              int queueSize, int *nextQueueSize, int *d_currentQueue, int *d_nextQueue);

__global__ void nextLayer(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
                              int queueSize, int *d_currentQueue);

__global__ void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_parent,
                                                int queueSize, int *d_currentQueue, int *d_degrees) ;

__global__ void scanDegrees(int size, int *d_degrees, int *incrDegrees);

__global__ void assignVerticesNextQueue(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_parent, int queueSize,
                             int *d_currentQueue, int *d_nextQueue, int *d_degrees, int *incrDegrees,
                             int nextQueueSize);

#endif 