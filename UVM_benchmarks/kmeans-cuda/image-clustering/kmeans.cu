
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>


// Number of threads
#define BLOCK_SIZE 16
#define GRID_SIZE 256


#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}}

// nCentroids and size on device
__constant__ int dev_nCentroids;
__constant__ int dev_size;

// global variables
int CLUSTER_BYTES = 0; // nCentroids * sizeof(int)
int IMAGE_BYTES = 0;  // width * height * sizeof(int)

//R,G,B Centroid's triple on device
__constant__ int dev_RedCentroid[1024];
__constant__ int dev_GreenCentroid[1024];
__constant__ int dev_BlueCentroid[1024];

void initialise_centroids(int nCentroids, int* redCentroid, int* greenCentroid, int*  blueCentroid, int r[], int g[], int b[], int size) {

	int i;

	for(i=0;i<nCentroids;++i)
	{
		int index = rand()%size;
		redCentroid[i] = r[index];
		blueCentroid[i] = b[index];
		greenCentroid[i] = g[index];
		
	
	}
}


bool loadRawImage(char* filename, int* r, int* g, int* b, int size) {
	FILE *imageFile;
	imageFile = fopen(filename, "r");

	if (imageFile == NULL) {
		return false;
	} else {
		for (int i = 0; i < size; i++) {

			r[i] = fgetc(imageFile);
			g[i] = fgetc(imageFile);
			b[i] = fgetc(imageFile);
		}
		fclose(imageFile);

		/*for(int j = 0; j < h * w; j++) {
			printf("%d, %d, %d ", r[j], g[j], b[j]);
		}*/
		return true;
	}
}



bool writeRawImage(char* filename, int* labelArray, int* redCentroid, int* greenCentroid, int* blueCentroid, int size){
	FILE *imageFile;
	imageFile = fopen(filename, "wb");

	if(imageFile == NULL) {
		return false;
	} else {
		for (int i = 0; i < size; i++) {
			fputc((char) redCentroid[labelArray[i]], imageFile);
			fputc((char) greenCentroid[labelArray[i]], imageFile);
			fputc((char) blueCentroid[labelArray[i]], imageFile);
		}
		fclose(imageFile);
		return true;
	}
}




 
__global__ void clearArrays(int *dev_sumRed,int *dev_sumGreen,int *dev_sumBlue, int* dev_pixelClusterCounter, int* dev_tempRedCentroid, int* dev_tempGreenCentroid, int* dev_tempBlueCentroid ) {

	// 1 block, 16x16 threads
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;

	if(threadID < dev_nCentroids) { 


		// nCentroids long
		dev_sumRed[threadID] = 0;
		dev_sumGreen[threadID] = 0;
		dev_sumBlue[threadID] = 0;
		dev_pixelClusterCounter[threadID] = 0;
		dev_tempRedCentroid[threadID] = 0;
		dev_tempGreenCentroid[threadID] = 0;
		dev_tempBlueCentroid[threadID] = 0;
	}
}
__global__ void clearLabelArray(int *dev_labelArray){

	// Global thread index
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
	//int threadID = (blockIdx.y*gridDim.x + blockIdx.x)*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	// labelArray is "size" long
	if(threadID < dev_size) {
		dev_labelArray[threadID] = 0;
	}
}



__global__ void getClusterLabel(int *dev_Red,int *dev_Green,int *dev_Blue,int *dev_labelArray) {


	
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
	
	float min = 500.0, value;
	
	int index = 0;


	if(threadID < dev_size) {
		
		for(int i = 0; i < dev_nCentroids; i++) {
			
			value = sqrtf(powf((dev_Red[threadID]-dev_RedCentroid[i]),2.0) + powf((dev_Green[threadID]-dev_GreenCentroid[i]),2.0) + powf((dev_Blue[threadID]-dev_BlueCentroid[i]),2.0));

			if(value < min){
				// saving new nearest centroid
				min = value;
				// Updating his index
				index = i;
			}
		}
		dev_labelArray[threadID] = index;

	}
}

__global__ void sumCluster(int *dev_Red,int *dev_Green,int *dev_Blue,int *dev_sumRed,int *dev_sumGreen,int *dev_sumBlue,int *dev_labelArray,int *dev_pixelClusterCounter) {

	
	int threadID = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x;
	
	if(threadID < dev_size) {
		int currentLabelArray = dev_labelArray[threadID];
		int currentRed = dev_Red[threadID];
		int currentGreen = dev_Green[threadID];
		int currentBlue = dev_Blue[threadID];
		// Writing to global memory needs a serialization. Many threads are writing into the same few locations
		atomicAdd(&dev_sumRed[currentLabelArray], currentRed);
		atomicAdd(&dev_sumGreen[currentLabelArray], currentGreen);
		atomicAdd(&dev_sumBlue[currentLabelArray], currentBlue);
		atomicAdd(&dev_pixelClusterCounter[currentLabelArray], 1);
	}
}

__global__ void updateCentroids(int *dev_tempRedCentroid, int *dev_tempGreenCentroid, int *dev_tempBlueCentroid,int* dev_sumRed, int *dev_sumGreen,int *dev_sumBlue, int* dev_pixelClusterCounter,int *dev_flag) {

	// 1 block , 16*16 threads
	int threadID = threadIdx.x + threadIdx.y * blockDim.x;

	if(threadID < dev_nCentroids) {
		int currentPixelCounter = dev_pixelClusterCounter[threadID];
		int sumRed = dev_sumRed[threadID];
		int sumGreen = dev_sumGreen[threadID];
		int sumBlue = dev_sumBlue[threadID];
		
		
		dev_tempRedCentroid[threadID] = (int)(sumRed/currentPixelCounter);
		
		dev_tempGreenCentroid[threadID] = (int)(sumGreen/currentPixelCounter);
		dev_tempBlueCentroid[threadID] = (int)(sumBlue/currentPixelCounter);
		
		if(dev_tempGreenCentroid[threadID]!=dev_GreenCentroid[threadID] || dev_tempRedCentroid[threadID]!=dev_RedCentroid[threadID] || dev_tempBlueCentroid[threadID]!=dev_BlueCentroid[threadID])
		*dev_flag=1;
	}

}

int main(int argc, char *argv[]) {

		cudaSetDevice(0);
		cudaDeviceSynchronize();
		cudaThreadSynchronize();

		
		char *inputFile, *outputFile;
		int *r, *g, *b, *redCentroid, *greenCentroid, *blueCentroid;
		int *dev_Red, *dev_Green, *dev_Blue, *dev_tempRedCentroid, *dev_tempGreenCentroid, *dev_tempBlueCentroid;
		int *labelArray, *dev_labelArray;

		
		int width, height, nCentroids, nIterations,size;
		//int IMAGE_BYTES, CLUSTER_BYTES;
		int *pixelClusterCounter, *dev_pixelClusterCounter;
		int *sumRed, *sumGreen, *sumBlue;
		int flag = 0;
		int *dev_sumRed, *dev_sumGreen, *dev_sumBlue;
		int *dev_flag;

		inputFile = argv[1];
		outputFile = argv[2];
		width = atoi(argv[3]);
		
		height = atoi(argv[4]);
		nCentroids = atoi(argv[5]);  
		nIterations = atoi(argv[6]);
		

		
		IMAGE_BYTES = width * height * sizeof(int);
		CLUSTER_BYTES = nCentroids * sizeof(int);
		size = width * height;


		printf("Image: %s\n",inputFile);
		printf("Width: %d, Height: %d\n", width, height);
		printf("#Clusters: %d, #Iterations: %d\n", nCentroids, nIterations);


		
		r = (int*)(malloc(IMAGE_BYTES));
		g = (int*)(malloc(IMAGE_BYTES));
		b = (int*)(malloc(IMAGE_BYTES));
		redCentroid = (int*)(malloc(CLUSTER_BYTES));
		greenCentroid = (int*)(malloc(CLUSTER_BYTES));
		blueCentroid = (int*)(malloc(CLUSTER_BYTES));
		labelArray = (int*)(malloc(IMAGE_BYTES));
		sumRed = (int*)(malloc(CLUSTER_BYTES));
		sumGreen = (int*)(malloc(CLUSTER_BYTES));
		sumBlue = (int*)(malloc(CLUSTER_BYTES));
		pixelClusterCounter = (int*)(malloc(CLUSTER_BYTES));

		
		printf("Image loading...\n");
		if (loadRawImage(inputFile, r, g, b, size)) {
			printf("Image loaded!\n");
		} else {
			printf("NOT loaded!\n");
			return -1;
		}
		cudaEvent_t start;
  		cudaEvent_t stop;
		cudaEventCreate(&start);
    		cudaEventCreate(&stop);


		
		printf("Initial Centroids: \n");
		
		initialise_centroids(nCentroids, redCentroid, greenCentroid, blueCentroid,r,g,b,size);

		printf("\n");	


		if(IMAGE_BYTES == 0 || CLUSTER_BYTES == 0) {
			return -1;
		}

		// allocate memory on GPU
		CUDA_CALL(cudaMalloc((void**) &dev_Red, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_Green, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_Blue, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_tempRedCentroid, CLUSTER_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_tempGreenCentroid, CLUSTER_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_tempBlueCentroid, CLUSTER_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_labelArray, IMAGE_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_sumRed, CLUSTER_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_sumGreen, CLUSTER_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_sumBlue, CLUSTER_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_pixelClusterCounter, CLUSTER_BYTES));
		CUDA_CALL(cudaMalloc((void**) &dev_flag, sizeof(int)));


		CUDA_CALL(cudaMemcpy(dev_Red, r, IMAGE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_Green, g, IMAGE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_Blue, b, IMAGE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_tempRedCentroid, redCentroid,CLUSTER_BYTES,cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy(dev_tempGreenCentroid, greenCentroid,CLUSTER_BYTES,cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy(dev_tempBlueCentroid, blueCentroid,CLUSTER_BYTES,cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy(dev_labelArray, labelArray, IMAGE_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpy(dev_flag,&flag,sizeof(int),cudaMemcpyHostToDevice));

		CUDA_CALL(cudaMemcpy(dev_pixelClusterCounter, pixelClusterCounter, CLUSTER_BYTES, cudaMemcpyHostToDevice));
		CUDA_CALL(cudaMemcpyToSymbol(dev_RedCentroid, redCentroid, CLUSTER_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_GreenCentroid, greenCentroid, CLUSTER_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_BlueCentroid, blueCentroid, CLUSTER_BYTES));
		CUDA_CALL(cudaMemcpyToSymbol(dev_nCentroids,&nCentroids, sizeof(int)));
		CUDA_CALL(cudaMemcpyToSymbol(dev_size, &size, sizeof(int)));


		
		for(int i = 0; i < nCentroids; i++) {
			redCentroid[i] = 0;
			greenCentroid[i] = 0;
			blueCentroid[i] = 0;
		}

		// Defining grid size

		int BLOCK_X, BLOCK_Y;
		BLOCK_X = ceil(width/BLOCK_SIZE);
		BLOCK_Y = ceil(height/BLOCK_SIZE);
		if(BLOCK_X > GRID_SIZE)
			BLOCK_X = GRID_SIZE;
		if(BLOCK_Y > GRID_SIZE)
			BLOCK_Y = GRID_SIZE;

	 	dim3 dimGRID(BLOCK_X,BLOCK_Y);
		dim3 dimBLOCK(BLOCK_SIZE,BLOCK_SIZE);

		//Starting timer
		cudaEventRecord(start, 0);
		printf("Launching K-Means Kernels..	\n");
		//Iteration of kmeans algorithm
		int num_iterations;
		for(int i = 0; i < nIterations; i++) {

			num_iterations = i;
			flag=0;
			CUDA_CALL(cudaMemcpy(dev_flag,&flag,sizeof(int),cudaMemcpyHostToDevice));
			
			
			clearArrays<<<1, dimBLOCK>>>(dev_sumRed, dev_sumGreen, dev_sumBlue, dev_pixelClusterCounter, dev_tempRedCentroid, dev_tempGreenCentroid, dev_tempBlueCentroid);

			
			clearLabelArray<<<dimGRID, dimBLOCK>>>(dev_labelArray);

	
			getClusterLabel<<< dimGRID, dimBLOCK >>> (dev_Red, dev_Green, dev_Blue,dev_labelArray);

			
			sumCluster<<<dimGRID, dimBLOCK>>> (dev_Red, dev_Green, dev_Blue, dev_sumRed, dev_sumGreen, dev_sumBlue, dev_labelArray,dev_pixelClusterCounter);

			
			updateCentroids<<<1,dimBLOCK >>>(dev_tempRedCentroid, dev_tempGreenCentroid, dev_tempBlueCentroid, dev_sumRed, dev_sumGreen, dev_sumBlue, dev_pixelClusterCounter,dev_flag);

		
			CUDA_CALL(cudaMemcpy(redCentroid, dev_tempRedCentroid, CLUSTER_BYTES,cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(greenCentroid, dev_tempGreenCentroid, CLUSTER_BYTES,cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(blueCentroid, dev_tempBlueCentroid, CLUSTER_BYTES,cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(&flag, dev_flag,sizeof(int),cudaMemcpyDeviceToHost));
			

			CUDA_CALL(cudaMemcpyToSymbol(dev_RedCentroid, redCentroid, CLUSTER_BYTES));
			CUDA_CALL(cudaMemcpyToSymbol(dev_GreenCentroid, greenCentroid, CLUSTER_BYTES));
			CUDA_CALL(cudaMemcpyToSymbol(dev_BlueCentroid, blueCentroid, CLUSTER_BYTES));

			if(flag==0)
				break;
			

		}
		cudaEventRecord(stop, 0);
		float elapsed;
		cudaEventSynchronize(stop);
   		cudaEventElapsedTime(&elapsed, start, stop);
		
		CUDA_CALL(cudaMemcpy(labelArray, dev_labelArray, IMAGE_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(sumRed, dev_sumRed, CLUSTER_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(sumGreen, dev_sumGreen, CLUSTER_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(sumBlue, dev_sumBlue, CLUSTER_BYTES, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(pixelClusterCounter, dev_pixelClusterCounter, CLUSTER_BYTES, cudaMemcpyDeviceToHost));

		
		printf("Kmeans code ran in: %f secs.\n", elapsed/1000.0);
		printf("Converged in %d iterations.\n",num_iterations);
		printf("\n");

	

		// labelArray DEBUG
		  int counter = 0;

		printf("Label Array:\n");
		for(int i = 0; i < (size); i++) {
			//printf("%d\n", labelArray[i]);
			counter++;
		}
		printf("printing counter %d\n", counter);
		counter = 0;

		printf("Sum Arrays:\n");
		for(int j = 0; j < nCentroids; j++) {
			printf("r: %u g: %u b: %u \n", sumRed[j], sumGreen[j], sumBlue[j]);
			counter++;
		}

		printf("\n");

		printf("Pixels per centroids:\n");
		for(int k = 0; k < nCentroids; k++){
			printf("%d centroid: %d pixels\n", k, pixelClusterCounter[k]);
		}

		printf("\n");



		printf("New centroids:\n");
		for(int i = 0; i < nCentroids; i++) {

			printf("%d, %d, %d \n", redCentroid[i], greenCentroid[i], blueCentroid[i]);
		}


		// writing...
		printf("Image writing...\n");

		if (writeRawImage(outputFile,labelArray, redCentroid, greenCentroid,  blueCentroid,  size)) {
			printf("Image written!\n");
		} else {
			printf("NOT written!\n");
			return -1;
		}

		free(r);
		free(g);
		free(b);
		free(redCentroid);
		free(greenCentroid);
		free(blueCentroid);
		free(labelArray);
		free(sumRed);
		free(sumGreen);
		free(sumBlue);
		free(pixelClusterCounter);

		CUDA_CALL(cudaFree(dev_Red));
		CUDA_CALL(cudaFree(dev_Green));
		CUDA_CALL(cudaFree(dev_Blue));
		CUDA_CALL(cudaFree(dev_tempRedCentroid));
		CUDA_CALL(cudaFree(dev_tempGreenCentroid));
		CUDA_CALL(cudaFree(dev_tempBlueCentroid));
		CUDA_CALL(cudaFree(dev_labelArray));
		CUDA_CALL(cudaFree(dev_sumRed));
		CUDA_CALL(cudaFree(dev_sumGreen));
		CUDA_CALL(cudaFree(dev_sumBlue));
		CUDA_CALL(cudaFree(dev_pixelClusterCounter));

		printf("That's the end.\n");
		return 0;
}
