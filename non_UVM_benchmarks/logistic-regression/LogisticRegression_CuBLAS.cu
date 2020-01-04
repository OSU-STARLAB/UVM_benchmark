#include "Helper.h"
#include "ArffImporter.h"

#include <cublas_v2.h>


#define MAX_ITER      1000
#define LEARNING_RATE 10.0f

Node initNode( unsigned int numFeatures )
{
    Node node;
    node.numFeatures = numFeatures;
    node.weights = (float*) malloc( numFeatures * sizeof( float ) );
    memset( node.weights, 0, numFeatures * sizeof( float ) );

    return node;
}

__global__ void ComputeCost(
    float* __restrict__ dCostArr,
    const unsigned short* __restrict__ dClassArr,
    const unsigned int numInstances )
{
    unsigned int instanceId = blockIdx.x * blockDim.x + threadIdx.x;
    if (instanceId >= numInstances) return;

    float cost = dCostArr[instanceId];
    cost = 1.0f / (1.0f + expf(-cost)) - (float) dClassArr[instanceId];
    dCostArr[instanceId] = cost;
}

inline void cudaErrorCheck( cudaError_t cudaStatus )
{
    if (cudaStatus != cudaSuccess)
        printf(
            "kernel launch failed with error \"%s\".\n",
            cudaGetErrorString( cudaStatus )
        );
}

inline void cublasErrorCheck( cublasStatus_t cublasStatus )
{
    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        printf( "CuBLAS launch failed with error\n" );
        switch (cublasStatus)
        {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                printf( "CUBLAS_STATUS_NOT_INITIALIZED\n" );

            case CUBLAS_STATUS_ALLOC_FAILED:
                printf( "CUBLAS_STATUS_ALLOC_FAILED\n" );

            case CUBLAS_STATUS_INVALID_VALUE:
                printf( "CUBLAS_STATUS_INVALID_VALUE\n" );

            case CUBLAS_STATUS_ARCH_MISMATCH:
                printf( "CUBLAS_STATUS_ARCH_MISMATCH\n" );

            case CUBLAS_STATUS_MAPPING_ERROR:
                printf( "CUBLAS_STATUS_MAPPING_ERROR\n" );

            case CUBLAS_STATUS_EXECUTION_FAILED:
                printf( "CUBLAS_STATUS_EXECUTION_FAILED\n" );

            case CUBLAS_STATUS_INTERNAL_ERROR:
                printf( "CUBLAS_STATUS_INTERNAL_ERROR\n" );
        }
    }
}

int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    // ArffImporter testSetImporter;
    // testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    // Init host data
    float* featureMatTrans = trainSetImporter.GetFeatureMatTrans();
    unsigned short* classArr = trainSetImporter.GetClassIndex();
    unsigned int numInstances = trainSetImporter.GetNumInstances();
    unsigned int numFeatures = trainSetImporter.GetNumFeatures();
    Node node = initNode( numFeatures );

    // Init device data
    float* dCostArr = nullptr;
    float* dWeightArr = nullptr;
    float* dFeatureMatTrans = nullptr;
    float* dFeaCostProdArr = nullptr;
    unsigned short* dClassArr = nullptr;
    // One instance per row, one feature per column, as cublas prefers column-major matrix (faster)
    cudaErrorCheck( cudaMalloc( (void**) &dFeatureMatTrans, numInstances * numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dWeightArr, numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dCostArr, numInstances * sizeof( float ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dClassArr, numInstances * sizeof( unsigned short ) ) );
    cudaErrorCheck( cudaMalloc( (void**) &dFeaCostProdArr, numFeatures * sizeof( float ) ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dFeatureMatTrans,
        featureMatTrans,
        numInstances * numFeatures * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dWeightArr,
        node.weights,
        numFeatures * sizeof( float ),
        cudaMemcpyHostToDevice ) );
    cudaErrorCheck( cudaMemcpyAsync(
        dClassArr,
        classArr,
        numInstances * sizeof( unsigned short ),
        cudaMemcpyHostToDevice ) );

    /* Determine block and grid size of ComputeCost kernel */
    dim3 ccBlockDim;
    dim3 ccGridDim;
    if (numInstances > 128)
    {
        ccBlockDim.x = 128;
        ccGridDim.x = (numInstances + 127) / 128;
    }
    else ccBlockDim.x = numInstances;

    // Init CuBLAS
    cublasHandle_t cublasHandle;
    cublasErrorCheck( cublasCreate( &cublasHandle ) );

    // Gradient descent params
    float updateWParam = -LEARNING_RATE / (float) numInstances;
    unsigned int iter = 0;

    time_t start, end;
    float dif;
    time( &start );

    printf( "\nStart gradient descent...\n" );

    float default_alpha = 1.0f;
    float default_beta = 0.0f;
    // Gradient descent
    while (iter++ < MAX_ITER)
    {
        // Classify
        cublasErrorCheck( cublasSgemv(
            cublasHandle,
            CUBLAS_OP_N,
            numInstances,
            numFeatures,
            &default_alpha,
            dFeatureMatTrans,
            numInstances,
            dWeightArr,
            1,
            &default_beta,
            dCostArr,
            1 ) );
        ComputeCost<<< ccGridDim, ccBlockDim >>>(
            dCostArr,
            dClassArr,
            numInstances );
        cudaErrorCheck( cudaGetLastError() );
        // Cost vec dot FeaMat-Transpose
        cublasErrorCheck( cublasSgemv(
            cublasHandle,
            CUBLAS_OP_T,
            numInstances,
            numFeatures,
            &default_alpha,
            dFeatureMatTrans,
            numInstances,
            dCostArr,
            1,
            &default_beta,
            dFeaCostProdArr,
            1 ) );
        // Update weights
        cublasErrorCheck( cublasSaxpy(
            cublasHandle,
            numFeatures,
            &updateWParam,
            dFeaCostProdArr,
            1,
            dWeightArr,
            1 ) );
    }
    cudaErrorCheck( cudaThreadSynchronize() );

    cublasErrorCheck( cublasDestroy( cublasHandle ) );
    cudaErrorCheck( cudaMemcpy(
        node.weights,
        dWeightArr,
        numFeatures * sizeof( float ),
        cudaMemcpyDeviceToHost ) );

    time( &end );
    dif = difftime( end, start );
    printf( "Time taken is %.2lf seconds.\n", dif );

    printf( "Updating weights completed, weight: %f\n", node.weights[0] );

    cudaFree( dFeatureMatTrans );
    cudaFree( dClassArr );
    cudaFree( dWeightArr );
    cudaFree( dCostArr );
    cudaFree( dFeaCostProdArr );
    free( node.weights );

    return 0;
}
