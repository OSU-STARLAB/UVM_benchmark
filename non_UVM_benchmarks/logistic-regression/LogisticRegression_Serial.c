#include "Helper.h"
#include "ArffImporter.h"


Node initNode( unsigned int numFeatures )
{
    Node node;
    node.numFeatures = numFeatures;
    node.weights = (float*) malloc( numFeatures * sizeof( float ) );
    memset( node.weights, 0, numFeatures * sizeof( float ) );

    return node;
}

inline float activate(
    Node* node,
    float* inputArr )
{
    float linearRes = 0.0f;
    node->inputs = inputArr;

    unsigned int numFeatures = node->numFeatures;
    for (unsigned int i = 0; i < numFeatures; i++)
        linearRes += node->weights[i] * node->inputs[i];
    node->output = 1.0 / (1.0 + expf(-linearRes));

    return node->output;
}

inline float computeCost( float hRes, unsigned short y )
{
    return (y)? -log(hRes) : -log(1.0 - hRes);
    // return -y * log(hRes) - (1 - y) * (1 - log(hRes));
}

int main()
{
    ArffImporter trainSetImporter;
    trainSetImporter.Read( "Dataset/train/train-first1000.arff" );

    // ArffImporter testSetImporter;
    // testSetImporter.Read( "Dataset/test/dev-first1000.arff" );

    float* featureMat = trainSetImporter.GetFeatureMat();
    unsigned short* classArr = trainSetImporter.GetClassIndex();
    unsigned int numInst = trainSetImporter.GetNumInstances();
    unsigned int numFeatures = trainSetImporter.GetNumFeatures();
    Node node = initNode( numFeatures );

    unsigned int iter = 0;
    unsigned int maxIter = 1000;
    float costSumPre = 0.0;
    float deltaCostSum = 0.0;
    float alpha = 10.0;
    float* batchArr = (float*) malloc( numFeatures * sizeof( float ) );

    time_t start, end;
    double dif;
    time( &start );
    
    printf( "\nStart gradient descent...\n" );

    // Gradient descent
    do
    {
        float costSumNew = 0.0;
        memset( batchArr, 0, numFeatures * sizeof( float ) );

        for (unsigned int i = 0; i < numInst; i++)
        {
            float hRes = activate( &node, &featureMat[i * numFeatures] );
            float diff = hRes - (float) classArr[i];
            costSumNew += computeCost( hRes, classArr[i] );
            for (unsigned int j = 0; j < numFeatures; j++)
                batchArr[j] += diff * node.inputs[j];
        }

        deltaCostSum = costSumPre - costSumNew;
        costSumPre = costSumNew;

        // printf( "Total cost: %f\n", costSumNew );
        // printf( "Delta cost: %f\n", deltaCostSum );
        // printf( "Weight: %f\n", node.weights[0] );

        // Update weights
        for (unsigned int j = 0; j < numFeatures; j++)
            node.weights[j] -= alpha / (float) numInst * batchArr[j];

        iter++;
    }
    while (iter == 1 || (deltaCostSum > 0.05f && iter < maxIter));

    time( &end );
    dif = difftime( end, start );
    printf( "Time taken is %.2lf seconds.\n", dif );

    printf( "Weight: %f\n", node.weights[0] );

    return 0;
}
