
#ifndef _BASIC_DATA_STRUCTURES_H_
#define _BASIC_DATA_STRUCTURES_H_


namespace BasicDataStructures
{
    struct Instance
    {
        float* featureAttrArray;
        unsigned short classIndex;
    };

    struct NumericAttr
    {
        char* name;
        float min;
        float max;
        float mean;
    };

    struct Node
    {
        float* inputs;
        // Weight array has numFeatures + 1 elements.
        // The last element is bias parameter.
        float* weights;
        float output;
        float error;
        unsigned int numFeatures;
    };
}

#endif
