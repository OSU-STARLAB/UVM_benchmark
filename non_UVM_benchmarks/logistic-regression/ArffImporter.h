
#ifndef _ARFF_IMPORTER_H_
#define _ARFF_IMPORTER_H_


#include "BasicDataStructures.h"
#include "Helper.h"

#include <stdio.h>
#include <string.h>


using namespace BasicDataStructures;
using namespace MyHelper;

class ArffImporter
{
#define READ_LINE_MAX     5000
#define TOKEN_LENGTH_MAX  35

#define KEYWORD_ATTRIBUTE "@ATTRIBUTE"
#define KEYWORD_DATA      "@DATA"
#define KEYWORD_NUMERIC   "NUMERIC"

public:
    ArffImporter();
    ~ArffImporter();

    void Read( const char* fileName );
    std::vector<char*> GetClassAttr();
    std::vector<NumericAttr> GetFeatures();
    float* GetFeatureMat();
    float* GetFeatureMatTrans();
    unsigned short* GetClassIndex();
    unsigned int GetNumInstances();
    unsigned int GetNumFeatures();


private:
    void BuildFeatureMatrix();
    void Normalize();
    void Transpose();

    std::vector<char*> classVec;
    std::vector<NumericAttr> featureVec;
    std::vector<Instance> instanceVec;

    float* featureMat        = nullptr;
    float* featureMatTrans   = nullptr;
    unsigned short* classArr = nullptr;

    unsigned int numFeatures       = 0;
    unsigned int numInstances      = 0;
    unsigned short numClasses      = 0;
};

#endif
