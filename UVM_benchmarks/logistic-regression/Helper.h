
#ifndef _HELPER_H_
#define _HELPER_H_

#include "BasicDataStructures.h"
#include <math.h>
#include <algorithm>
#include <vector>
#include <time.h>


using namespace BasicDataStructures;

namespace MyHelper
{
    // bool Compare(
    //     const MiniInstance& eleX,
    //     const MiniInstance& eleY );
    bool StrEqualCaseSen( const char* str1, const char* str2 );
    bool StrEqualCaseInsen( const char* str1, const char* str2 );
    // Include string terminator
    unsigned int GetStrLength( const char* str );
    bool IsLetter( const char c );
    Instance Tokenize(
        const char* str, 
        const std::vector<NumericAttr>& featureVec );

    unsigned int getIndexOfMax(
        const unsigned int* uintArray, 
        const unsigned int length );
    // Consume a sorted array, remove duplicates in place, 
    // and return the number of unique elements.
    unsigned int removeDuplicates(
        float* sortedArr,
        unsigned int length );
}

#endif
