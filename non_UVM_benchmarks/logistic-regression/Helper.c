
#include "Helper.h"


// bool MyHelper::Compare(
//     const MiniInstance& eleX, 
//     const MiniInstance& eleY )
// {
//     return eleX.featureValue < eleY.featureValue;
// }

Instance MyHelper::Tokenize(
    const char* str, 
    const std::vector<NumericAttr>& featureVec )
{
    unsigned int numFeatures = featureVec.size();
    Instance instance;
    instance.featureAttrArray = 
        (float*) calloc( numFeatures, sizeof( float ) );

    unsigned int iter = 0;

    while (str[iter] != '\0')
    {
        unsigned int startIndex = iter;

        while (IsLetter( str[iter] ) ||
            str[iter] == '\?' || str[iter] == '_')
            iter++;

        // Found a token
        if (iter > startIndex)
        {
            unsigned int tokenLen = iter - startIndex;

            // Compare the token with every feature name
            // Might use a hashmap (with key: name, value: index) 
            // to speed up
            for (unsigned int feaIndex = 0;
                feaIndex < numFeatures; feaIndex++)
            {
                const char* feaName = featureVec[feaIndex].name;

                unsigned index = 0;
                while (index < tokenLen && feaName[index] != '\0'
                    && (feaName[index] == str[startIndex + index] ||
                    feaName[index] == str[startIndex + index] + 32))
                    index++;
                
                if (index == tokenLen && feaName[index] == '\0')
                    instance.featureAttrArray[feaIndex]++;
            }
        }

        if (str[iter] != '\0') iter++;
    }

    return instance;
}

bool MyHelper::IsLetter( const char c )
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool MyHelper::StrEqualCaseSen( const char* str1, const char* str2 )
{
    unsigned short i = 0;
    while (str1[i] != '\0' && str1[i] == str2[i]) i++;

    return (str1[i] == '\0' && str2[i] == '\0') ? true : false;
}

bool MyHelper::StrEqualCaseInsen( const char* str1, const char* str2 )
{
    unsigned short i = 0;
    while (str1[i] != '\0' &&
        (str1[i] == str2[i] ||
        (IsLetter( str1[i] ) && IsLetter( str2[i] ) &&
        abs( str1[i] - str2[i] ) == 32))) i++;

    return (str1[i] == '\0' && str2[i] == '\0') ? true : false;
}

unsigned int MyHelper::GetStrLength( const char* str )
{
    unsigned int len = 0;

    while (str[len++] != '\0');

    return len;
}

unsigned int MyHelper::getIndexOfMax(
    const unsigned int* uintArray, 
    const unsigned int length )
{
    return std::max_element( uintArray, uintArray + length ) - uintArray;
}

unsigned int MyHelper::removeDuplicates(
    float* sortedArr, 
    unsigned int length )
{
    if (sortedArr == nullptr) return 0;

    unsigned int uniqueId = 1;
    unsigned int iter = 1;

    while (iter < length)
    {
        if (sortedArr[iter - 1] != sortedArr[iter])
            sortedArr[uniqueId++] = sortedArr[iter];

        iter++;
    }

    return uniqueId;
}
