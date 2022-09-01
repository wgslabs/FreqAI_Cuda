// #pragma once

// #include <cstdio>
// #include <cmath>
// #include <stdint.h>

// #if defined(_MSC_VER)
//     //  Microsoft 
//     #define EXPORT __declspec(dllexport)
//     #define IMPORT __declspec(dllimport)
// #elif defined(__GNUC__)
//     //  GCC
//     #define EXPORT __attribute__((visibility("default")))
//     #define IMPORT
// #else
//     //  do nothing and hope for the best?
//     #define EXPORT
//     #define IMPORT
//     #pragma warning Unknown dynamic link import/export semantics.
// #endif

// #define _USE_MATH_DEFINES

// using namespace std;

// typedef unsigned long ULONG;
// typedef unsigned int UINT;

// #define UNIT_COUNT 200
// #define BEFORE_DATA_COUNT 100

// EXPORT int cudaHighPassFilter(double * times,const uint8_t * before_data_1, const uint8_t * before_data_2,const uint8_t* src, const int cnt, float* max_1, float* max_2, double hf_st1, double hf_cf1, double hf_st2, double hf_cf2);

#pragma once

#include <cstdio>
#include <cmath>
#include <stdint.h>

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

#define _USE_MATH_DEFINES

using namespace std;

typedef unsigned long ULONG;
typedef unsigned int UINT;

#define UNIT_COUNT 200
#define BEFORE_DATA_COUNT 100

EXPORT int cudaHighPassFilter(const uint8_t* src, const int cnt,  const uint8_t * before_data_1, const uint8_t * before_data_2,float* max_1, float* max_2,float* min_1, float* min_2,const double hf_st1,const double hf_cf1,const double hf_st2,const double hf_cf2, const int offset_1, const double scale_1, const int offset_2, const double scale_2);