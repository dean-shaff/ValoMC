#include <iostream>

#include "catch.hpp"

#include "GPUArray.cuh"

template<typename T>
__global__ void iter_array (T* arr, unsigned size)
{
  const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == 0) {
    for (unsigned istep=0; istep<size; istep++) {
      printf("%f\n", arr[istep]);
    }
  }
}

template<typename T>
__global__ void iter_array (GPUArray<T>* arr) {
  const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  printf("%d\n", arr->N);
  // if (idx == 0) {
  //   for (unsigned istep=0; istep<arr.N; istep++) {
  //     printf("%f\n", (*arr)[istep]);
  //   }
  // }
}

TEMPLATE_TEST_CASE(
  "test GPUArray on device",
  "[GPUArray][cuda]",
  float
)
{
  TestType* ptr;
  unsigned size = 10;

  cudaMalloc((void**)&ptr, size*sizeof(TestType));
  iter_array<TestType><<<1, 1>>>(ptr, size);
  cudaFree(ptr);

  GPUArray<TestType> arr;
  arr.IsGPU = 1;
  arr.resize(100);
}

TEMPLATE_TEST_CASE (
  "ensure constructors and copy operators work",
  "[GPUArray][cuda][unit][copy][constructor]",
  float
)
{
  SECTION ("Empty Constructor") {
    GPUArray<TestType> arr;
    REQUIRE(arr.data == NULL);
    REQUIRE(arr.IsGPU == 0);
    REQUIRE(arr.IsRef == 0);
    REQUIRE(arr.rank == 0);
  }

  SECTION ("Copy Constructor") {
    GPUArray<TestType> arr;
    GPUArray<TestType> arr1(arr);
    REQUIRE(arr1.IsRef == 1);
  }

  SECTION ("Copy Operator") {
    GPUArray<TestType> arr;
    GPUArray<TestType> arr1 = arr;
    REQUIRE(arr1.IsRef == 1);
    arr.IsGPU = 1;
    REQUIRE_THROWS(arr1 = arr);
  }
}

TEMPLATE_TEST_CASE (
  "Ensure resize works",
  "[GPUArray][cuda][unit][resize]",
  float
)
{
  SECTION ("resize works on host array") {
    GPUArray<TestType> arrh;
    arrh.resize(10);
    // arrh[0] = 1;
  }

  SECTION ("resize works on device array") {
    GPUArray<TestType> arrd;
    arrd.IsGPU = 1;
    arrd.resize(10);
    // arrd[0] = 1;
  }
}

TEMPLATE_TEST_CASE (
  "ensure GPUArray h2d works",
  "[GPUArray][cuda][unit][h2d]",
  float
)
{
  GPUArray<TestType> arrh;
  GPUArray<TestType> arrd;
  arrh.resize(10);
  arrh.h2d(arrd);

  arrh.IsGPU = 1;
  REQUIRE_THROWS(arrh.h2d(arrd));
}

TEMPLATE_TEST_CASE (
  "ensure GPUArray d2h works",
  "[GPUArray][cuda][unit][d2h]",
  float
)
{
  GPUArray<TestType> arrd;
  GPUArray<TestType> arrh;
  arrd.IsGPU = 1;
  arrd.resize(10);
  arrd.d2h(arrh);

  arrd.IsGPU = 0;
  REQUIRE_THROWS(arrd.d2h(arrh));
}
