#include <iostream>

#include "catch.hpp"

#include "GPUArray.cuh"



template<typename T>
__global__ void iter_array (GPUArray<T>* arr) {
  const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  printf("%d\n", arr->N);
  if (idx == 0) {
    for (unsigned istep=0; istep<arr->N; istep++) {
      printf("%f\n", (*arr)[istep]);
    }
  }
}

// TEMPLATE_TEST_CASE(
//   "test GPUArray on device",
//   "[GPUArray][cuda]",
//   float
// )
// {
//   unsigned size = 10;
//   std::vector<TestType> vec(size);
//   for (unsigned idx=0; idx<vec.size(); idx++) {
//     vec[idx] = static_cast<TestType>(idx);
//   }
//
//   GPUArray<TestType> arr_h;
//   arr_h.N = size;
//   GPUArray<TestType>* arr;
//
//   gpuErrchk(
//     cudaMalloc((void**)&arr, sizeof(GPUArray<TestType>)));
//   gpuErrchk(
//     cudaMemcpy(arr, &arr_h, sizeof(GPUArray<TestType>), cudaMemcpyHostToDevice));
//
//   TestType* ptr;
//   gpuErrchk(
//     cudaMalloc((void**) &ptr, sizeof(TestType)*size));
//   gpuErrchk(
//     cudaMemcpy(ptr, vec.data(), sizeof(TestType)*size, cudaMemcpyHostToDevice));
//   gpuErrchk(
//     cudaMemcpy(&(arr->data), &ptr, sizeof(TestType*), cudaMemcpyHostToDevice));
//   gpuErrchk(cudaDeviceSynchronize());
//   // gpuErrchk(
//   //   cudaMalloc((void**)&(arr->data), sizeof(TestType)*size));
//   // gpuErrchk(
//   //   cudaMemcpy(arr->data, vec.data(), sizeof(TestType)*size, cudaMemcpyHostToDevice));
//
//   iter_array<TestType><<<1, 1>>>(arr);
//   gpuErrchk(cudaGetLastError());
//   // gpuErrchk(cudaFree(arr->data));
//   gpuErrchk(cudaFree(arr));
//   // gpuErrchk(cudaFree(ptr));
//   gpuErrchk(cudaDeviceSynchronize());
// }

TEMPLATE_TEST_CASE(
  "h2d works as expected",
  "[GPUArray][unit]",
  float
)
{
  GPUArray<TestType> arr_h;
  arr_h.resize(10, 2);
  for (unsigned idx=0; idx<arr_h.Nx; idx++) {
    for (unsigned idy=0; idy<arr_h.Ny; idy++) {
      arr_h(idx, idy) = static_cast<TestType>(idx + arr_h.Nx*idy);
    }
  }

  GPUArray<TestType> arr*;
  cudaMalloc((void**)&arr, sizeof(GPUArray<TestType>));

  SECTION ("h2d works in isolation") {
    arr_h.h2d(arr);
  }

  SECTION ("can pass GPUArray pointer to CUDA kernel") {
    arr_h.h2d(arr);
    iter_array<TestType><<<1, 1>>>(arr);
    gpuErrchk(cudaGetLastError());
  }
  gpuErrchk(cudaFree(arr));
  gpuErrchk(cudaDeviceSynchronize());
}

//
// TEMPLATE_TEST_CASE (
//   "ensure constructors and copy operators work",
//   "[GPUArray][cuda][unit][copy][constructor]",
//   float
// )
// {
//   SECTION ("Empty Constructor") {
//     GPUArray<TestType> arr;
//     REQUIRE(arr.data == NULL);
//     REQUIRE(arr.IsGPU == 0);
//     REQUIRE(arr.IsRef == 0);
//     REQUIRE(arr.rank == 0);
//   }
//
//   SECTION ("Copy Constructor") {
//     GPUArray<TestType> arr;
//     GPUArray<TestType> arr1(arr);
//     REQUIRE(arr1.IsRef == 1);
//   }
//
//   SECTION ("Copy Operator") {
//     GPUArray<TestType> arr;
//     GPUArray<TestType> arr1 = arr;
//     REQUIRE(arr1.IsRef == 1);
//     arr.IsGPU = 1;
//     REQUIRE_THROWS(arr1 = arr);
//   }
// }
//
// TEMPLATE_TEST_CASE (
//   "Ensure resize works",
//   "[GPUArray][cuda][unit][resize]",
//   float
// )
// {
//   SECTION ("resize works on host array") {
//     GPUArray<TestType> arrh;
//     arrh.resize(10);
//     // arrh[0] = 1;
//   }
//
//   SECTION ("resize works on device array") {
//     GPUArray<TestType> arrd;
//     arrd.IsGPU = 1;
//     arrd.resize(10);
//     // arrd[0] = 1;
//   }
// }
//
// TEMPLATE_TEST_CASE (
//   "ensure GPUArray h2d works",
//   "[GPUArray][cuda][unit][h2d]",
//   float
// )
// {
//   GPUArray<TestType> arrh;
//   GPUArray<TestType> arrd;
//   arrh.resize(10);
//   arrh.h2d(arrd);
//
//   arrh.IsGPU = 1;
//   REQUIRE_THROWS(arrh.h2d(arrd));
// }
//
// TEMPLATE_TEST_CASE (
//   "ensure GPUArray d2h works",
//   "[GPUArray][cuda][unit][d2h]",
//   float
// )
// {
//   GPUArray<TestType> arrd;
//   GPUArray<TestType> arrh;
//   arrd.IsGPU = 1;
//   arrd.resize(10);
//   arrd.d2h(arrh);
//
//   arrd.IsGPU = 0;
//   REQUIRE_THROWS(arrd.d2h(arrh));
// }
