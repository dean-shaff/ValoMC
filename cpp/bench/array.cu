#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <math.h>
#include <cuda_runtime.h>

#include "3d/MC3D_util.cuh"
#include "3d/Array.hpp"

using duration = std::chrono::duration<double, std::ratio<1>>;

std::chrono::time_point<std::chrono::high_resolution_clock> now () {
  return std::chrono::high_resolution_clock::now();
}


template<typename T>
__global__ void log_array_shared (Array<T>* arr)
{
  const int size_a = 4;
  const int thread_idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int total_size_x = gridDim.x*blockDim.x;
  const int size = arr->N;


  if (thread_idx > size) {
    return;
  }

  extern __shared__ T a[];
  T* a_ptr = a + size_a * threadIdx.x;


  for (int idx=thread_idx; idx < size; idx+=total_size_x) {
    for (int idy=0; idy<size_a; idy++) {
      a_ptr[idy] = log((*arr)(idx));
    }

    (*arr)(idx) = a_ptr[0] + a_ptr[1] + a_ptr[2] + a_ptr[3];
  }
}


template<typename T>
__global__ void log_array_local (Array<T>* arr)
{
  const int size_a = 4;
  const int thread_idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int total_size_x = gridDim.x*blockDim.x;
  // const int warp_idx = threadIdx.x / warpSize;
  // const int warp_lane = threadIdx.x % warpSize;
  const int size = arr->N;
  double a[size_a] = {1.0, 1.0, 1.0, 1.0};
  if (thread_idx > size) {
    return;
  }

  for (int idx=thread_idx; idx < size; idx+=total_size_x) {
    // if (warp_idx == 0) {
    //   (*arr)(idx) = log((*arr)(idx)) + 1.0;
    // } else {
    //   (*arr)(idx) = log((*arr)(idx));
    // }
    for (int idy=0;idy<size_a;idy++) {
      a[idy] = log((*arr)(idx));
    }

    (*arr)(idx) = a[0] + a[1] + a[2] + a[3];
  }
}


template<typename T>
__global__ void log_array (Array<T>* arr)
{
  const int thread_idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int total_size_x = gridDim.x*blockDim.x;
  // const int warp_idx = threadIdx.x / warpSize;
  // const int warp_lane = threadIdx.x % warpSize;
  const int size = arr->N;

  if (thread_idx > size) {
    return;
  }

  for (int idx=thread_idx; idx < size; idx+=total_size_x) {
    // if (warp_idx == 0) {
    //   (*arr)(idx) = log((*arr)(idx)) + 1.0;
    // } else {
    //   (*arr)(idx) = log((*arr)(idx));
    // }
    (*arr)(idx) = log((*arr)(idx));
  }
}

template<typename T>
__global__ void log_ptr (T* data, const int size)
{
  const int thread_idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int total_size_x = gridDim.x*blockDim.x;

  if (thread_idx > size) {
    return;
  }

  for (int idx=thread_idx; idx < size; idx+=total_size_x) {
    data[idx] = log(data[idx]);
  }

}


int main (int argc, char *argv[]) {
  unsigned size = 1000000;
  unsigned niter = 50;
  if (argc > 1) {
    size = std::stoi(argv[1]);
  }

  // cudaFuncAttributes log_array_attr;
  // cudaFuncAttributes log_ptr_attr;
  //
  // gpuErrchk(cudaFuncGetAttributes(&log_array_attr, log_array<double>));
  // gpuErrchk(cudaFuncGetAttributes(&log_ptr_attr, log_ptr<double>));
  //
  // std::cerr << "log_array registers: " << log_array_attr.numRegs << std::endl;
  // std::cerr << "log_ptr registers: " << log_ptr_attr.numRegs << std::endl;

  unsigned block_size = 128;
  unsigned grid_size = size / block_size;

  std::vector<double> vec_ref(size, 1.0);
  std::vector<double> vec_array(size, 1.0);
  std::vector<double> vec_data(size, 1.0);

  Array<double> arr_h;
  arr_h.data = vec_array.data();
  arr_h.N = size;
  arr_h.IsRef = true;

  Array<double>* arr_d;
  gpuErrchk(cudaMalloc((void**)&arr_d, sizeof(Array<double>)));
  ValoMC::util::h2d(arr_d, &arr_h);

  double* data;
  gpuErrchk(cudaMalloc((void**)&data, sizeof(double)*size));
  gpuErrchk(cudaMemcpy(data, vec_data.data(), sizeof(double)*size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaDeviceSynchronize());

  auto t0 = now();
  for (unsigned iter=0; iter<niter; iter++) {
    log_array_local<<<grid_size, block_size>>>(arr_d);
    cudaDeviceSynchronize();
  }
  duration delta_array = now() - t0;
  std::cerr << "Array Kernel took " << delta_array.count() << " s or "
    << delta_array.count() / niter << " s per iteration" << std::endl;

  size_t shared_size = block_size * 4 * sizeof(double);
  std::cerr << "shared_size=" << shared_size << std::endl;

  t0 = now();
  for (unsigned iter=0; iter<niter; iter++) {
    // log_ptr<<<grid_size, block_size>>>(data, size);
    log_array_shared<<<grid_size, block_size, shared_size>>>(arr_d);
    cudaDeviceSynchronize();
  }
  duration delta_data = now() - t0;

  gpuErrchk(cudaFree(arr_d));
  gpuErrchk(cudaFree(data));

  std::cerr << "Raw Pointer Kernel took " << delta_data.count() << " s or "
    << delta_data.count() / niter << " s per iteration" << std::endl;


  t0 = now();
  for (unsigned iter=0; iter<niter; iter++) {
    for (unsigned idx=0; idx<size; idx++) {
      vec_ref[idx] = log(vec_ref[idx]);
    }
  }
  duration delta_ref = now() - t0;

  std::cerr << "CPU version took " << delta_ref.count() << " s or "
    << delta_ref.count() / niter << " s per iteration" << std::endl;

  std::cerr << "Raw Pointer Kernel " << delta_array.count() /delta_data.count()
    << " times faster that Array Kernel" << std::endl;
  std::cerr << "Raw Pointer Kernel " << delta_ref.count() / delta_data.count()
    << " times faster than CPU version" << std::endl;
}
