#ifndef __MC3D_util_cuh
#define __MC3D_util_cuh

#include <limits>
#include <cuda.h>

#include "Array.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
   }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val==0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif


namespace ValoMC {
namespace util {

  struct limit_map {

    // 4294967295 as a double
    static constexpr double u32_max = static_cast<double>(
      std::numeric_limits<unsigned>::max());

    // 4294967296 as a double
    static constexpr double u32_max_p1 = static_cast<double>(
      std::numeric_limits<unsigned>::max()) + 1.0;

    // 1.0 / 4294967295.0
    static constexpr double u32_max_inv = 1.0/static_cast<double>(
      std::numeric_limits<unsigned>::max());

    // 1.0 / 4294967296.0
    static constexpr double u32_max_p1_inv = 1.0/(static_cast<double>(
      std::numeric_limits<unsigned>::max()) + 1.0);
  };

  __host__ __device__ int ray_triangle_intersects (
    double O[3],
    double D[3],
    double V0[3],
    double V1[3],
    double V2[3],
    double *t
  );

  __host__ __device__ void cross (double dest[3], double v1[3], double v2[3]);

  __host__ __device__ double dot (double v1[3], double v2[3]);

  __host__ __device__ void sub (double dest[3], double v1[3], double v2[3]);

  // Create a random number on [0,1]-real-interval
  template<typename State, typename T>
  __device__ T rand_closed (State* state) {
    return (T)(((double) curand(state)) * limit_map::u32_max_inv);
  }

  // Create a random number on ]0,1[-real-interval
  template<typename State, typename T>
  __device__ T rand_open (State* state) {
    return (T)(((double) curand(state) + 0.5) * limit_map::u32_max_p1_inv);
  }

  // Create a random number on [0,1[-real-interval
  template<typename State, typename T>
  __device__ T rand_open_up (State* state){
    return (T)(((double) curand(state)) * limit_map::u32_max_p1_inv);
  }

  // Create a random number on ]0,1]-real-interval
  template<typename State, typename T>
  __device__ T rand_open_down (State* state) {
    return (T)(((double) curand(state) + 1.0) * limit_map::u32_max_p1_inv);
  }

  /**
   * Check if ptr has been allocated on device
   * @param  ptr test pointer
   * @return     true if allocated on device, false otherwise
   */
  template<typename T>
  __host__ bool check_device_ptr (T* ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    // std::cerr << "Error code: " <<  cudaGetErrorString(err) << std::endl;
    // std::cerr << "attr.type: " << attr.type << std::endl;
    if (err != cudaSuccess) {
      return false;
    } else {
      if (attr.type == 2) {
        return true;
      } else {
        return false;
      }
    }
  }


  template<typename T>
  __host__ void copy_attributes (Array<T>* dst, Array<T>* src)
  {
    if (! check_device_ptr(dst)) {
      throw std::runtime_error("dst Array must be allocated on device");
    }

    gpuErrchk(
      cudaMemcpy(dst, src, sizeof(Array<T>), cudaMemcpyHostToDevice));
  }

  /**
   * Reserve size bytes for dst
   * @param dst  pointer to array allocated on device
   * @param size number of samples to allocate
   */
  template<typename T>
  __host__ void reserve (Array<T>* dst, unsigned size)
  {
    if (! check_device_ptr(dst)) {
      throw std::runtime_error("dst Array must be allocated on device");
    }
    T* ptr;
    gpuErrchk(
      cudaMalloc((void**) &ptr, sizeof(T)*size));
    gpuErrchk(
      cudaMemcpy(&(dst->data), &ptr, sizeof(T*), cudaMemcpyHostToDevice));
    // gpuErrchk(
    //   cudaFree(ptr));
  }

  /**
   * Copy contents of src to dst that has been allocated on device.
   * @param dst Array that has been allocated on device
   * @param src Array that has been allocated on host
   */
  template<typename T>
  __host__ void h2d (Array<T>* dst, Array<T>* src)
  {
    if (! check_device_ptr(dst)) {
      throw std::runtime_error("dst Array must be allocated on device");
    }
    // if (check_device_ptr(src)) {
    //   throw std::runtime_error("src Array must be allocated on host");
    // }

    // copy properties of src to dst
    gpuErrchk(
      cudaMemcpy(dst, src, sizeof(Array<T>), cudaMemcpyHostToDevice));
    // now copy memory from src to dst
    T* ptr;
    gpuErrchk(
      cudaMalloc((void**) &ptr, sizeof(T)*(src->N)));
    gpuErrchk(
      cudaMemcpy(ptr, src->data, sizeof(T)*(src->N), cudaMemcpyHostToDevice));
    gpuErrchk(
      cudaMemcpy(&(dst->data), &ptr, sizeof(T*), cudaMemcpyHostToDevice));
    // gpuErrchk(
    //   cudaFree(ptr));
  }

  /**
   * copy contents of src to dst. Does no error checking to ensure that
   * the sizes are compatible.
   * @param dst Array that has been allocated on host
   * @param src Array that has been allocated on device (by cudaMalloc)
   */
  template<typename T>
  __host__ void d2h (Array<T>* dst, Array<T>* src)
  {
    if (!check_device_ptr(src)) {
      throw std::runtime_error("src Array must be allocated on device");
    }
    // if (check_device_ptr(dst)) {
    //   throw std::runtime_error("dst Array must be allocated on host");
    // }

    T* ptr;
    gpuErrchk(
      cudaMalloc((void**) &ptr, sizeof(T)*(dst->N)));
    gpuErrchk(
      cudaMemcpy(&ptr, &(src->data), sizeof(T*), cudaMemcpyDeviceToHost));
    gpuErrchk(
      cudaMemcpy(dst->data, ptr, sizeof(T)*(dst->N), cudaMemcpyDeviceToHost));
    // gpuErrchk(
    //   cudaFree(ptr));
  }


}
}


#endif
