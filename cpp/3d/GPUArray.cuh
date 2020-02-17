#ifndef __GPUArray_cuh
#define __GPUArray_cuh

#include <iostream>
#include <exception>

#include "cuda_runtime.h"

#include "Array.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
   }
}

template<typename T>
class GPUArray : public Array<T> {

public:

  /**
   * Copy contents of this to arr that has been allocated on device.
   * @param arr GPUArray that has been allocated via cudaMalloc
   */
  __host__ void h2d (GPUArray<T>* arr)
  {

    // copy properties of this to arr
    gpuErrchk(
      cudaMemcpy(arr, this, sizeof(GPUArray<T>), cudaMemcpyHostToDevice));

    // now copy memory from this to arr
    T* ptr;
    gpuErrchk(
      cudaMalloc((void**) &ptr, sizeof(T)*(this->N)));
    gpuErrchk(
      cudaMemcpy(ptr, this->data, sizeof(T)*(this->N), cudaMemcpyHostToDevice));
    gpuErrchk(
      cudaMemcpy(&(arr->data), &ptr, sizeof(T*), cudaMemcpyHostToDevice));
  }

  /**
   * copy contents of arr to this. Does no error checking to ensure that
   * the sizes are compatible.
   * @param arr GPUArray that has been allocated via cudaMalloc
   */
  __host__ void d2h (GPUArray<T>* arr)
  {
    T* ptr;
    gpuErrchk(
      cudaMalloc((void**) &ptr, sizeof(T)*(this->N)));
    gpuErrchk(
      cudaMemcpy(&ptr, &(arr->data), sizeof(T*), cudaMemcpyDeviceToHost));
    gpuErrchk(
      cudaMemcpy(this->data, ptr, sizeof(T)*(this->N), cudaMemcpyDeviceToHost));
  }

};


// template<typename T>
// class GPUArray : public Array<T> {
// public:
//
//   int IsGPU;
//
//   GPUArray () : Array<T>() {
//     IsGPU = 0;
//   }
//
//   GPUArray (GPUArray<T>& ref) : Array<T> (ref) {
//     IsGPU = ref.IsGPU;
//   }
//
//   ~GPUArray () {
//     std::cerr << "GPUArray::~GPUArray" << std::endl;
//     destroy();
//   }
//
//   void destroy () {
//     if (!this->IsRef && this->data != NULL) {
//       deallocate();
//     }
//     this->data = NULL;
//   }
//
//   GPUArray& operator=(const GPUArray<T>& ref) {
//     if (this != &ref) {
//       if (IsGPU != ref.IsGPU) {
//         if (ref.IsGPU != IsGPU) {
//           throw std::runtime_error("Can only use copy operator if IsGPU is equal");
//         }
//       } else {
//         GPUArray<T>(ref);
//       }
//     }
//     return *this;
//   }
//   // T* resize(int_fast64_t _Nx) {
//   //   std::cerr << "GPUArray::resize" << std::endl;
//   //   return Array<T>::resize(_Nx); }
//
//   __host__ void h2d (GPUArray<T>& ref) {
//     if (IsGPU) {
//       throw std::runtime_error("Cannot copy from host to device if IsGPU == 1");
//     }
//     std::cerr << "GPUArray::h2d" <<  std::endl;
//     ref.copy_attributes(*this);
//     ref.IsGPU = 1;
//     ref.allocate(ref.N);
//     gpuErrchk(
//       cudaMemcpy(ref.data, this->data, sizeof(T)*ref.N, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaDeviceSynchronize());
//   }
//
//   __host__ void d2h (GPUArray<T>& ref) {
//     if (! IsGPU) {
//       throw std::runtime_error("Cannot copy from host to device if IsGPU == 0");
//     }
//     std::cerr << "GPUArray::d2h" << std::endl;
//     ref.copy_attributes(*this);
//     ref.IsGPU = 0;
//     ref.allocate(ref.N);
//     gpuErrchk(
//       cudaMemcpy(ref.data, this->data, sizeof(T)*ref.N, cudaMemcpyDeviceToHost));
//     gpuErrchk(cudaDeviceSynchronize());
//   }
//
//   __host__ void allocate (int_fast64_t N) {
//     std::cerr << "GPUArray::allocate: N=" << N << std::endl;
//     if (! IsGPU) {
//       std::cerr << "GPUArray::allocate: host alloc" <<  std::endl;
//       Array<T>::allocate(N);
//     } else {
//       std::cerr << "GPUArray::allocate: device alloc" <<  std::endl;
//       std::cerr << "GPUArray::allocate: N*sizeof(T)=" << N*sizeof(T) << std::endl;
//       T* ptr;
//       gpuErrchk(
//         cudaMalloc((void**) &ptr, N*sizeof(T)));
//       gpuErrchk(cudaDeviceSynchronize());
//       this->data = ptr;
//       std::cerr << "GPUArray::allocate: this->data=" << this->data << std::endl;
//     }
//   }
//
//   __host__ void deallocate () {
//     std::cerr << "GPUArray:deallocate" << std::endl;
//     if (! IsGPU) {
//       std::cerr << "GPUArray:deallocate: host free" << std::endl;
//       Array<T>::deallocate();
//     } else {
//       std::cerr << "GPUArray:deallocate: device free" << std::endl;
//       gpuErrchk(cudaFree(this->data));
//       gpuErrchk(cudaDeviceSynchronize());
//     }
//   }
//
//   __host__ __device__ void set_N (int_fast64_t _N) { this->N = _N; }
//
//   __host__ __device__ int_fast64_t get_N () { return this->N; }
//
// private:
//
//   void copy_attributes (const GPUArray<T>& ref) {
//     this->rank = ref.rank;
//     this->Nx = ref.Nx;
//     this->Ny = ref.Ny;
//     this->Nz = ref.Nz;
//     this->N = ref.N;
//   }
//
// };


#endif
