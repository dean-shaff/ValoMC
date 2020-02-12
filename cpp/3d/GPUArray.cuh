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

  int IsGPU;

  GPUArray () : Array<T>() {
    IsGPU = 0;
  }

  GPUArray (GPUArray<T>& ref) : Array<T> (ref) {
    IsGPU = ref.IsGPU;
  }

  GPUArray& operator=(const GPUArray<T>& ref) {
    if (this != &ref) {
      if (IsGPU != ref.IsGPU) {
        if (ref.IsGPU != IsGPU) {
          throw std::runtime_error("Can only use copy operator if IsGPU is equal");
        }
      } else {
        GPUArray<T>(ref);
      }
    }
    return *this;
  }

  T* resize(int_fast64_t _Nx) {
    std::cerr << "GPUArray::resize" << std::endl;
    return Array<T>::resize(_Nx); }

  void h2d (GPUArray<T>& ref) {
    if (IsGPU) {
      throw std::runtime_error("Cannot copy from host to device if IsGPU == 1");
    }
    ref.copy_attributes(*this);
    std::cerr << "GPUArray::h2d: ref.rank=" << ref.rank
      << " ref.N=" << ref.N << std::endl;
    ref.IsGPU = 1;
    ref.allocate(ref.N);
    gpuErrchk(
      cudaMemcpy(ref.data, this->data, sizeof(T)*ref.N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
  }

  void d2h (GPUArray<T>& ref) {
    if (! IsGPU) {
      throw std::runtime_error("Cannot copy from host to device if IsGPU == 0");
    }
    std::cerr << "GPUArray::d2h" << std::endl;
    ref.copy_attributes(*this);
    ref.IsGPU = 0;
    ref.allocate(ref.N);
    std::cerr << "GPUArray::d2h: this->data=" << this->data << std::endl;
    std::cerr << "GPUArray::d2h: ref.data=" << ref.data << std::endl;
    gpuErrchk(
      cudaMemcpy(ref.data, this->data, sizeof(T)*ref.N, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());
  }

  void allocate (int_fast64_t N) {
    std::cerr << "GPUArray::allocate: N=" << N << std::endl;
    if (! IsGPU) {
      std::cerr << "GPUArray::allocate: host alloc" <<  std::endl;
      Array<T>::allocate(N);
    } else {
      std::cerr << "GPUArray::allocate: device alloc" <<  std::endl;
      T* ptr;
      gpuErrchk(
        cudaMalloc((void**) &ptr, N*sizeof(T)));
      gpuErrchk(cudaDeviceSynchronize());
      this->data = ptr;
    }
  }

  void deallocate () {
    if (! IsGPU) {
      Array<T>::deallocate();
    } else {
      gpuErrchk(cudaFree(this->data));
      gpuErrchk(cudaDeviceSynchronize());
    }
  }

private:

  void copy_attributes (const GPUArray<T>& ref) {
    this->rank = ref.rank;
    this->Nx = ref.Nx;
    this->Ny = ref.Ny;
    this->Nz = ref.Nz;
    this->N = ref.N;
  }

};


#endif
