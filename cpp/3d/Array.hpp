#ifndef __ARRAY_HPP__
#define __ARRAY_HPP__

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
  #define CUDA 1
  #define CUDA_HOST_DEVICE __host__ __device__
  #define CUDA_HOST __host__
#else
  #define CUDA 0
  #define CUDA_HOST_DEVICE
  #define CUDA_HOST
#endif

#include <iostream>
#include <inttypes.h>

// Simple 1, 2, 3D array class with fortran style indexing

template <class T> class Array{
public:

  Array(){
    IsRef = 0;
    rank = 0;
    Nx = Ny = Nz = N = 0;
    data = NULL;
  }

  Array(Array<T> &ref){
    IsRef = 1;
    rank = ref.rank;
    Nx = ref.Nx; Ny = ref.Ny; Nz = ref.Nz; N = ref.N;
    data = ref.data;
  }

  Array &operator=(const Array &ref){
    if(this != &ref){
      IsRef = 1;
      rank = ref.rank;
      Nx = ref.Nx; Ny = ref.Ny; Nz = ref.Nz; N = ref.N;
      data = ref.data;
    }
    return *this;
  }

  T* resize(int_fast64_t _Nx, int_fast64_t _Ny, int_fast64_t _Nz){
    rank = 3;
    Nx = _Nx; Ny = _Ny; Nz = _Nz; N = Nx * Ny * Nz; Nxy = Nx * Ny;

    allocate(N);
    return data;
  }

  T* resize(int_fast64_t _Nx, int_fast64_t _Ny){
    rank = 2;
    Nx = _Nx; Ny = _Ny; Nz = 1; N = Nx * Ny * Nz; Nxy = Nx * Ny;

    allocate(N);
    return data;
  }

  T* resize(int_fast64_t _Nx){
    rank = 1;
    Nx = _Nx; Ny = 1; Nz = 1; N = Nx * Ny * Nz; Nxy = Nx * Ny;

    allocate(N);
    return data;
  }

  CUDA_HOST_DEVICE
  T& operator()(int_fast64_t ix, int_fast64_t iy, int_fast64_t iz){
    return( data[ ix + Nx * iy + Nxy * iz] );
  }

  CUDA_HOST_DEVICE
  T& operator()(int_fast64_t ix, int_fast64_t iy){
    return( data[ ix + Nx * iy] );
  }

  CUDA_HOST_DEVICE
  T& operator()(int_fast64_t ix){
    return( data[ ix ] );
  }

  CUDA_HOST_DEVICE
  T& operator[](int_fast64_t index){
    return( data[index] );
  }

  void destroy(){
    // std::cerr << "Array::destroy" << std::endl;
    rank = Nx = Ny = Nz = Nxy = N = 0;
    if (!IsRef && data != NULL) {
      deallocate();
    }
    data = NULL;
  }

  void deallocate () {
    // std::cerr << "Array::deallocate" << std::endl;
    delete[] data;
  }


  void allocate(int_fast64_t N){
    // std::cerr << "Array: allocate: N=" << N << std::endl;
    data = new T[N];
  }

  ~Array() {
    // std::cerr << "Array::~Array" << std::endl;
    destroy();
  }

  int IsRef;
  int_fast64_t rank, Nx, Ny, Nz, Nxy, N;
  T *data;
};




#endif
