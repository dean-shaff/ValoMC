#include <stdio.h> // for fprintf, stderr
#include <limits>

#include "curand_kernel.h"

#include "MC3D_kernels.cuh"

const double eps = std::numeric_limits<double>::epsilon();

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
   }
}

namespace util {

  __host__ __device__ void cross (
    double dest[3], double v1[3], double v2[3]
  )
  {
    dest[0] = v1[1] * v2[2] - v1[2] * v2[1];
    dest[1] = v1[2] * v2[0] - v1[0] * v2[2];
    dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
  }

  __host__ __device__ double dot (
    double v1[3], double v2[3]
  )
  {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  }

  __host__ __device__ void sub (
    double dest[3], double v1[3], double v2[3]
  )
  {
    dest[0] = v1[0] - v2[0];
    dest[1] = v1[1] - v2[1];
    dest[2] = v1[2] - v2[2];
  }

  __host__ __device__ int ray_triangle_intersects (
    double O[3],
    double D[3],
    double V0[3],
    double V1[3],
    double V2[3],
    double *t
  )
  {
    double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3], det, inv_det, u, v;
    util::sub(edge1, V1, V0);
    util::sub(edge2, V2, V0);
    util::cross(pvec, D, edge2);
    det = util::dot(edge1, pvec);
    if ((-eps < det) && (det < eps)) {
      // printf("here det\n");
      return 0;
    }
    inv_det = 1.0 / det;
    util::sub(tvec, O, V0);
    u = util::dot(tvec, pvec) * inv_det;
    if ((u < 0.0) || (u > 1.0)) {
      // printf("here u\n");
      return 0;
    }
    util::cross(qvec, tvec, edge1);
    v = util::dot(D, qvec) * inv_det;
    if ((v < 0.0) || (u + v > 1.0)) {
      // printf("here u + v\n");
      return 0;
    }
    *t = util::dot(edge2, qvec) * inv_det;
    return 1;
  }

  __device__ double uniform_closed (curandState_t* state)
  {
    return curand_uniform_double(state);
  }
  //
  // __device__ double uniform_open (curandState_t* state)
  // {
  //   return curand_uniform_double(state);
  // }
  //
  // __device__ double uniform_half_upper (curandState_t* state)
  // {
  //   return curand_uniform_double(state);
  // }
  //
  // __device__ double uniform_half_lower (curandState_t* state)
  // {
  //   return curand_uniform_double(state);
  // }
  //
  __global__ void init_random (
      unsigned long long seed, unsigned long long sequence,
      unsigned long long offset, curandState_t state
  )
  {
    printf("init_random: start\n");
    const unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx == 0) {
      curandState_t local_state = state;
      curand_init(seed, sequence, offset, &local_state);
      state = local_state;
    }
    printf("init_random: end\n");
  }

}
