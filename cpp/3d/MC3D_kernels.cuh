#ifndef __MC3D_kernels_cuh
#define __MC3D_kernels_cuh

#include <stdio.h>

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

namespace util {

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

  __device__ double uniform_closed (curandState_t* state);

  // __device__ double uniform_open (curandState_t* state);
  //
  // __device__ double uniform_half_upper (curandState_t* state);
  //
  // __device__ double uniform_half_lower (curandState_t* state);

  template<typename ReturnType>
  __global__ void random (curandState_t state, ReturnType* res, unsigned nsamples) {
    printf("random: start\n");
    const unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx == 0) {
      curandState_t local_state = state;
      double val;
      for (unsigned isample=0; isample<nsamples; isample++) {
        // res[isample] = util::uniform_closed(&local_state);
        val = curand_uniform_double(&local_state);
        // printf("isample=%d, val=%f\n", isample, val);
        res[isample] = val;
      }
      state = local_state;
    }
    printf("random: end\n");
  }

  __global__ void init_random (
    unsigned long long seed, unsigned long long sequence,
    unsigned long long offset, curandState_t state);

}

__host__ __device__ void normal ();

__host__ __device__ void which_face ();

__host__ __device__ void mirror_photon ();

__host__ __device__ void fresnel_photon ();

__host__ __device__ void scatter_photon ();

__host__ __device__ void create_photon ();

__host__ __device__ void propagate_photon ();

__global__ void monte_carlo_traverse ();

__global__ void monte_carlo_sum ();

#endif
