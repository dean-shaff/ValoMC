#include "curand_kernel.h"

#include "MC3D_cuda.hpp"

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

  __host__ __device__ double uniform_closed (curandState_t* state);

  __host__ __device__ double uniform_open (curandState_t* state);

  __host__ __device__ double uniform_half_upper (curandState_t* state);

  __host__ __device__ double uniform_half_lower (curandState_t* state);
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
