#include "curand_kernel.h"

#include "MC3D_cuda.hpp"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
   }
}

namespace util {
  __host__ __device__ void ray_triangle_intersects_cross (double dest[3], double v1[3], double v2[3]);

  __host__ __device__ double ray_triangle_intersects_dot (double v1[3], double v2[3]);

  __host__ __device__ void ray_triangle_intersects_sub (double dest[3], double v1[3], double v2[3]);
}


__host__ __device__ double uniform_closed (curandState_t* state);

__host__ __device__ double uniform_open (curandState_t* state);

__host__ __device__ double uniform_half_upper (curandState_t* state);

__host__ __device__ double uniform_half_lower (curandState_t* state);

__host__ __device__ void normal ();

__host__ __device__ void which_face ();

__host__ __device__ void mirror_photon ();

__host__ __device__ void fresnel_photon ();

__host__ __device__ void scatter_photon ();

__host__ __device__ int ray_triangle_intersects (
  double O[3],
  double D[3],
  double V0[3],
  double V1[3],
  double V2[3],
  double *t
);

__host__ __device__ void ray_triangle_intersects_cross (double dest[3], double v1[3], double v2[3]);

__host__ __device__ double ray_triangle_intersects_dot (double v1[3], double v2[3]);

__host__ __device__ void ray_triangle_intersects_sub (double dest[3], double v1[3], double v2[3]);

__host__ __device__ void create_photon ();

__host__ __device__ void propagate_photon ();

__global__ void monte_carlo_traverse ();

__global__ void monte_carlo_sum ();


__host__ __device__ void ray_triangle_intersects_cross (
  double dest[3], double v1[3], double v2[3]
)
{
  dest[0] = v1[1] * v2[2] - v1[2] * v2[1];
  dest[1] = v1[2] * v2[0] - v1[0] * v2[2];
  dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

__host__ __device__ double ray_triangle_intersects_dot (
  double v1[3], double v2[3]
)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

__host__ __device__ void ray_triangle_intersects_sub (
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
#define CROSS(dest, v1, v2)                \
  dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
  dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
  dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
// Dot product
#define DOT(v1, v2) (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
// Vector substraction
#define SUB(dest, v1, v2)  \
  dest[0] = v1[0] - v2[0]; \
  dest[1] = v1[1] - v2[1]; \
  dest[2] = v1[2] - v2[2];

  // The algorithm
  double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3], det, inv_det, u, v;
  SUB(edge1, V1, V0);
  SUB(edge2, V2, V0);
  CROSS(pvec, D, edge2);
  det = DOT(edge1, pvec);
  if ((-eps < det) && (det < eps))
    return (0);
  inv_det = 1.0 / det;
  SUB(tvec, O, V0);
  u = DOT(tvec, pvec) * inv_det;
  if ((u < 0.0) || (u > 1.0))
    return (0);
  CROSS(qvec, tvec, edge1);
  v = DOT(D, qvec) * inv_det;
  if ((v < 0.0) || (u + v > 1.0))
    return (0);
  *t = DOT(edge2, qvec) * inv_det;

  return (1);
}
