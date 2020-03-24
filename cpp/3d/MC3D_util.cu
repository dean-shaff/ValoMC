#include "MC3D_util.cuh"

namespace ValoMC {
namespace util {

template<typename dtype>
__host__ __device__ void cross (
  dtype dest[3], dtype v1[3], dtype v2[3]
)
{
  dest[0] = v1[1] * v2[2] - v1[2] * v2[1];
  dest[1] = v1[2] * v2[0] - v1[0] * v2[2];
  dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

template __host__ __device__ void cross<float> (
  float dest[3], float v1[3], float v2[3]
);

template __host__ __device__ void cross<double> (
  double dest[3], double v1[3], double v2[3]
);


template<typename dtype>
__host__ __device__ dtype dot (
  dtype v1[3], dtype v2[3]
)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template __host__ __device__ float dot<float> (
  float v1[3], float v2[3]
);

template __host__ __device__ double dot<double> (
  double v1[3], double v2[3]
);


template<typename dtype>
__host__ __device__ void sub (
  dtype dest[3], dtype v1[3], dtype v2[3]
)
{
  dest[0] = v1[0] - v2[0];
  dest[1] = v1[1] - v2[1];
  dest[2] = v1[2] - v2[2];
}

template __host__ __device__ void sub<float> (
  float dest[3], float v1[3], float v2[3]
);

template __host__ __device__ void sub<double> (
  double dest[3], double v1[3], double v2[3]
);


template<typename dtype>
__host__ __device__ int ray_triangle_intersects (
  dtype O[3],
  dtype D[3],
  dtype V0[3],
  dtype V1[3],
  dtype V2[3],
  dtype *t
)
{
  dtype edge1[3], edge2[3], tvec[3], pvec[3], qvec[3], det, inv_det, u, v;
  util::sub(edge1, V1, V0);
  util::sub(edge2, V2, V0);
  util::cross(pvec, D, edge2);
  det = util::dot(edge1, pvec);
  if ((-eps_map<dtype>::eps < det) && (det < eps_map<dtype>::eps)) {
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

template __host__ __device__ int ray_triangle_intersects<float> (
  float O[3],
  float D[3],
  float V0[3],
  float V1[3],
  float V2[3],
  float *t
);

template __host__ __device__ int ray_triangle_intersects<double> (
  double O[3],
  double D[3],
  double V0[3],
  double V1[3],
  double V2[3],
  double *t
);


}
}
