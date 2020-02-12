#ifndef __MC3D_util_hpp
#define __MC3D_util_hpp

namespace ValoMC {
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

}
}


#endif
