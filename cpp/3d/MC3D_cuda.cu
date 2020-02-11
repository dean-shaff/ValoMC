#include "curand_kernel.h"

#include "MC3D_cuda.hpp"
#include "MC3D_kernels.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
   }
}


namespace util {
  int invoke_ray_triangle_intersects (
    double O[3],
    double D[3],
    double V0[3],
    double V1[3],
    double V2[3],
    double *t
  )
  {
    return util::ray_triangle_intersects(O, D, V0, V1, V2, t);
  }

  void invoke_cross (double dest[3], double v1[3], double v2[3])
  {
    return util::cross (dest, v1, v2);
  }

  double invoke_dot (double v1[3], double v2[3])
  {
    return util::dot (v1, v2);
  }

  void invoke_sub (double dest[3], double v1[3], double v2[3])
  {
    return util::sub (dest, v1, v2);
  }

  double invoke_uniform_closed (curandState_t* state)
  {
    return util::uniform_closed (state);
  }

  double invoke_uniform_open (curandState_t* state)
  {
    return util::uniform_open (state);
  }

  double invoke_uniform_half_upper (curandState_t* state)
  {
    return util::uniform_half_upper (state);
  }

  double invoke_uniform_half_lower (curandState_t* state)
  {
    return util::uniform_half_lower (state);
  }


}
