#ifndef __MC3D_util_hpp
#define __MC3D_util_hpp

#include <limits>

namespace ValoMC {
namespace util {

  struct limit_map {

    // 4294967295 as a double
    static constexpr double u32_max = static_cast<double>(
      std::numeric_limits<unsigned>::max());

    // 4294967296 as a double
    static constexpr double u32_max_p1 = static_cast<double>(
      std::numeric_limits<unsigned>::max()) + 1.0;

    // 1.0 / 4294967295.0
    static constexpr double u32_max_inv = 1.0/static_cast<double>(
      std::numeric_limits<unsigned>::max());

    // 1.0 / 4294967296.0
    static constexpr double u32_max_p1_inv = 1.0/(static_cast<double>(
      std::numeric_limits<unsigned>::max()) + 1.0);
  };

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

  // Create a random number on [0,1]-real-interval
  template<typename T, typename State>
  __device__ T rand_closed (State* state) {
    return (T)(((double) curand(state)) * limit_map::u32_max_inv);
  }

  // Create a random number on ]0,1[-real-interval
  template<typename T, typename State>
  __device__ T rand_open (State* state) {
    return (T)(((double) curand(state) + 0.5) * limit_map::u32_max_p1_inv);
  }

  // Create a random number on [0,1[-real-interval
  template<typename T, typename State>
  __device__ T rand_open_up (State* state){
    return (T)(((double) curand(state)) * limit_map::u32_max_p1_inv);
  }

  // Create a random number on ]0,1]-real-interval
  template<typename T, typename State>
  __device__ T rand_open_down (State* state) {
    return (T)(((double) curand(state) + 1.0) * limit_map::u32_max_p1_inv);
  }


}
}


#endif
