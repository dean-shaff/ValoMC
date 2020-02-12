#ifndef __MC3D_cuda_hpp
#define __MC3D_cuda_hpp

#include <vector>

namespace ValoMC {

__device__ void normal (int_fast64_t ib, double *normal);

__device__ void normal (int_fast64_t el, long f, double *normal);

__device__ void which_face ();

__device__ void mirror_photon ();

__device__ void fresnel_photon ();

__device__ void scatter_photon ();

__device__ void create_photon ();

__device__ void propagate_photon ();

__global__ void monte_carlo_traverse ();

__global__ void monte_carlo_sum ();

}



#endif
