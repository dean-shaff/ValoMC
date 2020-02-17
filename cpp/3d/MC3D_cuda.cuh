#ifndef __MC3D_cuda_hpp
#define __MC3D_cuda_hpp

#include <inttypes.h> // for int_fast64_t without std:: prefix

#include "MC3D.hpp"
#include "MC3D_util.cuh"
#include "GPUArray.cuh"

namespace ValoMC {

class MC3DCUDA {

public:

  MC3DCUDA () {}

  ~MC3DCUDA () {}

  __host__ __device__ void normal (
    int_fast64_t ib,
    double *normal
  );

  __host__ __device__ void normal (
    int_fast64_t el,
    long f,
    double *normal
  );

  /**
   * Determine which face a photon hit
   * @param  grid_nodes [description]
   * @param  topology   [description]
   * @param  phot       photon under test
   * @param  dist       distance the photon can travel before hitting the face
   * @return            -1 if no hit
   *                    0, 1, 2, 3 for faces formed by
   *                    (V0, V1, V2),
   *                    (V0, V1, V3),
   *                    (V0, V2, V3),
   *                    (V1, V2, V3) respectively
   */
  __host__ __device__ int which_face (
    Photon* phot,
    double* dist
  );

  __device__ void create_photon ();

  __device__ void scatter_photon (Photon *phot);

  __device__ void mirror_photon (Photon *phot, int_fast64_t ib);

  __device__ void mirror_photon (Photon *phot, int_fast64_t el, long f);

  __device__ int fresnel_photon (Photon *phot);

  __device__ void propagate_photon ();

  __global__ void monte_carlo ();


  GPUArray<int_fast64_t>* get_topology() { return topology; }
  void set_topology(GPUArray<int_fast64_t>* _topology) { topology = _topology; }

  GPUArray<int_fast64_t>* get_neighborhood() { return neighborhood; }
  void set_neighborhood (GPUArray<int_fast64_t>* _neighborhood) { neighborhood = _neighborhood; }

  GPUArray<int_fast64_t>* get_boundary() { return boundary; }
  void set_boundary (GPUArray<int_fast64_t>* _boundary) { boundary = _boundary; }

  GPUArray<double>* get_grid_nodes () { return grid_nodes; }
  void set_grid_nodes (GPUArray<double>* _grid_nodes) { grid_nodes = _grid_nodes; }

  const MC3D& get_mc3d () { return mc3d; }
  MC3D& get_mc3d () { return mc3d; }

  void set_mc3d (MC3D& _mc3d) { mc3d = _mc3d; }

private:

  GPUArray<int_fast64_t>* topology;
  GPUArray<int_fast64_t>* neighborhood;
  GPUArray<int_fast64_t>* boundary;

  GPUArray<double>* grid_nodes;

  MC3D mc3d;

  /**
   * Helper function to allocate GPUArray objects in device memory.
   */
  void allocate_attributes ();

};
}



#endif
