#ifndef __MC3D_cuda_hpp
#define __MC3D_cuda_hpp

#include <inttypes.h> // for int_fast64_t without std:: prefix

#include <curand_kernel.h>

#include "MC3D.hpp"
#include "MC3D_util.cuh"

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

  __device__ void create_photon (curandState_t* state);

  __device__ void scatter_photon (Photon *phot, curandState_t* state);

  __host__ __device__ void mirror_photon (Photon *phot, int_fast64_t ib);

  __host__ __device__ void mirror_photon (Photon *phot, int_fast64_t el, long f);

  __device__ int fresnel_photon (Photon *phot, curandState_t* state);

  __device__ void propagate_photon (Photon *phot, curandState_t* state);

  // Array<int_fast64_t>* get_topology() { return topology; }
  // void set_topology(Array<int_fast64_t>* _topology) { topology = _topology; }
  //
  // Array<int_fast64_t>* get_neighborhood() { return neighborhood; }
  // void set_neighborhood (Array<int_fast64_t>* _neighborhood) { neighborhood = _neighborhood; }
  //
  // Array<int_fast64_t>* get_boundary() { return boundary; }
  // void set_boundary (Array<int_fast64_t>* _boundary) { boundary = _boundary; }
  //
  // Array<double>* get_grid_nodes () { return grid_nodes; }
  // void set_grid_nodes (Array<double>* _grid_nodes) { grid_nodes = _grid_nodes; }

  const MC3D& get_mc3d () const { return mc3d; }
  MC3D& get_mc3d () { return mc3d; }

  void set_mc3d (MC3D& _mc3d) { mc3d = _mc3d; }

  void set_seed (unsigned long _seed) { seed = _seed; mc3d.seed = _seed; };

  unsigned long get_seed () const { return seed; }

  void set_states_size (unsigned _states_size) { states_size = _states_size; }

  unsigned get_states_size () const { return states_size; }

  /**
   * Helper function to allocate Array objects in device memory.
   */
  void allocate_attributes ();

  /**
   * Calculate the amount of GPU memory need to hold all of data for Monte Carlo simulation
   * @return unsigned int in bytes
   */
  unsigned long get_total_memory_usage ();


private:

  Array<int_fast64_t>* topology; // H
  Array<int_fast64_t>* neighborhood; // HN
  Array<int_fast64_t>* boundary; // BH

  Array<double>* grid_nodes; // r

  Array<int>* light_sources; // LightSources
  Array<int>* light_sources_mother; // LightSourcesMother
  Array<double>* light_sources_cdf; // LightSourcesCDF

  Array<char>* BC_light_direction_type; // BCLightDirectionType
  Array<double>* BCL_normal; // BCLNormal
  Array<double>* BCn; // BCn
  Array<char>* BC_type; // BCType

  Array<double>* absorption; // mua
  Array<double>* scattering; // mus
  Array<double>* scattering_inhom; // g, scattering inhomogeneity
  Array<double>* idx_refrc; // n,  index of refraction
  Array<double>* wave_number; // k, wave number
  Array<double>* scattering_inhom_2; // g2, scattering inhomogeneity squared

  Array<double>* pow_den_vol_real; // ER
  Array<double>* pow_den_vol_imag; // EI

  Array<double>* pow_den_boun_real; // EBR
  Array<double>* pow_den_boun_imag; // EBI

  double omega;

  MC3D mc3d;

  // curand states for generating random numbers
  curandState_t* states;
  // number of states
  unsigned states_size;

  // random number seed
  unsigned long seed;

};

__global__ void init_state (MC3DCUDA mc3d);
__global__ void monte_carlo (MC3DCUDA mc3d);

}



#endif
