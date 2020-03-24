#ifndef __MC3D_cuda_cuh
#define __MC3D_cuda_cuh

#include <inttypes.h> // for int_fast64_t without std:: prefix

#include <curand_kernel.h>

#include "MC3D.hpp"
#include "MC3D_util.cuh"

namespace ValoMC {

template<typename dtype>
class MC3DCUDA {

public:

  MC3DCUDA (MC3D<dtype>& _mc3d, unsigned _states_size, unsigned _gpu_device_num=0) : mc3d(_mc3d), states_size(_states_size), gpu_device_num(_gpu_device_num) {
    is_allocated = false;
    init();
  }

  ~MC3DCUDA () {
    deallocate();
  }

  __host__ __device__ void normal (
    int_fast64_t ib,
    dtype *normal
  );

  __host__ __device__ void normal (
    int_fast64_t el,
    long f,
    dtype *normal
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
    Photon<dtype>* phot,
    dtype* dist
  );

  __device__ void create_photon (Photon<dtype>* phot, curandState_t* state);

  __device__ void scatter_photon (Photon<dtype>* phot, curandState_t* state);

  __host__ __device__ void mirror_photon (Photon<dtype>* phot, int_fast64_t ib);

  __host__ __device__ void mirror_photon (Photon<dtype>* phot, int_fast64_t el, long f);

  __device__ int fresnel_photon (Photon<dtype>* phot, curandState_t* state);

  __device__ void propagate_photon_atomic (Photon<dtype>* phot, curandState_t* state);

  __device__ int propagate_photon_single_step_atomic (Photon<dtype>* phot, curandState_t* state, dtype* prop, dtype* dist, dtype* ds, int_fast64_t* ib);

  const MC3D<dtype>& get_mc3d () const { return mc3d; }
  MC3D<dtype>& get_mc3d () { return mc3d; }
  void set_mc3d (MC3D<dtype>& _mc3d) { mc3d = _mc3d; }

  __host__ __device__ unsigned long get_seed () const { return seed; }
  __host__ __device__ void set_seed (unsigned long _seed) { seed = _seed; mc3d.seed = _seed; };

  __host__ __device__ unsigned get_states_size () const { return states_size; }
  __host__ __device__ void set_states_size (unsigned _states_size) { states_size = _states_size; }

  __host__ __device__ curandState_t* get_states () { return states; }
  __host__ __device__ void set_states (curandState_t* _states) { states = _states; }

  __host__ __device__ unsigned long get_nphotons () { return nphotons; }
  __host__ __device__ void set_nphotons (unsigned long _nphotons) { nphotons = _nphotons; }

  __host__ __device__ dtype get_omega () { return omega; }
  __host__ __device__ void set_omega (dtype _omega) { omega = _omega; }

  /**
   * Allocate Array objects in device memory.
   */
  void allocate ();

  /**
   * Free up any memory allocates
   */
  void deallocate ();

  /**
   * Copy arrays from host to device memory
   */
  void h2d ();

  /**
   * Copy results arrays from device to host memory
   */
  void d2h ();

  // initialize non Array properties from mc3d object.
  void init ();

  /**
   * Do monte carlo simulation
   */
  void monte_carlo ();

  void monte_carlo_atomic ();


  /**
   * Calculate the amount of GPU memory need to hold all of data for Monte Carlo simulation
   * @return unsigned int in bytes
   */
  unsigned long get_total_memory_usage ();

  Array<int_fast64_t>* get_topology () { return topology; }
  const Array<int_fast64_t>* get_topology () const { return topology; }

  Array<int_fast64_t>* get_neighborhood () { return neighborhood; }
  const Array<int_fast64_t>* get_neighborhood () const { return neighborhood; }

  Array<int_fast64_t>* get_boundary () { return boundary; }
  const Array<int_fast64_t>* get_boundary () const { return boundary; }

  Array<dtype>* get_grid_nodes () {return grid_nodes;}
  const Array<dtype>* get_grid_nodes () const {return grid_nodes;}

  Array<int>* get_light_sources () {return light_sources;}
  const Array<int>* get_light_sources () const {return light_sources;}

  Array<int>* get_light_sources_mother () {return light_sources_mother;}
  const Array<int>* get_light_sources_mother () const {return light_sources_mother;}

  Array<dtype>* get_light_sources_cdf () {return light_sources_cdf;}
  const Array<dtype>* get_light_sources_cdf () const {return light_sources_cdf;}

  Array<char>* get_BC_light_direction_type () {return BC_light_direction_type;}
  const Array<char>* get_BC_light_direction_type () const {return BC_light_direction_type;}

  Array<dtype>* get_BCL_normal () {return BCL_normal;}
  const Array<dtype>* get_BCL_normal () const {return BCL_normal;}

  Array<dtype>* get_BC_n () {return BC_n;}
  const Array<dtype>* get_BC_n () const {return BC_n;}

  Array<char>* get_BC_type () {return BC_type;}
  const Array<char>* get_BC_type () const {return BC_type;}

  Array<dtype>* get_absorption () {return absorption;}
  const Array<dtype>* get_absorption () const {return absorption;}

  Array<dtype>* get_scattering () {return scattering;}
  const Array<dtype>* get_scattering () const {return scattering;}

  Array<dtype>* get_scattering_inhom () {return scattering_inhom;}
  const Array<dtype>* get_scattering_inhom () const {return scattering_inhom;}

  Array<dtype>* get_idx_refrc () {return idx_refrc;}
  const Array<dtype>* get_idx_refrc () const {return idx_refrc;}

  Array<dtype>* get_wave_number () {return wave_number;}
  const Array<dtype>* get_wave_number () const {return wave_number;}

  Array<dtype>* get_scattering_inhom_2 () {return scattering_inhom_2;}
  const Array<dtype>* get_scattering_inhom_2 () const {return scattering_inhom_2;}

  Array<dtype>* get_pow_den_vol_real () {return pow_den_vol_real;}
  const Array<dtype>* get_pow_den_vol_real () const {return pow_den_vol_real;}

  Array<dtype>* get_pow_den_vol_imag () {return pow_den_vol_imag;}
  const Array<dtype>* get_pow_den_vol_imag () const {return pow_den_vol_imag;}

  Array<dtype>* get_pow_den_boun_real () {return pow_den_boun_real;}
  const Array<dtype>* get_pow_den_boun_real () const {return pow_den_boun_real;}

  Array<dtype>* get_pow_den_boun_imag () {return pow_den_boun_imag;}
  const Array<dtype>* get_pow_den_boun_imag () const {return pow_den_boun_imag;}

  unsigned get_max_block_size_init_state () const { return max_block_size_init_state; }
  unsigned get_max_block_size_monte_carlo () const { return max_block_size_monte_carlo; }
  unsigned get_gpu_device_num () const { return gpu_device_num; }

private:

  Array<int_fast64_t>* topology; // H
  Array<int_fast64_t>* neighborhood; // HN
  Array<int_fast64_t>* boundary; // BH

  Array<dtype>* grid_nodes; // r

  Array<int>* light_sources; // LightSources
  Array<int>* light_sources_mother; // LightSourcesMother
  Array<dtype>* light_sources_cdf; // LightSourcesCDF

  Array<char>* BC_light_direction_type; // BCLightDirectionType
  Array<dtype>* BCL_normal; // BCLNormal
  Array<dtype>* BC_n; // BCn
  Array<char>* BC_type; // BCType

  Array<dtype>* absorption; // mua
  Array<dtype>* scattering; // mus
  Array<dtype>* scattering_inhom; // g, scattering inhomogeneity
  Array<dtype>* idx_refrc; // n,  index of refraction
  Array<dtype>* wave_number; // k, wave number
  Array<dtype>* scattering_inhom_2; // g2, scattering inhomogeneity squared

  Array<dtype>* pow_den_vol_real; // ER
  Array<dtype>* pow_den_vol_imag; // EI

  Array<dtype>* pow_den_boun_real; // EBR
  Array<dtype>* pow_den_boun_imag; // EBI


  MC3D<dtype> mc3d;

  // curand states for generating random numbers
  curandState_t* states;

  // number of states
  unsigned states_size;

  // random number seed
  unsigned long seed;

  // number of photons
  unsigned long nphotons;

  // omega factor from mc3d object
  dtype omega;

  // weight factor from mc3d object
  dtype weight0;

  // chance factor from mc3d object
  dtype chance;

  // phase0 factor from mc3d object
  dtype phase0;

  // flag indicating whether `allocate` has been called
  bool is_allocated;

  unsigned max_block_size_init_state;
  unsigned max_block_size_monte_carlo;
  unsigned gpu_device_num;

};

template<typename dtype>
__global__ void _init_state (MC3DCUDA<dtype>* mc3d);

template<typename dtype>
__global__ void _monte_carlo_atomic (MC3DCUDA<dtype>* mc3d);

}



#endif
