#include <stdio.h> // for fprintf, stderr

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "MC3D_cuda.cuh"
#include "MC3D_util.cuh"

namespace ValoMC {

template<typename dtype>
__global__ void _init_state (MC3DCUDA<dtype>* mc3d) {
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int total_size_x = gridDim.x*blockDim.x;
  const int states_size = mc3d->get_states_size();
  if (idx > states_size) {
    return;
  }
  const unsigned seed = mc3d->get_seed();
  curandState_t* states = mc3d->get_states();
  // if (total_size_x == 1) {
  //   printf("init_state: seed=%u\n", seed);
  // }

  for (int istate=idx; istate<states_size; istate+=total_size_x) {
    curand_init(seed, idx, 0, &states[istate]);
  }
}

template __global__ void _init_state<float> (MC3DCUDA<float>* mc3d);
template __global__ void _init_state<double> (MC3DCUDA<double>* mc3d);

template<typename dtype>
__global__ void _monte_carlo_atomic (MC3DCUDA<dtype>* mc3d) {
  // if (threadIdx.x % warpSize != 0) {
  //   return;
  // }
  const int warp_lane = threadIdx.x % warpSize;
  const int warp_idx = threadIdx.x/warpSize;
  const int idx = warp_idx + blockDim.x*blockIdx.x/warpSize;
  const int total_size_x = gridDim.x*blockDim.x/warpSize;

  const int states_size = mc3d->get_states_size();
  const int nphotons = mc3d->get_nphotons();

  const int increment_size = states_size > total_size_x ? states_size: total_size_x;

  // printf("_monte_carlo_atomic: increment_size=%u, idx=%u, warpSize=%u\n", increment_size, idx, warpSize);
  // if (idx > increment_size) {
  //   return;
  // }
  curandState_t* states = mc3d->get_states();
  curandState_t local_state = states[idx];
  Photon<dtype> photon;

  for (int iphoton=idx; iphoton<nphotons; iphoton+=increment_size) {
    // if (idx == 0) {
    //   printf("_monte_carlo_atomic: increment_size=%u, iphoton=%u\n", increment_size, iphoton);
    // }
    if (warp_lane == 0) {
      mc3d->create_photon(&photon, &local_state);
      mc3d->propagate_photon_atomic(&photon, &local_state);
    }
    __syncthreads();
  }
}

template __global__ void _monte_carlo_atomic<float> (MC3DCUDA<float>* mc3d);
template __global__ void _monte_carlo_atomic<double> (MC3DCUDA<double>* mc3d);

template<typename dtype>
__global__ void _monte_carlo_atomic_create_photons (
  MC3DCUDA<dtype>* mc3d,
  Photon<dtype>* photons,
  PhotonAttr<dtype>* photon_attrs,
  int* dead
)
{
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int total_size_x = gridDim.x*blockDim.x;

  const int states_size = mc3d->get_states_size();
  const int nphotons = mc3d->get_nphotons();

  const int increment_size = states_size > total_size_x ? states_size: total_size_x;

  curandState_t* states = mc3d->get_states();
  curandState_t local_state = states[idx];

  for (int iphoton=idx; iphoton<nphotons; iphoton+=increment_size) {
    mc3d->create_photon(&photons[iphoton], &local_state);
    photon_attrs[iphoton].prop = -log(ValoMC::util::rand_open<curandState_t, dtype>(&local_state)) / (*(mc3d->get_scattering()))[photons[iphoton].curel];
    dead[iphoton] = 0;
  }
}

template __global__ void _monte_carlo_atomic_create_photons (MC3DCUDA<float>* mc3d, Photon<float>* photons, PhotonAttr<float>* photon_attrs, int* dead);
template __global__ void _monte_carlo_atomic_create_photons (MC3DCUDA<double>* mc3d, Photon<double>* photons, PhotonAttr<double>* photon_attrs, int* dead);

template<typename dtype>
__global__ void _monte_carlo_atomic_single_step (
  MC3DCUDA<dtype>* mc3d,
  Photon<dtype>* photons,
  PhotonAttr<dtype>* photon_attrs,
  int* dead
)
{
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int total_size_x = gridDim.x*blockDim.x;

  const int states_size = mc3d->get_states_size();
  const int nphotons = mc3d->get_nphotons();

  const int increment_size = states_size > total_size_x ? states_size: total_size_x;

  curandState_t* states = mc3d->get_states();
  curandState_t local_state = states[idx];


  Photon<dtype>* phot;
  PhotonAttr<dtype>* phot_attr;
  for (int iphoton=idx; iphoton<nphotons; iphoton+=increment_size) {
    if (dead[iphoton]) {
      continue;
    }
    phot = &photons[iphoton];
    phot_attr = &photon_attrs[iphoton];
    int single_step_res = mc3d->propagate_photon_single_step_atomic(
      phot, &local_state,
      &phot_attr->prop,
      &phot_attr->dist,
      &phot_attr->ds,
      &phot_attr->ib
    );
    if (single_step_res == 0) {
      dead[iphoton] = 1;
    } else if (single_step_res == 1) {
      if ((*(mc3d->get_scattering()))[phot->curel] > 0.0) {
        mc3d->scatter_photon(phot, &local_state);
      }
      phot_attr->prop = -log(ValoMC::util::rand_open<curandState_t, dtype>(&local_state)) / (*(mc3d->get_scattering()))[phot->curel];
    }
  }
}

template __global__ void _monte_carlo_atomic_single_step (MC3DCUDA<float>* mc3d, Photon<float>* photons, PhotonAttr<float>* photon_attrs, int* dead);
template __global__ void _monte_carlo_atomic_single_step (MC3DCUDA<double>* mc3d, Photon<double>* photons, PhotonAttr<double>* photon_attrs, int* dead);


template<typename dtype>
void MC3DCUDA<dtype>::init () {
  omega = mc3d.omega;
  seed = mc3d.seed;
  weight0 = mc3d.weight0;
  chance = mc3d.chance;
  phase0 = mc3d.phase0;
  nphotons = static_cast<unsigned long>(mc3d.Nphoton);

  int n_devices = 0;
  cudaGetDeviceCount(&n_devices);

  if (n_devices == 0) {
    throw std::runtime_error("Need a CUDA enabled GPU to use ValoMC::MC3DCUDA");
  }

  cudaDeviceProp prop;
  gpuErrchk(cudaGetDeviceProperties(&prop, gpu_device_num));

  cudaFuncAttributes monte_carlo_atomic_attr;
  cudaFuncAttributes init_state_attr;

  gpuErrchk(cudaFuncGetAttributes(&monte_carlo_atomic_attr, _monte_carlo_atomic<dtype>));
  gpuErrchk(cudaFuncGetAttributes(&init_state_attr, _init_state<dtype>));

  max_block_size_init_state = 512;
  max_block_size_monte_carlo = 256;

  // const unsigned max_registers = prop.regsPerBlock;
  // max_block_size_init_state = (max_registers / init_state_attr.numRegs) - 1;
  // if (max_block_size_init_state > prop.maxThreadsPerBlock) {
  //   max_block_size_init_state = prop.maxThreadsPerBlock;
  // }
  // max_block_size_monte_carlo = (max_registers / monte_carlo_atomic_attr.numRegs) - 1;
  // if (max_block_size_monte_carlo > prop.maxThreadsPerBlock) {
  //   max_block_size_monte_carlo = prop.maxThreadsPerBlock;
  // }
}


  // Array<int_fast64_t> H = *topology;
  // Array<int_fast64_t> HN = *neighborhood;
  // Array<int_fast64_t> BH = *boundary;
  //
  // Array<dtype> r = *grid_nodes;
  // Array<int> LightSources = *light_sources;
  // Array<int> LightSourcesMother = *light_sources_mother;
  // Array<dtype> LightSourcesCDF = *light_sources_cdf;
  //
  // Array<char> BCLightDirectionType = *BC_light_direction_type;
  // Array<char> BCType = *BC_type;
  // Array<dtype> BCLNormal = *BCL_normal;
  // Array<dtype> BCn = *BC_n;

  // Array<dtype> mua = *absorption;
  // Array<dtype> mus = *scattering;
  // Array<dtype> g = *scattering_inhom;
  // Array<dtype> n = *idx_refrc;
  // Array<dtype> k = *wave_number;
  // Array<dtype> g2 = *scattering_inhom_2;
  //
  // Array<dtype> ER = *pow_den_vol_real;
  // Array<dtype> EI = *pow_den_vol_imag;
  //
  // Array<dtype> EBR = *pow_den_boun_real;
  // Array<dtype> EBI = *pow_den_boun_imag;


template<typename dtype>
__host__ __device__ void MC3DCUDA<dtype>::normal (
  int_fast64_t ib, dtype *normal
)
{
  dtype x1, y1, z1, x2, y2, z2, nx, ny, nz, norm;

  // Two edges of the face
  x1 = (*grid_nodes)((*boundary)(ib, 1), 0) - (*grid_nodes)((*boundary)(ib, 0), 0);
  y1 = (*grid_nodes)((*boundary)(ib, 1), 1) - (*grid_nodes)((*boundary)(ib, 0), 1);
  z1 = (*grid_nodes)((*boundary)(ib, 1), 2) - (*grid_nodes)((*boundary)(ib, 0), 2);
  x2 = (*grid_nodes)((*boundary)(ib, 2), 0) - (*grid_nodes)((*boundary)(ib, 0), 0);
  y2 = (*grid_nodes)((*boundary)(ib, 2), 1) - (*grid_nodes)((*boundary)(ib, 0), 1);
  z2 = (*grid_nodes)((*boundary)(ib, 2), 2) - (*grid_nodes)((*boundary)(ib, 0), 2);
  // Face normal by a cross product
  nx = y1 * z2 - z1 * y2;
  ny = z1 * x2 - x1 * z2;
  nz = x1 * y2 - y1 * x2;
  norm = sqrt(nx * nx + ny * ny + nz * nz);
  normal[0] = nx / norm;
  normal[1] = ny / norm;
  normal[2] = nz / norm;
}


template<typename dtype>
__host__ __device__ void MC3DCUDA<dtype>::normal (
  int_fast64_t el, long f, dtype *normal
)
{
  int i0, i1, i2;
  if (f == 0)
  {
    i0 = 0;
    i1 = 1;
    i2 = 2;
  }
  else if (f == 1)
  {
    i0 = 0;
    i1 = 1;
    i2 = 3;
  }
  else if (f == 2)
  {
    i0 = 0;
    i1 = 2;
    i2 = 3;
  }
  else if (f == 3)
  {
    i0 = 1;
    i1 = 2;
    i2 = 3;
  }
  else
  {
    normal[0] = normal[1] = normal[2] = 0.0;
    return;
  }

  dtype x1, y1, z1, x2, y2, z2, nx, ny, nz, norm;
  // Two edges of the face
  x1 = (*grid_nodes)((*topology)(el, i1), 0) - (*grid_nodes)((*topology)(el, i0), 0);
  y1 = (*grid_nodes)((*topology)(el, i1), 1) - (*grid_nodes)((*topology)(el, i0), 1);
  z1 = (*grid_nodes)((*topology)(el, i1), 2) - (*grid_nodes)((*topology)(el, i0), 2);
  x2 = (*grid_nodes)((*topology)(el, i2), 0) - (*grid_nodes)((*topology)(el, i0), 0);
  y2 = (*grid_nodes)((*topology)(el, i2), 1) - (*grid_nodes)((*topology)(el, i0), 1);
  z2 = (*grid_nodes)((*topology)(el, i2), 2) - (*grid_nodes)((*topology)(el, i0), 2);
  // Face normal by cross product
  nx = y1 * z2 - z1 * y2;
  ny = z1 * x2 - x1 * z2;
  nz = x1 * y2 - y1 * x2;
  norm = sqrt(nx * nx + ny * ny + nz * nz);
  normal[0] = nx / norm;
  normal[1] = ny / norm;
  normal[2] = nz / norm;
}


template<typename dtype>
__host__ __device__ int MC3DCUDA<dtype>::which_face (
  Photon<dtype>* phot,
  dtype* dist
)
{

  dtype V0[3] = {
    (*grid_nodes)((*topology)(phot->curel, 0), 0),
    (*grid_nodes)((*topology)(phot->curel, 0), 1),
    (*grid_nodes)((*topology)(phot->curel, 0), 2)
  };
  dtype V1[3] = {
    (*grid_nodes)((*topology)(phot->curel, 1), 0),
    (*grid_nodes)((*topology)(phot->curel, 1), 1),
    (*grid_nodes)((*topology)(phot->curel, 1), 2)
  };
  dtype V2[3] = {
    (*grid_nodes)((*topology)(phot->curel, 2), 0),
    (*grid_nodes)((*topology)(phot->curel, 2), 1),
    (*grid_nodes)((*topology)(phot->curel, 2), 2)
  };
  dtype V3[3] = {
    (*grid_nodes)((*topology)(phot->curel, 3), 0),
    (*grid_nodes)((*topology)(phot->curel, 3), 1),
    (*grid_nodes)((*topology)(phot->curel, 3), 2)
  };

  if (phot->curface != 0)
    if (util::ray_triangle_intersects(phot->pos, phot->dir, V0, V1, V2, dist))
      if (*dist > 0.0)
      {
        phot->nextface = 0;
        phot->nextel = (*neighborhood)(phot->curel, phot->nextface);
        return 0;
      }

  if (phot->curface != 1)
    if (util::ray_triangle_intersects(phot->pos, phot->dir, V0, V1, V3, dist))
      if (*dist > 0.0)
      {
        phot->nextface = 1;
        phot->nextel = (*neighborhood)(phot->curel, phot->nextface);
        return 1;
      }

  if (phot->curface != 2)
    if (util::ray_triangle_intersects(phot->pos, phot->dir, V0, V2, V3, dist))
      if (*dist > 0.0)
      {
        phot->nextface = 2;
        phot->nextel = (*neighborhood)(phot->curel, phot->nextface);
        return 2;
      }

  if (phot->curface != 3)
    if (util::ray_triangle_intersects(phot->pos, phot->dir, V1, V2, V3, dist))
      if (*dist > 0.0)
      {
        phot->nextface = 3;
        phot->nextel = (*neighborhood)(phot->curel, phot->nextface);
        return 3;
      }

  return -1;
}


template<typename dtype>
__device__ void MC3DCUDA<dtype>::create_photon (Photon<dtype> *phot, curandState_t* state)
{
  // printf("MC3DCUDA<dtype>::create_photon\n");
  dtype xi = ValoMC::util::rand_closed<curandState_t, dtype>(state);

  dtype n[3], t[3], norm;

  // Array<int_fast64_t> H = *topology;
  // Array<int_fast64_t> HN = *neighborhood;
  // Array<int_fast64_t> BH = *boundary;

  // Array<dtype> r = *grid_nodes;
  // Array<int> LightSources = *light_sources;
  // Array<int> LightSourcesMother = *light_sources_mother;
  // Array<dtype> LightSourcesCDF = *light_sources_cdf;

  // Array<char> BCLightDirectionType = *BC_light_direction_type;
  // Array<char> BCType = *BC_type;
  // Array<dtype> BCLNormal = *BCL_normal;

  // Find boundary element index that will create this photon
  int ib;
  for (ib = 0; ib < light_sources->Nx; ib++) {
    if (xi < (*light_sources_cdf)[ib]) {
      break;
    }
  }
  // Creator faces mother element
  phot->curel = (*light_sources_mother)[ib];
  // Creator face
  ib = (*light_sources)[ib];
  if (-1 - (*neighborhood)(phot->curel, 0) == ib)
    phot->curface = 0;
  else if (-1 - (*neighborhood)(phot->curel, 1) == ib)
    phot->curface = 1;
  else if (-1 - (*neighborhood)(phot->curel, 2) == ib)
    phot->curface = 2;
  else if (-1 - (*neighborhood)(phot->curel, 3) == ib)
    phot->curface = 3;
  else
    phot->curface = -1;

  // Initial photon position uniformly distributed on the boundary element
  dtype w0 = ValoMC::util::rand_open<curandState_t, dtype>(state);
  dtype w1 = ValoMC::util::rand_open<curandState_t, dtype>(state);
  dtype w2 = ValoMC::util::rand_open<curandState_t, dtype>(state);
  phot->pos[0] = (w0 * (*grid_nodes)((*boundary)(ib, 0), 0) +
    w1 * (*grid_nodes)((*boundary)(ib, 1), 0) +
    w2 * (*grid_nodes)((*boundary)(ib, 2), 0)) / (w0 + w1 + w2);

  phot->pos[1] = (w0 * (*grid_nodes)((*boundary)(ib, 0), 1) +
    w1 * (*grid_nodes)((*boundary)(ib, 1), 1) +
    w2 * (*grid_nodes)((*boundary)(ib, 2), 1)) / (w0 + w1 + w2);

  phot->pos[2] = (w0 * (*grid_nodes)((*boundary)(ib, 0), 2) +
    w1 * (*grid_nodes)((*boundary)(ib, 1), 2) +
    w2 * (*grid_nodes)((*boundary)(ib, 2), 2)) / (w0 + w1 + w2);

  // Face normal
  normal(ib, n);

  // Make sure that the normal points inside the element by checking dot product of normal & vector (phot->pos) to center of the element
  t[0] = ((*grid_nodes)((*topology)(phot->curel, 0), 0) +
    (*grid_nodes)((*topology)(phot->curel, 1), 0) +
    (*grid_nodes)((*topology)(phot->curel, 2), 0) +
    (*grid_nodes)((*topology)(phot->curel, 3), 0)) / 4.0 - phot->pos[0];

  t[1] = ((*grid_nodes)((*topology)(phot->curel, 0), 1) +
    (*grid_nodes)((*topology)(phot->curel, 1), 1) +
    (*grid_nodes)((*topology)(phot->curel, 2), 1) +
    (*grid_nodes)((*topology)(phot->curel, 3), 1)) / 4.0 - phot->pos[1];

  t[2] = ((*grid_nodes)((*topology)(phot->curel, 0), 2) +
    (*grid_nodes)((*topology)(phot->curel, 1), 2) +
    (*grid_nodes)((*topology)(phot->curel, 2), 2) +
    (*grid_nodes)((*topology)(phot->curel, 3), 2)) / 4.0 - phot->pos[2];

  norm = sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
  t[0] /= norm;
  t[1] /= norm;
  t[2] /= norm;
  if (n[0] * t[0] + n[1] * t[1] + n[2] * t[2] < 0.0)
  {
    n[0] = -n[0];
    n[1] = -n[1];
    n[2] = -n[2];
  }

  // [AL] Changed the old if clause
  // If light direction type is is 'n' do not read anything from BCLNormal,
  // as the values might not be well defined
  if ((*BC_light_direction_type)[ib] == 'n')
  {
    // No BCLNormal data -> Photos main propagation direction is towards inward normal
    //
    // Select photons intial direction based on the boundary condition
    if ((*BC_type)[ib] == 'l')
    {
      // Laser -- Photons created in normal direction
      phot->dir[0] = n[0];
      phot->dir[1] = n[1];
      phot->dir[2] = n[2];
    }
    else if ((*BC_type)[ib] == 'i')
    {
      // Isotropic -- Photons initial direction probality density is uniform on a sphere
      // Wolfram Mathworld / Sphere Point Picking
      dtype dot, r[3], theta, u;
      do
      {
        theta = 2.0 * M_PI * ValoMC::util::rand_open_up<curandState_t, dtype>(state);
        u = 2.0 * ValoMC::util::rand_closed<curandState_t, dtype>(state) - 1.0;
        r[0] = sqrt(1.0 - pow(u, 2)) * cos(theta);
        r[1] = sqrt(1.0 - pow(u, 2)) * sin(theta);
        r[2] = u;
        dot = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];
      } while (dot <= 0.0);
      phot->dir[0] = r[0];
      phot->dir[1] = r[1];
      phot->dir[2] = r[2];
    }
    else if ((*BC_type)[ib] == 'c' || (*BC_type)[ib] == 'C')
    {
      // Cosinic -- Directivity follows cosine pattern
      dtype phi, theta, dotprodn, dotprod1;
      dtype f[3], e1[3], e2[3];
      // Two edges of the face
      e1[0] = (*grid_nodes)((*boundary)(ib, 1), 0) - (*grid_nodes)((*boundary)(ib, 0), 0);
      e1[1] = (*grid_nodes)((*boundary)(ib, 1), 1) - (*grid_nodes)((*boundary)(ib, 0), 1);
      e1[2] = (*grid_nodes)((*boundary)(ib, 1), 2) - (*grid_nodes)((*boundary)(ib, 0), 2);
      e2[0] = (*grid_nodes)((*boundary)(ib, 2), 0) - (*grid_nodes)((*boundary)(ib, 0), 0);
      e2[1] = (*grid_nodes)((*boundary)(ib, 2), 1) - (*grid_nodes)((*boundary)(ib, 0), 1);
      e2[2] = (*grid_nodes)((*boundary)(ib, 2), 2) - (*grid_nodes)((*boundary)(ib, 0), 2);
      // Cosinically distributed spherical coordinates
      phi = asin(2.0 * ValoMC::util::rand_open<curandState_t, dtype>(state) - 1.0);
      theta = 2.0 * M_PI * ValoMC::util::rand_closed<curandState_t, dtype>(state);
      // Propagation direction of generated photon (random draw around x = 1, y = z = 0 direction with cosinic direction distribution)
      f[0] = cos(phi);
      f[1] = cos(theta) * sin(phi);
      f[2] = sin(theta) * sin(phi);
      // Perform coordinate transformation such that the mean direction [1, 0, 0] is mapped
      // to direction of the surface normal [nx, ny, nz]

      // Form tangential vectors on the element face based on [x1, y1, z1] and [x2, y2, z2]
      // by normalizing these vectors and performing Gram-Schmidt orthogonalization on
      // [x2, y2, z2] based on [nx, ny, nz] and [x1, y1, z1]
      norm = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
      e1[0] /= norm;
      e1[1] /= norm;
      e1[2] /= norm;
      // At this point [nx, ny, nz] and [x1, y1, z1] are orthogonal and hence only
      // [x2, y2, z2] needs to be orthonormalized
      dotprodn = n[0] * e2[0] + n[1] * e2[1] + n[2] * e2[2];
      dotprod1 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];

      e2[0] -= dotprodn * n[0] + dotprod1 * e1[0];
      e2[1] -= dotprodn * n[1] + dotprod1 * e1[1];
      e2[2] -= dotprodn * n[2] + dotprod1 * e1[2];

      norm = sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
      e2[0] /= norm;
      e2[1] /= norm;
      e2[2] /= norm;

      // Now [nx, ny, nz], [x1, y1, z1] and [x2, y2, z2] form orthogonal basis with [nx, ny, nz]
      // corresponding to direction [1, 0, 0] for the generated photon. What is left
      // is to transform generated direction [fx, fy, fz] to the element coordinates
      phot->dir[0] = f[0] * n[0] + f[1] * e1[0] + f[2] * e2[0];
      phot->dir[1] = f[0] * n[1] + f[1] * e1[1] + f[2] * e2[1];
      phot->dir[2] = f[0] * n[2] + f[1] * e1[2] + f[2] * e2[2];
    }
  }
  else
  {
    // BCLNormal provided -> Photons propagate mainly to this direction
    //
    if ((*BC_type)[ib] == 'l' || (*BC_type)[ib] == 'L')
    {
      // Laser -- Photons created in BCLNormal direction
      phot->dir[0] = (*BCL_normal)(ib, 0);
      phot->dir[1] = (*BCL_normal)(ib, 1);
      phot->dir[2] = (*BCL_normal)(ib, 2);
    }
    else if ((*BC_type)[ib] == 'p')
    {
      // [AL] Pencil beam
      // In the current implementation, it is always perpendicular to the surface
      // and wastes time by overwrites
      phot->dir[0] = n[0];
      phot->dir[1] = n[1];
      phot->dir[2] = n[2];

      phot->pos[0] = (*BCL_normal)(ib, 0);
      phot->pos[1] = (*BCL_normal)(ib, 1);
      phot->pos[2] = (*BCL_normal)(ib, 2);
      //printf("shooting photon at %18.10lf %18.10lf %18.10lf\n", phot->pos[0],phot->pos[1],phot->pos[2]);
      //printf("to direction %18.10lf %18.10lf %18.10lf\n", phot->dir[0],phot->dir[1],phot->dir[2]);
    }
    else if ((*BC_type)[ib] == 'i' || (*BC_type)[ib] == 'I')
    {
      // Isotropic -- Photons initial direction probality density is uniform on a sphere
      // Wolfram Mathworld / Sphere Point Picking
      dtype dot, r[3], theta, u;
      do
      {
        theta = 2.0 * M_PI * ValoMC::util::rand_open_up<curandState_t, dtype>(state);
        u = 2.0 * ValoMC::util::rand_closed<curandState_t, dtype>(state) - 1.0;
        r[0] = sqrt(1.0 - pow(u, 2)) * cos(theta);
        r[1] = sqrt(1.0 - pow(u, 2)) * sin(theta);
        r[2] = u;
        dot = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];
      } while (dot <= 0.0);
      phot->dir[0] = r[0];
      phot->dir[1] = r[1];
      phot->dir[2] = r[2];
    }
    else if ((*BC_type)[ib] == 'c' || (*BC_type)[ib] == 'C')
    {
      // Cosinic -- Directivity follows cosine pattern
      dtype phi, theta, dotprodn, dotprod1;
      dtype f[3], e1[3], e2[3];
      // Two edges of the face
      e1[0] = (*grid_nodes)((*boundary)(ib, 1), 0) - (*grid_nodes)((*boundary)(ib, 0), 0);
      e1[1] = (*grid_nodes)((*boundary)(ib, 1), 1) - (*grid_nodes)((*boundary)(ib, 0), 1);
      e1[2] = (*grid_nodes)((*boundary)(ib, 1), 2) - (*grid_nodes)((*boundary)(ib, 0), 2);
      e2[0] = (*grid_nodes)((*boundary)(ib, 2), 0) - (*grid_nodes)((*boundary)(ib, 0), 0);
      e2[1] = (*grid_nodes)((*boundary)(ib, 2), 1) - (*grid_nodes)((*boundary)(ib, 0), 1);
      e2[2] = (*grid_nodes)((*boundary)(ib, 2), 2) - (*grid_nodes)((*boundary)(ib, 0), 2);
      // Cosinically distributed spherical coordinates
      phi = asin(2.0 * ValoMC::util::rand_open<curandState_t, dtype>(state) - 1.0);
      theta = 2.0 * M_PI * ValoMC::util::rand_closed<curandState_t, dtype>(state);
      // Propagation direction of generated photon (random draw around x = 1, y = z = 0 direction with cosinic direction distribution)
      f[0] = cos(phi);
      f[1] = cos(theta) * sin(phi);
      f[2] = sin(theta) * sin(phi);
      // Perform coordinate transformation such that the mean direction [1, 0, 0] is mapped
      // to direction of BCLNormal

      // Form tangential vectors for BCLNormal, based on vectors e1, e2 by performing
      // Gram-Schmidt orthogonalization on e1 & e2
      dotprodn = (*BCL_normal)(ib, 0) * e1[0] + (*BCL_normal)(ib, 1) * e1[1] + (*BCL_normal)(ib, 2) * e1[2];
      e1[0] -= dotprodn * (*BCL_normal)(ib, 0);
      e1[1] -= dotprodn * (*BCL_normal)(ib, 1);
      e1[2] -= dotprodn * (*BCL_normal)(ib, 2);
      norm = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
      e1[0] /= norm;
      e1[1] /= norm;
      e1[2] /= norm;

      dotprodn = (*BCL_normal)(ib, 0) * e2[0] + (*BCL_normal)(ib, 1) * e2[1] + (*BCL_normal)(ib, 2) * e2[2];
      dotprod1 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
      e2[0] -= dotprodn * (*BCL_normal)(ib, 0) + dotprod1 * e1[0];
      e2[1] -= dotprodn * (*BCL_normal)(ib, 1) + dotprod1 * e1[1];
      e2[2] -= dotprodn * (*BCL_normal)(ib, 2) + dotprod1 * e1[2];
      norm = sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
      e2[0] /= norm;
      e2[1] /= norm;
      e2[2] /= norm;

      // Now [nx, ny, nz], [x1, y1, z1] and [x2, y2, z2] form orthogonal basis with [nx, ny, nz]
      // corresponding to direction [1, 0, 0] for the generated photon. What is left
      // is to transform generated direction [fx, fy, fz] to the element coordinates
      phot->dir[0] = f[0] * n[0] + f[1] * e1[0] + f[2] * e2[0];
      phot->dir[1] = f[0] * n[1] + f[1] * e1[1] + f[2] * e2[1];
      phot->dir[2] = f[0] * n[2] + f[1] * e1[2] + f[2] * e2[2];
    }
  }

  phot->nextel = -1;
  phot->nextface = -1;

  phot->weight = 1.0;
  phot->phase = phase0;
}

template<typename dtype>
__device__ void MC3DCUDA<dtype>::scatter_photon (Photon<dtype> *phot, curandState_t* state)
{
  dtype xi, theta, phi;
  dtype dxn, dyn, dzn;

  // Array<dtype> g = *scattering_inhom;
  // Array<dtype> g2 = *scattering_inhom_2;

  // Henye-Greenstein scattering
  if ((*scattering_inhom)[phot->curel] != 0.0)
  {
    xi = ValoMC::util::rand_closed<curandState_t, dtype>(state);
    if ((0.0 < xi) && (xi < 1.0))
      theta = acos((1.0 + (*scattering_inhom_2)[phot->curel] - pow((1.0 - (*scattering_inhom_2)[phot->curel]) / (1.0 - (*scattering_inhom)[phot->curel] * (1.0 - 2.0 * xi)), 2)) / (2.0 * (*scattering_inhom)[phot->curel]));
    else
      theta = (1.0 - xi) * M_PI;
  }
  else
    theta = acos(2.0 * ValoMC::util::rand_closed<curandState_t, dtype>(state) - 1.0);

  phi = 2.0 * M_PI * ValoMC::util::rand_closed<curandState_t, dtype>(state);

  if (fabs(phot->dir[2]) > 0.999)
  {
    dxn = sin(theta) * cos(phi);
    dyn = sin(theta) * sin(phi);
    dzn = phot->dir[2] * cos(theta) / fabs(phot->dir[2]);
  }
  else
  {
    dxn = sin(theta) * (phot->dir[0] * phot->dir[2] * cos(phi) - phot->dir[1] * sin(phi)) / sqrt(1.0 - phot->dir[2] * phot->dir[2]) + phot->dir[0] * cos(theta);
    dyn = sin(theta) * (phot->dir[1] * phot->dir[2] * cos(phi) + phot->dir[0] * sin(phi)) / sqrt(1.0 - phot->dir[2] * phot->dir[2]) + phot->dir[1] * cos(theta);
    dzn = -sin(theta) * cos(phi) * sqrt(1.0 - phot->dir[2] * phot->dir[2]) + phot->dir[2] * cos(theta);
  }

  dtype norm = sqrt(dxn * dxn + dyn * dyn + dzn * dzn);
  dxn /= norm;
  dyn /= norm;
  dzn /= norm;

  phot->dir[0] = dxn;
  phot->dir[1] = dyn;
  phot->dir[2] = dzn;

  // This is to prevent RayTriangleIntersects from misbehaving after scattering event in the PropagatePhoton
  phot->curface = -1;
}

template<typename dtype>
__host__ __device__ void MC3DCUDA<dtype>::mirror_photon (Photon<dtype> *phot, int_fast64_t ib)
{
  dtype n[3], cdot;
  normal(ib, n);
  cdot = n[0] * phot->dir[0] + n[1] * phot->dir[1] + n[2] * phot->dir[2];
  phot->dir[0] -= 2.0 * cdot * n[0];
  phot->dir[1] -= 2.0 * cdot * n[1];
  phot->dir[2] -= 2.0 * cdot * n[2];
}

template<typename dtype>
__host__ __device__ void MC3DCUDA<dtype>::mirror_photon (Photon<dtype> *phot, int_fast64_t el, long f)
{
  dtype n[3], cdot;
  normal(el, f, n);
  cdot = n[0] * phot->dir[0] + n[1] * phot->dir[1] + n[2] * phot->dir[2];
  phot->dir[0] -= 2.0 * cdot * n[0];
  phot->dir[1] -= 2.0 * cdot * n[1];
  phot->dir[2] -= 2.0 * cdot * n[2];
}

template<typename dtype>
__device__ int MC3DCUDA<dtype>::fresnel_photon (Photon<dtype> *phot, curandState_t* state)
{

  // Array<dtype> BCn = *BC_n;
  // Array<dtype> n = *idx_refrc;

  // Likelyhood of reflection:
  //   R = 0.5 ( sin^2(theta_i - theta_t) / sin^2(theta_i + theta_t) + tan^2(theta_i - theta_t) / tan^2(theta_i + theta_t))
  //
  // For theta_i + theta_t < eps:
  //   R = ( (ni / nt - 1) / (ni / nt + 1) )^2
  // which is the limit as theta_i -> 0
  //
  // Transmission/Reflection of a incident direction di, through face with normal n:
  //   costhi = -n * di';
  //   costht = sqrt( 1 - (ni/nt)^2 * (1 - costhi^2) );
  //   dr = di + 2 * costhi * n;
  //   if(costhi > 0);
  //     dt = (ni/nt) * di + (ni/nt * costhi - costht) * n;
  //   else;
  //     dt = (ni/nt) * di + (ni/nt * costhi + costht) * n;
  //   end;

  // normal of the tranmitting face
  dtype nor[3];
  normal((int)phot->curel, (int)phot->nextface, nor);

  dtype nipnt;
  // Check special case where the photon escapes through the boundary
  if (phot->nextel < 0)
    nipnt = (*idx_refrc)[phot->curel] / (*BC_n)[-1 - phot->nextel];
  else
    nipnt = (*idx_refrc)[phot->curel] / (*idx_refrc)[phot->nextel];

  dtype costhi = -(phot->dir[0] * nor[0] + phot->dir[1] * nor[1] + phot->dir[2] * nor[2]);

  if (1.0 - pow(nipnt, 2) * (1.0 - pow(costhi, 2)) <= 0.0)
  {
    // Total reflection due to critical angle of Snell's law
    phot->dir[0] += 2.0 * costhi * nor[0];
    phot->dir[1] += 2.0 * costhi * nor[1];
    phot->dir[2] += 2.0 * costhi * nor[2];
    phot->curface = phot->nextface;
    return (1);
  }

  dtype costht = sqrt(1.0 - pow(nipnt, 2) * (1.0 - pow(costhi, 2)));

  dtype thi;
  if (costhi > 0.0)
    thi = acos(costhi);
  else
    thi = acos(-costhi);
  dtype tht = acos(costht);
  dtype R;
  if (!(sin(thi + tht) > ValoMC::util::eps_map<dtype>::eps))
    R = pow((nipnt - 1.0) / (nipnt + 1.0), 2);
  else
    R = 0.5 * (pow(sin(thi - tht) / sin(thi + tht), 2) + pow(tan(thi - tht) / tan(thi + tht), 2));
  dtype xi = ValoMC::util::rand_closed<curandState_t, dtype>(state);

  if (xi <= R)
  {
    // Photon is reflected
    phot->dir[0] += 2.0 * costhi * nor[0];
    phot->dir[1] += 2.0 * costhi * nor[1];
    phot->dir[2] += 2.0 * costhi * nor[2];
    phot->curface = phot->nextface;
    return (1);
  }

  // Photon is transmitted - update propagation direction via Snell's law
  if (costhi > 0.0)
  {
    phot->dir[0] = nipnt * phot->dir[0] + (nipnt * costhi - costht) * nor[0];
    phot->dir[1] = nipnt * phot->dir[1] + (nipnt * costhi - costht) * nor[1];
    phot->dir[2] = nipnt * phot->dir[2] + (nipnt * costhi - costht) * nor[2];
  }
  else
  {
    phot->dir[0] = nipnt * phot->dir[0] + (nipnt * costhi + costht) * nor[0];
    phot->dir[1] = nipnt * phot->dir[1] + (nipnt * costhi + costht) * nor[1];
    phot->dir[2] = nipnt * phot->dir[2] + (nipnt * costhi + costht) * nor[2];
  }

  return 0;
}

/**
 * [MC3DCUDA<dtype>::propagate_photon_single_step_atomic  description]
 * @param  phot  [description]
 * @param  state [description]
 * @param  prop  [description]
 * @param  dist  [description]
 * @param  ds    [description]
 * @param  ib    [description]
 * @return       0, 1, or 2. 0 if photon has died, 1 if it needs to be scattered, 2 if its propagation needs to be continued.
 */
template<typename dtype>
__device__ int MC3DCUDA<dtype>::propagate_photon_single_step_atomic (
  Photon<dtype> *phot,
  curandState_t* state,
  dtype* prop,
  dtype* dist,
  dtype* ds,
  int_fast64_t* ib
)
{
  // Check through which face the photon will exit the current element
  if (which_face(phot, dist) == -1)
  {
    // [DS]
    // loss++;
    return 0;
  }

  // Travel distance -- Either propagate to the boundary of the element, or to the end of the leap, whichever is closer
  *ds = fmin(*prop, *dist);

  // Move photon
  phot->pos[0] += phot->dir[0] * (*ds);
  phot->pos[1] += phot->dir[1] * (*ds);
  phot->pos[2] += phot->dir[2] * (*ds);

  // Upgrade element fluence
  if (omega <= 0.0) {
    // Unmodulated light
    if ((*absorption)[phot->curel] > 0.0) {
      // (*pow_den_vol_real)[phot->curel] += (1.0 - exp(-(*absorption)[phot->curel] * (*ds))) * phot->weight;
      atomicAdd(&(*pow_den_vol_real)[phot->curel], (1.0 - exp(-(*absorption)[phot->curel] * (*ds))) * phot->weight);
      // (1.0 - exp(-(*absorption)[phot->curel] * (*ds))) * phot->weight;
    } else {
      // (*pow_den_vol_real)[phot->curel] += phot->weight * (*ds);
      atomicAdd(&(*pow_den_vol_real)[phot->curel], phot->weight * (*ds));
      // phot->weight * (*ds);
    }
  } else {
    // Modulated light

    /*
  cls;

  syms w0 mua k x ph0 s real;

  % k = 0; ph0 = 0;

  e = w0 * exp(-mua * x - j * (k * x + ph0));

  g = int(e, x, 0, s);

  syms a b real;

  f = (a + i * b) / (mua + i * k);

  % Change of element as photon passes it
  pretty(simplify( real( g * (mua + i * k) ) ))
  pretty(simplify( imag( g * (mua + i * k) ) ))

  % Final divider / normalization
  pretty( simplify( real(f) ) )
  pretty( simplify( imag(f) ) )
*/

    // (*pow_den_vol_real)[phot->curel] += phot->weight * (cos(phot->phase) - cos(-phot->phase - (*wave_number)[phot->curel] * (*ds)) * exp(-(*absorption)[phot->curel] * (*ds)));
    // (*pow_den_vol_imag)[phot->curel] += phot->weight * (-sin(phot->phase) + sin(phot->phase + (*wave_number)[phot->curel] * (*ds)) * exp(-(*absorption)[phot->curel] * (*ds)));
    atomicAdd(&(*pow_den_vol_real)[phot->curel], phot->weight * (cos(phot->phase) - cos(-phot->phase - (*wave_number)[phot->curel] * (*ds)) * exp(-(*absorption)[phot->curel] * (*ds))));
    atomicAdd(&(*pow_den_vol_imag)[phot->curel], phot->weight * (-sin(phot->phase) + sin(phot->phase + (*wave_number)[phot->curel] * (*ds)) * exp(-(*absorption)[phot->curel] * (*ds))));
    // phot->weight * (cos(phot->phase) - cos(-phot->phase - (*wave_number)[phot->curel] * (*ds)) * exp(-(*absorption)[phot->curel] * (*ds)));
    // phot->weight * (-sin(phot->phase) + sin(phot->phase + (*wave_number)[phot->curel] * (*ds)) * exp(-(*absorption)[phot->curel] * (*ds)));

    phot->phase += (*wave_number)[phot->curel] * (*ds);
  }

  // Upgrade photon weigh
  phot->weight *= exp(-(*absorption)[phot->curel] * (*ds));

  // Photon has reached a situation where it has to be scattered
  *prop -= *ds;
  if (*prop <= 0.0) {
    // break;
    return 1;
  }

  // Otherwise the photon will continue to pass through the boundaries of the current element

  // Test for boundary conditions
  if (phot->nextel < 0)
  {
    // Boundary element index
    *ib = -1 - phot->nextel;

    if (((*BC_type)[(*ib)] == 'm') || ((*BC_type)[(*ib)] == 'L') || ((*BC_type)[(*ib)] == 'I') || ((*BC_type)[(*ib)] == 'C'))
    {
      // Mirror boundary condition -- Reflect the photon
      mirror_photon(phot, (*ib));
      phot->curface = phot->nextface;
      return 2;
      // continue;
    }
    else
    {
      // Absorbing (a, l, i and c)
      // Check for mismatch between inner & outer index of refraction causes Fresnel transmission
      if ((*BC_n)[(*ib)] > 0.0) {
        if (fresnel_photon(phot, state)) {
          return 2;
          // continue;
        }
      }

      if (omega <= 0.0) {
        // (*pow_den_boun_real)[(*ib)] += phot->weight;
        atomicAdd(&(*pow_den_boun_real)[(*ib)], phot->weight);
        // phot->weight;
      } else {
        // (*pow_den_boun_real)[(*ib)] += phot->weight * cos(phot->phase);
        // (*pow_den_boun_imag)[(*ib)] -= phot->weight * sin(phot->phase);
        atomicAdd(&(*pow_den_boun_real)[(*ib)], phot->weight * cos(phot->phase));
        atomicAdd(&(*pow_den_boun_imag)[(*ib)], -phot->weight * sin(phot->phase));
        // phot->weight * cos(phot->phase);
        // -phot->weight * sin(phot->phase);

      }
      // Photon propagation will terminate
      return 0;
    }
  }

  // Test transmission from vacuum -> scattering media
  if (((*scattering)[phot->curel] <= 0.0) && ((*scattering)[phot->nextel] > 0.0))
  {
    // Draw new propagation distance -- otherwise photon might travel without scattering
    *prop = -log(ValoMC::util::rand_open<curandState_t, dtype>(state)) / (*scattering)[phot->nextel];
  }

  // Test for surival of the photon via roulette
  if (phot->weight < weight0)
  {
    if (ValoMC::util::rand_closed<curandState_t, dtype>(state) > chance) {
      return 0;
    }
    phot->weight /= chance;
  }

  // Fresnel transmission/reflection
  if ((*idx_refrc)[phot->curel] != (*idx_refrc)[phot->nextel])
  {
    if (fresnel_photon(phot, state)) {
      return 1;
    }
  }

  // Upgrade remaining photon propagation lenght in case it is transmitted to different mus domain
  *prop *= (*scattering)[phot->curel] / (*scattering)[phot->nextel];


  // Update current face of the photon to that face which it will be on in the next element
  if ((*neighborhood)(phot->nextel, 0) == phot->curel)
    phot->curface = 0;
  else if ((*neighborhood)(phot->nextel, 1) == phot->curel)
    phot->curface = 1;
  else if ((*neighborhood)(phot->nextel, 2) == phot->curel)
    phot->curface = 2;
  else if ((*neighborhood)(phot->nextel, 3) == phot->curel)
    phot->curface = 3;
  else
  {
    // loss++;
    return 0;
  }

  // Update current element of the photon
  phot->curel = phot->nextel;
  return 2;
}

// Propagate until the photon dies
template<typename dtype>
__device__ void MC3DCUDA<dtype>::propagate_photon_atomic (Photon<dtype> *phot, curandState_t* state)
{
  dtype prop;
  dtype dist;
  dtype ds;
  int_fast64_t ib;

  char BC_type_ib;
  // int single_step;
  // bool alive = true;
  //
  // prop = -log(ValoMC::util::rand_open<curandState_t, dtype>(state)) / (*scattering)[phot->curel];
  //
  // while (alive)
  // {
  //
  //   single_step = propagate_photon_single_step_atomic(phot, state, &prop, &dist, &ds, &ib);
  //
  //   if (single_step == 0) {
  //     alive = false;
  //   } else if (single_step == 1) {
  //     if (blockIdx.x == 0 && threadIdx.x == 0) {
  //       printf("here\n");
  //     }
  //     if ((*scattering)[phot->curel] > 0.0) {
  //       scatter_photon(phot, state);
  //     }
  //     prop = -log(ValoMC::util::rand_open<curandState_t, dtype>(state)) / (*scattering)[phot->curel];
  //   }
  // }

  while (1) {
    // Draw the propagation distance

    prop = -log(ValoMC::util::rand_open<curandState_t, dtype>(state)) / (*scattering)[phot->curel];

    // Propagate until the current propagation distance runs out (and a scattering will occur)

    while (1)
    {
      // Check through which face the photon will exit the current element
      // [DS]
      if (which_face(phot, &dist) == -1)
      {
        // loss++;
        return;
      }

      // Travel distance -- Either propagate to the boundary of the element, or to the end of the leap, whichever is closer
      ds = fmin(prop, dist);

      // Move photon
      phot->pos[0] += phot->dir[0] * ds;
      phot->pos[1] += phot->dir[1] * ds;
      phot->pos[2] += phot->dir[2] * ds;

      // Upgrade element fluence
      if (omega <= 0.0)
      {
        // Unmodulated light
        if ((*absorption)[phot->curel] > 0.0)
        {
          // (*pow_den_vol_real)[phot->curel] += (1.0 - exp(-(*absorption)[phot->curel] * ds)) * phot->weight;
          atomicAdd(&(*pow_den_vol_real)[phot->curel], (1.0 - exp(-(*absorption)[phot->curel] * ds)) * phot->weight);
          // (1.0 - exp(-(*absorption)[phot->curel] * ds)) * phot->weight;
        }
        else
        {
          // (*pow_den_vol_real)[phot->curel] += phot->weight * ds;
          atomicAdd(&(*pow_den_vol_real)[phot->curel], phot->weight * ds);
          // phot->weight * ds;
        }
      }
      else
      {
        // Modulated light

        /*
      cls;

      syms w0 mua k x ph0 s real;

      % k = 0; ph0 = 0;

      e = w0 * exp(-mua * x - j * (k * x + ph0));

      g = int(e, x, 0, s);

      syms a b real;

      f = (a + i * b) / (mua + i * k);

      % Change of element as photon passes it
      pretty(simplify( real( g * (mua + i * k) ) ))
      pretty(simplify( imag( g * (mua + i * k) ) ))

      % Final divider / normalization
      pretty( simplify( real(f) ) )
      pretty( simplify( imag(f) ) )
  */

        // (*pow_den_vol_real)[phot->curel] += phot->weight * (cos(phot->phase) - cos(-phot->phase - (*wave_number)[phot->curel] * ds) * exp(-(*absorption)[phot->curel] * ds));
        // (*pow_den_vol_imag)[phot->curel] += phot->weight * (-sin(phot->phase) + sin(phot->phase + (*wave_number)[phot->curel] * ds) * exp(-(*absorption)[phot->curel] * ds));
        atomicAdd(&(*pow_den_vol_real)[phot->curel], phot->weight * (cos(phot->phase) - cos(-phot->phase - (*wave_number)[phot->curel] * ds) * exp(-(*absorption)[phot->curel] * ds)));
        atomicAdd(&(*pow_den_vol_imag)[phot->curel], phot->weight * (-sin(phot->phase) + sin(phot->phase + (*wave_number)[phot->curel] * ds) * exp(-(*absorption)[phot->curel] * ds)));
        // phot->weight * (cos(phot->phase) - cos(-phot->phase - (*wave_number)[phot->curel] * ds) * exp(-(*absorption)[phot->curel] * ds));
        // phot->weight * (-sin(phot->phase) + sin(phot->phase + (*wave_number)[phot->curel] * ds) * exp(-(*absorption)[phot->curel] * ds));

        phot->phase += (*wave_number)[phot->curel] * ds;
      }

      // Upgrade photon weigh
      phot->weight *= exp(-(*absorption)[phot->curel] * ds);

      // Photon has reached a situation where it has to be scattered
      prop -= ds;
      if (prop <= 0.0) {
        break;
      }

      // Otherwise the photon will continue to pass through the boundaries of the current element

      // Test for boundary conditions
      if (phot->nextel < 0)
      {
        // Boundary element index
        ib = -1 - phot->nextel;
        BC_type_ib = (*BC_type)[ib];
        if ((BC_type_ib == 'm') || (BC_type_ib == 'L') || (BC_type_ib == 'I') || (BC_type_ib == 'C'))
        {
          // Mirror boundary condition -- Reflect the photon
          mirror_photon(phot, ib);
          phot->curface = phot->nextface;
          continue;
        }
        else
        {
          // Absorbing (a, l, i and c)
          // Check for mismatch between inner & outer index of refraction causes Fresnel transmission
          if ((*BC_n)[ib] > 0.0) {
            if (fresnel_photon(phot, state)) {
              continue;
            }
          }

          if (omega <= 0.0) {
            // (*pow_den_boun_real)[ib] += phot->weight;
            atomicAdd(&(*pow_den_boun_real)[ib], phot->weight);
            // phot->weight;
          } else {
            // (*pow_den_boun_real)[ib] += phot->weight * cos(phot->phase);
            // (*pow_den_boun_imag)[ib] -= phot->weight * sin(phot->phase);
            atomicAdd(&(*pow_den_boun_real)[ib], phot->weight * cos(phot->phase));
            atomicAdd(&(*pow_den_boun_imag)[ib], -phot->weight * sin(phot->phase));
            // phot->weight * cos(phot->phase);
            // -phot->weight * sin(phot->phase);

          }
          // Photon propagation will terminate
          return;
        }
      }

      // Test transmission from vacuum -> scattering media
      if (((*scattering)[phot->curel] <= 0.0) && ((*scattering)[phot->nextel] > 0.0))
      {
        // Draw new propagation distance -- otherwise photon might travel without scattering
        prop = -log(ValoMC::util::rand_open<curandState_t, dtype>(state)) / (*scattering)[phot->nextel];
      }

      // Test for surival of the photon via roulette
      if (phot->weight < weight0)
      {
        if (ValoMC::util::rand_closed<curandState_t, dtype>(state) > chance) {
          return;
        }
        phot->weight /= chance;
      }

      // Fresnel transmission/reflection
      if ((*idx_refrc)[phot->curel] != (*idx_refrc)[phot->nextel])
      {
        if (fresnel_photon(phot, state)) {
          continue;
        }
      }

      // Upgrade remaining photon propagation lenght in case it is transmitted to different mus domain
      prop *= (*scattering)[phot->curel] / (*scattering)[phot->nextel];


      // Update current face of the photon to that face which it will be on in the next element
      if ((*neighborhood)(phot->nextel, 0) == phot->curel)
        phot->curface = 0;
      else if ((*neighborhood)(phot->nextel, 1) == phot->curel)
        phot->curface = 1;
      else if ((*neighborhood)(phot->nextel, 2) == phot->curel)
        phot->curface = 2;
      else if ((*neighborhood)(phot->nextel, 3) == phot->curel)
        phot->curface = 3;
      else
      {
        // loss++;
        return;
      }

      // Update current element of the photon
      phot->curel = phot->nextel;
    }

    // Scatter photon
    if ((*scattering)[phot->curel] > 0.0) {
      scatter_photon(phot, state);
    }
  }

}


template<typename dtype>
void MC3DCUDA<dtype>::allocate () {
  if (! is_allocated) {
    gpuErrchk(cudaMalloc((void**)&topology, sizeof(Array<int_fast64_t>)));
    gpuErrchk(cudaMalloc((void**)&neighborhood, sizeof(Array<int_fast64_t>)));
    gpuErrchk(cudaMalloc((void**)&boundary, sizeof(Array<int_fast64_t>)));

    gpuErrchk(cudaMalloc((void**)&grid_nodes, sizeof(Array<dtype>)));

    gpuErrchk(cudaMalloc((void**)&light_sources, sizeof(Array<int>)));
    gpuErrchk(cudaMalloc((void**)&light_sources_mother, sizeof(Array<int>)));
    gpuErrchk(cudaMalloc((void**)&light_sources_cdf, sizeof(Array<dtype>)));

    gpuErrchk(cudaMalloc((void**)&BC_light_direction_type, sizeof(Array<char>)));
    gpuErrchk(cudaMalloc((void**)&BCL_normal, sizeof(Array<dtype>)));
    gpuErrchk(cudaMalloc((void**)&BC_n, sizeof(Array<dtype>)));
    gpuErrchk(cudaMalloc((void**)&BC_type, sizeof(Array<char>)));

    gpuErrchk(cudaMalloc((void**)&absorption, sizeof(Array<dtype>)));
    gpuErrchk(cudaMalloc((void**)&scattering, sizeof(Array<dtype>)));

    gpuErrchk(cudaMalloc((void**)&scattering_inhom, sizeof(Array<dtype>)));
    gpuErrchk(cudaMalloc((void**)&idx_refrc, sizeof(Array<dtype>)));
    gpuErrchk(cudaMalloc((void**)&wave_number, sizeof(Array<dtype>)));
    gpuErrchk(cudaMalloc((void**)&scattering_inhom_2, sizeof(Array<dtype>)));

    gpuErrchk(cudaMalloc((void**)&pow_den_vol_real, sizeof(Array<dtype>)));
    gpuErrchk(cudaMalloc((void**)&pow_den_vol_imag, sizeof(Array<dtype>)));

    gpuErrchk(cudaMalloc((void**)&pow_den_boun_real, sizeof(Array<dtype>)));
    gpuErrchk(cudaMalloc((void**)&pow_den_boun_imag, sizeof(Array<dtype>)));

    // allocate curand states
    gpuErrchk(cudaMalloc((void**)&states, sizeof(curandState_t)*states_size));
    is_allocated = true;
  }

}

template<typename dtype>
void MC3DCUDA<dtype>::deallocate () {
  if (is_allocated) {
    gpuErrchk(cudaFree(topology));
    gpuErrchk(cudaFree(neighborhood));
    gpuErrchk(cudaFree(boundary));

    gpuErrchk(cudaFree(grid_nodes));

    gpuErrchk(cudaFree(light_sources));
    gpuErrchk(cudaFree(light_sources_mother));
    gpuErrchk(cudaFree(light_sources_cdf));

    gpuErrchk(cudaFree(BC_light_direction_type));
    gpuErrchk(cudaFree(BCL_normal));
    gpuErrchk(cudaFree(BC_n));
    gpuErrchk(cudaFree(BC_type));

    gpuErrchk(cudaFree(absorption));
    gpuErrchk(cudaFree(scattering));

    gpuErrchk(cudaFree(scattering_inhom));
    gpuErrchk(cudaFree(idx_refrc));
    gpuErrchk(cudaFree(wave_number));
    gpuErrchk(cudaFree(scattering_inhom_2));

    gpuErrchk(cudaFree(pow_den_vol_real));
    gpuErrchk(cudaFree(pow_den_vol_imag));

    gpuErrchk(cudaFree(pow_den_boun_real));
    gpuErrchk(cudaFree(pow_den_boun_imag));

    // allocate curand states
    gpuErrchk(cudaFree(states));
  }
}

template<typename dtype>
void MC3DCUDA<dtype>::h2d () {
  ValoMC::util::h2d(topology, &mc3d.H);
  ValoMC::util::h2d(neighborhood, &mc3d.HN);
  ValoMC::util::h2d(boundary, &mc3d.BH);

  ValoMC::util::h2d(grid_nodes, &mc3d.r);

  ValoMC::util::h2d(light_sources, &mc3d.LightSources);
  ValoMC::util::h2d(light_sources_mother, &mc3d.LightSourcesMother);
  ValoMC::util::h2d(light_sources_cdf, &mc3d.LightSourcesCDF);

  ValoMC::util::h2d(BC_light_direction_type, &mc3d.BCLightDirectionType);
  ValoMC::util::h2d(BCL_normal, &mc3d.BCLNormal);
  ValoMC::util::h2d(BC_n, &mc3d.BCn);
  ValoMC::util::h2d(BC_type, &mc3d.BCType);

  ValoMC::util::h2d(absorption, &mc3d.mua);
  ValoMC::util::h2d(scattering, &mc3d.mus);

  ValoMC::util::h2d(scattering_inhom, &mc3d.g);
  ValoMC::util::h2d(idx_refrc, &mc3d.n);
  ValoMC::util::h2d(wave_number, &mc3d.k);
  ValoMC::util::h2d(scattering_inhom_2, &mc3d.g2);

  ValoMC::util::copy_attributes(pow_den_vol_real, &mc3d.ER);
  ValoMC::util::reserve(pow_den_vol_real, mc3d.ER.N);
  ValoMC::util::copy_attributes(pow_den_vol_imag, &mc3d.EI);
  ValoMC::util::reserve(pow_den_vol_imag, mc3d.EI.N);

  ValoMC::util::copy_attributes(pow_den_boun_real, &mc3d.EBR);
  ValoMC::util::reserve(pow_den_boun_real, mc3d.EBR.N);
  ValoMC::util::copy_attributes(pow_den_boun_imag, &mc3d.EBI);
  ValoMC::util::reserve(pow_den_boun_imag, mc3d.EBI.N);
}

template<typename dtype>
void MC3DCUDA<dtype>::d2h () {
  ValoMC::util::d2h(&mc3d.ER, pow_den_vol_real);
  ValoMC::util::d2h(&mc3d.EI, pow_den_vol_imag);

  ValoMC::util::d2h(&mc3d.EBR, pow_den_boun_real);
  ValoMC::util::d2h(&mc3d.EBI, pow_den_boun_imag);
}

template<typename dtype>
void MC3DCUDA<dtype>::monte_carlo (bool use_alt) {
  if (use_alt) {
    monte_carlo_atomic_alt();
  } else {
    monte_carlo_atomic();
  }
}

template<typename dtype>
void MC3DCUDA<dtype>::monte_carlo_atomic () {

  MC3DCUDA* mc3dcuda_d;
  gpuErrchk(cudaMalloc((void**)&mc3dcuda_d, sizeof(MC3DCUDA)));
  gpuErrchk(cudaMemcpy(mc3dcuda_d, this, sizeof(MC3DCUDA), cudaMemcpyHostToDevice));

  unsigned block_size_init_state = max_block_size_init_state;
  unsigned grid_size_init_state = states_size / block_size_init_state;
  if (grid_size_init_state == 0) {
    grid_size_init_state++;
  }

 /* unsigned block_size_monte_carlo = max_block_size_monte_carlo;
  unsigned grid_size_monte_carlo = states_size / block_size_monte_carlo;
  if (grid_size_monte_carlo == 0) {
    grid_size_monte_carlo++;
  }*/

  unsigned block_size_monte_carlo = max_block_size_monte_carlo;
  unsigned grid_size_monte_carlo = states_size / (block_size_monte_carlo/32);
  if (grid_size_monte_carlo == 0) {
    grid_size_monte_carlo++;
  }

  // std::cerr << "_init_state<<<" << grid_size_init_state << ", " << block_size_init_state << ">>>" << std::endl;
  // std::cerr << "_monte_carlo_atomic<<<" << grid_size_monte_carlo << ", " << block_size_monte_carlo << ">>>" << std::endl;

  _init_state<<<grid_size_init_state, block_size_init_state>>>(mc3dcuda_d);
  gpuErrchk(cudaGetLastError());
  _monte_carlo_atomic<<<grid_size_monte_carlo, block_size_monte_carlo>>>(mc3dcuda_d);
  gpuErrchk(cudaGetLastError());

  gpuErrchk(cudaFree(mc3dcuda_d));
}


template<typename dtype>
void MC3DCUDA<dtype>::monte_carlo_atomic_alt () {

  MC3DCUDA* mc3dcuda_d;
  gpuErrchk(cudaMalloc((void**)&mc3dcuda_d, sizeof(MC3DCUDA)));
  gpuErrchk(cudaMemcpy(mc3dcuda_d, this, sizeof(MC3DCUDA), cudaMemcpyHostToDevice));

  unsigned nphotons = get_nphotons();

  Photon<dtype>* photons;
  gpuErrchk(cudaMalloc((void**)&photons, sizeof(Photon<dtype>)*nphotons));
  gpuErrchk(cudaMemcpy(photons, this, sizeof(Photon<dtype>)*nphotons, cudaMemcpyHostToDevice));

  PhotonAttr<dtype>* photon_attrs;
  gpuErrchk(cudaMalloc((void**)&photon_attrs, sizeof(PhotonAttr<dtype>)*nphotons));
  gpuErrchk(cudaMemcpy(photon_attrs, this, sizeof(PhotonAttr<dtype>)*nphotons, cudaMemcpyHostToDevice));

  int* dead;
  gpuErrchk(cudaMalloc((void**)&dead, sizeof(int)*nphotons));
  gpuErrchk(cudaMemcpy(dead, this, sizeof(int)*nphotons, cudaMemcpyHostToDevice));


  unsigned block_size_init_state = max_block_size_init_state;
  unsigned grid_size_init_state = states_size / block_size_init_state;
  if (grid_size_init_state == 0) {
    grid_size_init_state++;
  }

  unsigned block_size_monte_carlo = max_block_size_monte_carlo;
  unsigned grid_size_monte_carlo = states_size / block_size_monte_carlo;
  if (grid_size_monte_carlo == 0) {
    grid_size_monte_carlo++;
  }

  // unsigned block_size_monte_carlo = max_block_size_monte_carlo;
  // unsigned grid_size_monte_carlo = states_size / (block_size_monte_carlo/32);
  // if (grid_size_monte_carlo == 0) {
  //   grid_size_monte_carlo++;
  // }

  // std::cerr << "_init_state<<<" << grid_size_init_state << ", " << block_size_init_state << ">>>" << std::endl;
  // std::cerr << "_monte_carlo_atomic<<<" << grid_size_monte_carlo << ", " << block_size_monte_carlo << ">>>" << std::endl;

  _init_state<<<grid_size_init_state, block_size_init_state>>>(mc3dcuda_d);
  gpuErrchk(cudaGetLastError());

  _monte_carlo_atomic_create_photons<<<grid_size_monte_carlo, block_size_monte_carlo>>>(mc3dcuda_d, photons, photon_attrs, dead);
  gpuErrchk(cudaGetLastError());
  int total = 0;

  while (total != nphotons) {
    _monte_carlo_atomic_single_step<<<grid_size_monte_carlo, block_size_monte_carlo>>>(mc3dcuda_d, photons, photon_attrs, dead);
    total = thrust::reduce(thrust::device, dead, dead+nphotons, 0);
  }

  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaFree(mc3dcuda_d));
  gpuErrchk(cudaFree(photons));
  gpuErrchk(cudaFree(photon_attrs));
}




template<typename dtype>
unsigned long MC3DCUDA<dtype>::get_total_memory_usage () {
  unsigned long total_memory_usage = 0;
  total_memory_usage += sizeof(int_fast64_t) * mc3d.H.N;
  total_memory_usage += sizeof(int_fast64_t) * mc3d.HN.N;
  total_memory_usage += sizeof(int_fast64_t) * mc3d.BH.N;

  total_memory_usage += sizeof(dtype) * mc3d.r.N;

  total_memory_usage += sizeof(int) * mc3d.LightSources.N;
  total_memory_usage += sizeof(int) * mc3d.LightSourcesMother.N;
  total_memory_usage += sizeof(dtype) * mc3d.LightSourcesCDF.N;

  total_memory_usage += sizeof(char) * mc3d.BCLightDirectionType.N;
  total_memory_usage += sizeof(dtype) * mc3d.BCLNormal.N;
  total_memory_usage += sizeof(dtype) * mc3d.BCn.N;
  total_memory_usage += sizeof(char) * mc3d.BCType.N;

  total_memory_usage += sizeof(dtype) * mc3d.mua.N;
  total_memory_usage += sizeof(dtype) * mc3d.mus.N;
  total_memory_usage += sizeof(dtype) * mc3d.g.N;
  total_memory_usage += sizeof(dtype) * mc3d.n.N;
  total_memory_usage += sizeof(dtype) * mc3d.k.N;
  total_memory_usage += sizeof(dtype) * mc3d.g2.N;

  total_memory_usage += sizeof(dtype) * mc3d.ER.N;
  total_memory_usage += sizeof(dtype) * mc3d.EI.N;

  total_memory_usage += sizeof(dtype) * mc3d.EBR.N;
  total_memory_usage += sizeof(dtype) * mc3d.EBI.N;

  total_memory_usage += sizeof(curandState_t) * states_size;

  return total_memory_usage;
}

template class MC3DCUDA<float> ;
template class MC3DCUDA<double> ;

}
