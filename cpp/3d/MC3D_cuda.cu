#include <stdio.h> // for fprintf, stderr

#include "MC3D_cuda.cuh"
#include "MC3D_util.cuh"

namespace ValoMC {

__global__ void init_state (MC3DCUDA* mc3d) {
  const unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;
  const unsigned total_size_x = gridDim.x*blockDim.x;
  const unsigned states_size = mc3d->get_states_size();
  if (idx > states_size) {
    return;
  }
  const unsigned seed = mc3d->get_seed();

  for (unsigned istate=idx; istate<states_size; istate+=total_size_x) {
    curand_init(seed, idx, 0, &mc3d->get_states()[istate]);
  }
}


__global__ void monte_carlo (MC3DCUDA* mc3d) {
  const unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;
  const unsigned total_size_x = gridDim.x*blockDim.x;
  const unsigned states_size = mc3d->get_states_size();
  const unsigned increment_size = total_size_x > states_size ? states_size: total_size_x;

  if (idx > increment_size) {
    return;
  }
  curandState_t local_state = mc3d->get_states()[idx];
  Photon* photon;
  for (unsigned iphoton=idx; iphoton<mc3d.get_nphotons(); iphoton+=increment_size) {
    mc3d.create_photon(photon, &local_state);
    mc3d.propagate_photon(photon, &local_state);
  }
}

void MC3DCUDA::init () {
  omega = mc3d.omega;
  seed = mc3d.seed;
  weight0 = mc3d.weight0;
  chance = mc3d.chance;
  phase0 = mc3d.phase0;
  nphotons = static_cast<unsigned long>(mc3d.Nphoton);
}


  // Array<int_fast64_t> H = *topology;
  // Array<int_fast64_t> HN = *neighborhood;
  // Array<int_fast64_t> BH = *boundary;
  //
  // Array<double> r = *grid_nodes;
  // Array<int> LightSources = *light_sources;
  // Array<int> LightSourcesMother = *light_sources_mother;
  // Array<double> LightSourcesCDF = *light_sources_cdf;
  //
  // Array<char> BCLightDirectionType = *BC_light_direction_type;
  // Array<char> BCType = *BC_type;
  // Array<double> BCLNormal = *BCL_normal;
  // Array<double> BCn = *BC_n;

  // Array<double> mua = *absorption;
  // Array<double> mus = *scattering;
  // Array<double> g = *scattering_inhom;
  // Array<double> n = *idx_refrc;
  // Array<double> k = *wave_number;
  // Array<double> g2 = *scattering_inhom_2;
  //
  // Array<double> ER = *pow_den_vol_real;
  // Array<double> EI = *pow_den_vol_imag;
  //
  // Array<double> EBR = *pow_den_boun_real;
  // Array<double> EBI = *pow_den_boun_imag;


__host__ __device__ void MC3DCUDA::normal (
  int_fast64_t ib, double *normal
)
{
  double x1, y1, z1, x2, y2, z2, nx, ny, nz, norm;

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


__host__ __device__ void MC3DCUDA::normal (
  int_fast64_t el, long f, double *normal
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

  double x1, y1, z1, x2, y2, z2, nx, ny, nz, norm;
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


__host__ __device__ int MC3DCUDA::which_face (
  Photon* phot,
  double* dist
)
{

  double V0[3] = {
    (*grid_nodes)((*topology)(phot->curel, 0), 0),
    (*grid_nodes)((*topology)(phot->curel, 0), 1),
    (*grid_nodes)((*topology)(phot->curel, 0), 2)
  };
  double V1[3] = {
    (*grid_nodes)((*topology)(phot->curel, 1), 0),
    (*grid_nodes)((*topology)(phot->curel, 1), 1),
    (*grid_nodes)((*topology)(phot->curel, 1), 2)
  };
  double V2[3] = {
    (*grid_nodes)((*topology)(phot->curel, 2), 0),
    (*grid_nodes)((*topology)(phot->curel, 2), 1),
    (*grid_nodes)((*topology)(phot->curel, 2), 2)
  };
  double V3[3] = {
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


__device__ void MC3DCUDA::create_photon (Photon* phot, curandState_t* state)
{
  double xi = ValoMC::util::rand_closed<curandState_t, double>(state);

  double n[3], t[3], norm;

  Array<int_fast64_t> H = *topology;
  Array<int_fast64_t> HN = *neighborhood;
  Array<int_fast64_t> BH = *boundary;

  Array<double> r = *grid_nodes;
  Array<int> LightSources = *light_sources;
  Array<int> LightSourcesMother = *light_sources_mother;
  Array<double> LightSourcesCDF = *light_sources_cdf;

  Array<char> BCLightDirectionType = *BC_light_direction_type;
  Array<char> BCType = *BC_type;
  Array<double> BCLNormal = *BCL_normal;

  // Find boundary element index that will create this photon
  int ib;
  for (ib = 0; ib < LightSources.Nx; ib++) {
    if (xi < LightSourcesCDF[ib]) {
      break;
    }
  }
  // Creator faces mother element
  phot->curel = LightSourcesMother[ib];
  // Creator face
  ib = LightSources[ib];
  if (-1 - HN(phot->curel, 0) == ib)
    phot->curface = 0;
  else if (-1 - HN(phot->curel, 1) == ib)
    phot->curface = 1;
  else if (-1 - HN(phot->curel, 2) == ib)
    phot->curface = 2;
  else if (-1 - HN(phot->curel, 3) == ib)
    phot->curface = 3;
  else
    phot->curface = -1;

  // Initial photon position uniformly distributed on the boundary element
  double w0 = ValoMC::util::rand_open<curandState_t, double>(state);
  double w1 = ValoMC::util::rand_open<curandState_t, double>(state);
  double w2 = ValoMC::util::rand_open<curandState_t, double>(state);
  phot->pos[0] = (w0 * r(BH(ib, 0), 0) + w1 * r(BH(ib, 1), 0) + w2 * r(BH(ib, 2), 0)) / (w0 + w1 + w2);
  phot->pos[1] = (w0 * r(BH(ib, 0), 1) + w1 * r(BH(ib, 1), 1) + w2 * r(BH(ib, 2), 1)) / (w0 + w1 + w2);
  phot->pos[2] = (w0 * r(BH(ib, 0), 2) + w1 * r(BH(ib, 1), 2) + w2 * r(BH(ib, 2), 2)) / (w0 + w1 + w2);

  // Face normal
  normal(ib, n);

  // Make sure that the normal points inside the element by checking dot product of normal & vector (phot->pos) to center of the element
  t[0] = (r(H(phot->curel, 0), 0) + r(H(phot->curel, 1), 0) + r(H(phot->curel, 2), 0) + r(H(phot->curel, 3), 0)) / 4.0 - phot->pos[0];
  t[1] = (r(H(phot->curel, 0), 1) + r(H(phot->curel, 1), 1) + r(H(phot->curel, 2), 1) + r(H(phot->curel, 3), 1)) / 4.0 - phot->pos[1];
  t[2] = (r(H(phot->curel, 0), 2) + r(H(phot->curel, 1), 2) + r(H(phot->curel, 2), 2) + r(H(phot->curel, 3), 2)) / 4.0 - phot->pos[2];
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
  if ((BCLightDirectionType[ib] == 'n'))
  {
    // No BCLNormal data -> Photos main propagation direction is towards inward normal
    //
    // Select photons intial direction based on the boundary condition
    if ((BCType[ib] == 'l'))
    {
      // Laser -- Photons created in normal direction
      phot->dir[0] = n[0];
      phot->dir[1] = n[1];
      phot->dir[2] = n[2];
    }
    else if ((BCType[ib] == 'i'))
    {
      // Isotropic -- Photons initial direction probality density is uniform on a sphere
      // Wolfram Mathworld / Sphere Point Picking
      double dot, r[3], theta, u;
      do
      {
        theta = 2.0 * M_PI * ValoMC::util::rand_open_up<curandState_t, double>(state);
        u = 2.0 * ValoMC::util::rand_closed<curandState_t, double>(state) - 1.0;
        r[0] = sqrt(1.0 - pow(u, 2)) * cos(theta);
        r[1] = sqrt(1.0 - pow(u, 2)) * sin(theta);
        r[2] = u;
        dot = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];
      } while (dot <= 0.0);
      phot->dir[0] = r[0];
      phot->dir[1] = r[1];
      phot->dir[2] = r[2];
    }
    else if ((BCType[ib] == 'c') || (BCType[ib] == 'C'))
    {
      // Cosinic -- Directivity follows cosine pattern
      double phi, theta, dotprodn, dotprod1;
      double f[3], e1[3], e2[3];
      // Two edges of the face
      e1[0] = r(BH(ib, 1), 0) - r(BH(ib, 0), 0);
      e1[1] = r(BH(ib, 1), 1) - r(BH(ib, 0), 1);
      e1[2] = r(BH(ib, 1), 2) - r(BH(ib, 0), 2);
      e2[0] = r(BH(ib, 2), 0) - r(BH(ib, 0), 0);
      e2[1] = r(BH(ib, 2), 1) - r(BH(ib, 0), 1);
      e2[2] = r(BH(ib, 2), 2) - r(BH(ib, 0), 2);
      // Cosinically distributed spherical coordinates
      phi = asin(2.0 * ValoMC::util::rand_open<curandState_t, double>(state) - 1.0);
      theta = 2.0 * M_PI * ValoMC::util::rand_closed<curandState_t, double>(state);
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
    if ((BCType[ib] == 'l') || (BCType[ib] == 'L'))
    {
      // Laser -- Photons created in BCLNormal direction
      phot->dir[0] = BCLNormal(ib, 0);
      phot->dir[1] = BCLNormal(ib, 1);
      phot->dir[2] = BCLNormal(ib, 2);
    }
    else if ((BCType[ib] == 'p'))
    {
      // [AL] Pencil beam
      // In the current implementation, it is always perpendicular to the surface
      // and wastes time by overwrites
      phot->dir[0] = n[0];
      phot->dir[1] = n[1];
      phot->dir[2] = n[2];

      phot->pos[0] = BCLNormal(ib, 0);
      phot->pos[1] = BCLNormal(ib, 1);
      phot->pos[2] = BCLNormal(ib, 2);
      //printf("shooting photon at %18.10lf %18.10lf %18.10lf\n", phot->pos[0],phot->pos[1],phot->pos[2]);
      //printf("to direction %18.10lf %18.10lf %18.10lf\n", phot->dir[0],phot->dir[1],phot->dir[2]);
    }
    else if ((BCType[ib] == 'i') || (BCType[ib] == 'I'))
    {
      // Isotropic -- Photons initial direction probality density is uniform on a sphere
      // Wolfram Mathworld / Sphere Point Picking
      double dot, r[3], theta, u;
      do
      {
        theta = 2.0 * M_PI * ValoMC::util::rand_open_up<curandState_t, double>(state);
        u = 2.0 * ValoMC::util::rand_closed<curandState_t, double>(state) - 1.0;
        r[0] = sqrt(1.0 - pow(u, 2)) * cos(theta);
        r[1] = sqrt(1.0 - pow(u, 2)) * sin(theta);
        r[2] = u;
        dot = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];
      } while (dot <= 0.0);
      phot->dir[0] = r[0];
      phot->dir[1] = r[1];
      phot->dir[2] = r[2];
    }
    else if ((BCType[ib] == 'c') || (BCType[ib] == 'C'))
    {
      // Cosinic -- Directivity follows cosine pattern
      double phi, theta, dotprodn, dotprod1;
      double f[3], e1[3], e2[3];
      // Two edges of the face
      e1[0] = r(BH(ib, 1), 0) - r(BH(ib, 0), 0);
      e1[1] = r(BH(ib, 1), 1) - r(BH(ib, 0), 1);
      e1[2] = r(BH(ib, 1), 2) - r(BH(ib, 0), 2);
      e2[0] = r(BH(ib, 2), 0) - r(BH(ib, 0), 0);
      e2[1] = r(BH(ib, 2), 1) - r(BH(ib, 0), 1);
      e2[2] = r(BH(ib, 2), 2) - r(BH(ib, 0), 2);
      // Cosinically distributed spherical coordinates
      phi = asin(2.0 * ValoMC::util::rand_open<curandState_t, double>(state) - 1.0);
      theta = 2.0 * M_PI * ValoMC::util::rand_closed<curandState_t, double>(state);
      // Propagation direction of generated photon (random draw around x = 1, y = z = 0 direction with cosinic direction distribution)
      f[0] = cos(phi);
      f[1] = cos(theta) * sin(phi);
      f[2] = sin(theta) * sin(phi);
      // Perform coordinate transformation such that the mean direction [1, 0, 0] is mapped
      // to direction of BCLNormal

      // Form tangential vectors for BCLNormal, based on vectors e1, e2 by performing
      // Gram-Schmidt orthogonalization on e1 & e2
      dotprodn = BCLNormal(ib, 0) * e1[0] + BCLNormal(ib, 1) * e1[1] + BCLNormal(ib, 2) * e1[2];
      e1[0] -= dotprodn * BCLNormal(ib, 0);
      e1[1] -= dotprodn * BCLNormal(ib, 1);
      e1[2] -= dotprodn * BCLNormal(ib, 2);
      norm = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
      e1[0] /= norm;
      e1[1] /= norm;
      e1[2] /= norm;

      dotprodn = BCLNormal(ib, 0) * e2[0] + BCLNormal(ib, 1) * e2[1] + BCLNormal(ib, 2) * e2[2];
      dotprod1 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
      e2[0] -= dotprodn * BCLNormal(ib, 0) + dotprod1 * e1[0];
      e2[1] -= dotprodn * BCLNormal(ib, 1) + dotprod1 * e1[1];
      e2[2] -= dotprodn * BCLNormal(ib, 2) + dotprod1 * e1[2];
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

__device__ void MC3DCUDA::scatter_photon (Photon *phot, curandState_t* state)
{
  double xi, theta, phi;
  double dxn, dyn, dzn;

  Array<double> g = *scattering_inhom;
  Array<double> g2 = *scattering_inhom_2;

  // Henye-Greenstein scattering
  if (g[phot->curel] != 0.0)
  {
    xi = ValoMC::util::rand_closed<curandState_t, double>(state);
    if ((0.0 < xi) && (xi < 1.0))
      theta = acos((1.0 + g2[phot->curel] - pow((1.0 - g2[phot->curel]) / (1.0 - g[phot->curel] * (1.0 - 2.0 * xi)), 2)) / (2.0 * g[phot->curel]));
    else
      theta = (1.0 - xi) * M_PI;
  }
  else
    theta = acos(2.0 * ValoMC::util::rand_closed<curandState_t, double>(state) - 1.0);

  phi = 2.0 * M_PI * ValoMC::util::rand_closed<curandState_t, double>(state);

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

  double norm = sqrt(dxn * dxn + dyn * dyn + dzn * dzn);
  dxn /= norm;
  dyn /= norm;
  dzn /= norm;

  phot->dir[0] = dxn;
  phot->dir[1] = dyn;
  phot->dir[2] = dzn;

  // This is to prevent RayTriangleIntersects from misbehaving after scattering event in the PropagatePhoton
  phot->curface = -1;
}

__host__ __device__ void MC3DCUDA::mirror_photon (Photon *phot, int_fast64_t ib)
{
  double n[3], cdot;
  normal(ib, n);
  cdot = n[0] * phot->dir[0] + n[1] * phot->dir[1] + n[2] * phot->dir[2];
  phot->dir[0] -= 2.0 * cdot * n[0];
  phot->dir[1] -= 2.0 * cdot * n[1];
  phot->dir[2] -= 2.0 * cdot * n[2];
}

__host__ __device__ void MC3DCUDA::mirror_photon (Photon *phot, int_fast64_t el, long f)
{
  double n[3], cdot;
  normal(el, f, n);
  cdot = n[0] * phot->dir[0] + n[1] * phot->dir[1] + n[2] * phot->dir[2];
  phot->dir[0] -= 2.0 * cdot * n[0];
  phot->dir[1] -= 2.0 * cdot * n[1];
  phot->dir[2] -= 2.0 * cdot * n[2];
}

__device__ int MC3DCUDA::fresnel_photon (Photon *phot, curandState_t* state)
{

  Array<double> BCn = *BC_n;
  Array<double> n = *idx_refrc;

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
  double nor[3];
  normal((int)phot->curel, (int)phot->nextface, nor);

  double nipnt;
  // Check special case where the photon escapes through the boundary
  if (phot->nextel < 0)
    nipnt = n[phot->curel] / BCn[-1 - phot->nextel];
  else
    nipnt = n[phot->curel] / n[phot->nextel];

  double costhi = -(phot->dir[0] * nor[0] + phot->dir[1] * nor[1] + phot->dir[2] * nor[2]);

  if (1.0 - pow(nipnt, 2) * (1.0 - pow(costhi, 2)) <= 0.0)
  {
    // Total reflection due to critical angle of Snell's law
    phot->dir[0] += 2.0 * costhi * nor[0];
    phot->dir[1] += 2.0 * costhi * nor[1];
    phot->dir[2] += 2.0 * costhi * nor[2];
    phot->curface = phot->nextface;
    return (1);
  }

  double costht = sqrt(1.0 - pow(nipnt, 2) * (1.0 - pow(costhi, 2)));

  double thi;
  if (costhi > 0.0)
    thi = acos(costhi);
  else
    thi = acos(-costhi);
  double tht = acos(costht);
  double R;
  if (!(sin(thi + tht) > eps))
    R = pow((nipnt - 1.0) / (nipnt + 1.0), 2);
  else
    R = 0.5 * (pow(sin(thi - tht) / sin(thi + tht), 2) + pow(tan(thi - tht) / tan(thi + tht), 2));
  double xi = ValoMC::util::rand_closed<curandState_t, double>(state);

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

__device__ void MC3DCUDA::propagate_photon (Photon *phot, curandState_t* state)
{
  double prop, dist, ds;
  int_fast64_t ib;
  // Propagate until the photon dies

  // Array<int_fast64_t> H = *topology;
  Array<int_fast64_t> HN = *neighborhood;
  // Array<int_fast64_t> BH = *boundary;
  //
  // Array<double> r = *grid_nodes;
  // Array<int> LightSources = *light_sources;
  // Array<int> LightSourcesMother = *light_sources_mother;
  // Array<double> LightSourcesCDF = *light_sources_cdf;
  //
  // Array<char> BCLightDirectionType = *BC_light_direction_type;
  Array<char> BCType = *BC_type;
  // Array<double> BCLNormal = *BCL_normal;
  Array<double> BCn = *BC_n;

  Array<double> mua = *absorption;
  Array<double> mus = *scattering;
  // Array<double> g = *scattering_inhom;
  Array<double> n = *idx_refrc;
  Array<double> k = *wave_number;
  // Array<double> g2 = *scattering_inhom_2;
  //
  Array<double> ER = *pow_den_vol_real;
  Array<double> EI = *pow_den_vol_imag;

  Array<double> EBR = *pow_den_boun_real;
  Array<double> EBI = *pow_den_boun_imag;

  while (1)
  {
    // Draw the propagation distance
    prop = -log(ValoMC::util::rand_open<curandState_t, double>(state)) / mus[phot->curel];

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
        if (mua[phot->curel] > 0.0)
        {
          // ER[phot->curel] += (1.0 - exp(-mua[phot->curel] * ds)) * phot->weight;
          atomicAdd(&ER[phot->curel], (1.0 - exp(-mua[phot->curel] * ds)) * phot->weight);
        }
        else
        {
          // ER[phot->curel] += phot->weight * ds;
          atomicAdd(&ER[phot->curel], phot->weight * ds);
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

        // ER[phot->curel] += phot->weight * (cos(phot->phase) - cos(-phot->phase - k[phot->curel] * ds) * exp(-mua[phot->curel] * ds));
        // EI[phot->curel] += phot->weight * (-sin(phot->phase) + sin(phot->phase + k[phot->curel] * ds) * exp(-mua[phot->curel] * ds));
        atomicAdd(&ER[phot->curel], phot->weight * (cos(phot->phase) - cos(-phot->phase - k[phot->curel] * ds) * exp(-mua[phot->curel] * ds)));
        atomicAdd(&EI[phot->curel], phot->weight * (-sin(phot->phase) + sin(phot->phase + k[phot->curel] * ds) * exp(-mua[phot->curel] * ds)));

        phot->phase += k[phot->curel] * ds;
      }

      // Upgrade photon weigh
      phot->weight *= exp(-mua[phot->curel] * ds);

      // Photon has reached a situation where it has to be scattered
      prop -= ds;
      if (prop <= 0.0)
        break;

      // Otherwise the photon will continue to pass through the boundaries of the current element

      // Test for boundary conditions
      if (phot->nextel < 0)
      {
        // Boundary element index
        ib = -1 - phot->nextel;

        if ((BCType[ib] == 'm') || (BCType[ib] == 'L') || (BCType[ib] == 'I') || (BCType[ib] == 'C'))
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
          if (BCn[ib] > 0.0)
            if (fresnel_photon(phot, state))
              continue;

          if (omega <= 0.0)
          {
            // EBR[ib] += phot->weight;
            atomicAdd(&EBR[ib], phot->weight);
          }
          else
          {
            // EBR[ib] += phot->weight * cos(phot->phase);
            // EBI[ib] -= phot->weight * sin(phot->phase);
            atomicAdd(&EBR[ib], phot->weight * cos(phot->phase));
            atomicAdd(&EBI[ib], -phot->weight * sin(phot->phase));

          }
          // Photon propagation will terminate
          return;
        }
      }

      // Test transmission from vacuum -> scattering media
      if ((mus[phot->curel] <= 0.0) && (mus[phot->nextel] > 0.0))
      {
        // Draw new propagation distance -- otherwise photon might travel without scattering
        prop = -log(ValoMC::util::rand_open<curandState_t, double>(state)) / mus[phot->nextel];
      }

      // Test for surival of the photon via roulette
      if (phot->weight < weight0)
      {
        if (ValoMC::util::rand_closed<curandState_t, double>(state) > chance)
          return;
        phot->weight /= chance;
      }

      // Fresnel transmission/reflection
      if (n[phot->curel] != n[phot->nextel])
      {
        if (fresnel_photon(phot, state))
          continue;
      }

      // Upgrade remaining photon propagation lenght in case it is transmitted to different mus domain
      prop *= mus[phot->curel] / mus[phot->nextel];



      // Update current face of the photon to that face which it will be on in the next element
      if (HN(phot->nextel, 0) == phot->curel)
        phot->curface = 0;
      else if (HN(phot->nextel, 1) == phot->curel)
        phot->curface = 1;
      else if (HN(phot->nextel, 2) == phot->curel)
        phot->curface = 2;
      else if (HN(phot->nextel, 3) == phot->curel)
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
    if (mus[phot->curel] > 0.0)
      scatter_photon(phot, state);
  }
}


void MC3DCUDA::allocate () {
  cudaMalloc((void**)&topology, sizeof(Array<int_fast64_t>));
  cudaMalloc((void**)&neighborhood, sizeof(Array<int_fast64_t>));
  cudaMalloc((void**)&boundary, sizeof(Array<int_fast64_t>));

  cudaMalloc((void**)&grid_nodes, sizeof(Array<double>));

  cudaMalloc((void**)&light_sources, sizeof(Array<int>));
  cudaMalloc((void**)&light_sources_mother, sizeof(Array<int>));
  cudaMalloc((void**)&light_sources_cdf, sizeof(Array<double>));

  cudaMalloc((void**)&BC_light_direction_type, sizeof(Array<char>));
  cudaMalloc((void**)&BCL_normal, sizeof(Array<double>));
  cudaMalloc((void**)&BC_n, sizeof(Array<double>));
  cudaMalloc((void**)&BC_type, sizeof(Array<char>));

  cudaMalloc((void**)&absorption, sizeof(Array<double>));
  cudaMalloc((void**)&scattering, sizeof(Array<double>));

  cudaMalloc((void**)&scattering_inhom, sizeof(Array<double>));
  cudaMalloc((void**)&idx_refrc, sizeof(Array<double>));
  cudaMalloc((void**)&wave_number, sizeof(Array<double>));
  cudaMalloc((void**)&scattering_inhom_2, sizeof(Array<double>));

  cudaMalloc((void**)&pow_den_vol_real, sizeof(Array<double>));
  cudaMalloc((void**)&pow_den_vol_imag, sizeof(Array<double>));

  cudaMalloc((void**)&pow_den_boun_real, sizeof(Array<double>));
  cudaMalloc((void**)&pow_den_boun_imag, sizeof(Array<double>));

  // allocate curand states
  cudaMalloc((void**)&states, sizeof(curandState_t)*states_size);
}

void MC3DCUDA::deallocate () {
  cudaFree(topology);
  cudaFree(neighborhood);
  cudaFree(boundary);

  cudaFree(grid_nodes);

  cudaFree(light_sources);
  cudaFree(light_sources_mother);
  cudaFree(light_sources_cdf);

  cudaFree(BC_light_direction_type);
  cudaFree(BCL_normal);
  cudaFree(BC_n);
  cudaFree(BC_type);

  cudaFree(absorption);
  cudaFree(scattering);

  cudaFree(scattering_inhom);
  cudaFree(idx_refrc);
  cudaFree(wave_number);
  cudaFree(scattering_inhom_2);

  cudaFree(pow_den_vol_real);
  cudaFree(pow_den_vol_imag);

  cudaFree(pow_den_boun_real);
  cudaFree(pow_den_boun_imag);

  // allocate curand states
  cudaFree(states);
}

void MC3DCUDA::h2d () {
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


void MC3DCUDA::d2h () {
  ValoMC::util::d2h(&mc3d.ER, pow_den_vol_real);
  ValoMC::util::d2h(&mc3d.EI, pow_den_vol_imag);

  ValoMC::util::d2h(&mc3d.EBR, pow_den_boun_real);
  ValoMC::util::d2h(&mc3d.EBI, pow_den_boun_imag);
}


unsigned long MC3DCUDA::get_total_memory_usage () {
  unsigned long total_memory_usage = 0;
  total_memory_usage += sizeof(int_fast64_t) * mc3d.H.N;
  total_memory_usage += sizeof(int_fast64_t) * mc3d.HN.N;
  total_memory_usage += sizeof(int_fast64_t) * mc3d.BH.N;

  total_memory_usage += sizeof(double) * mc3d.r.N;

  total_memory_usage += sizeof(int) * mc3d.LightSources.N;
  total_memory_usage += sizeof(int) * mc3d.LightSourcesMother.N;
  total_memory_usage += sizeof(double) * mc3d.LightSourcesCDF.N;

  total_memory_usage += sizeof(char) * mc3d.BCLightDirectionType.N;
  total_memory_usage += sizeof(double) * mc3d.BCLNormal.N;
  total_memory_usage += sizeof(double) * mc3d.BCn.N;
  total_memory_usage += sizeof(char) * mc3d.BCType.N;

  total_memory_usage += sizeof(double) * mc3d.mua.N;
  total_memory_usage += sizeof(double) * mc3d.mus.N;
  total_memory_usage += sizeof(double) * mc3d.g.N;
  total_memory_usage += sizeof(double) * mc3d.n.N;
  total_memory_usage += sizeof(double) * mc3d.k.N;
  total_memory_usage += sizeof(double) * mc3d.g2.N;

  total_memory_usage += sizeof(double) * mc3d.ER.N;
  total_memory_usage += sizeof(double) * mc3d.EI.N;

  total_memory_usage += sizeof(double) * mc3d.EBR.N;
  total_memory_usage += sizeof(double) * mc3d.EBI.N;

  total_memory_usage += sizeof(curandState_t) * states_size;

  return total_memory_usage;
}

}
