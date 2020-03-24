#ifndef __MC3D_HPP__
#define __MC3D_HPP__

#define _USE_MATH_DEFINES

#include <chrono>

#include <cmath>
// define USE_OMP prior to including MC3D.hpp to utilize OpenMP
// define USE_MPI prior to including MC3D.hpp to utilize MPI
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <inttypes.h>
#include <array>
#include <vector>
#include "Array.hpp"
#include "../Errors.hpp"
#include "mt_rng.hpp"

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include "mpi.h"
#include "ArrayMPI.hpp"
#endif

#include <emmintrin.h>

#ifndef INT_FAST64_MAX
#define INT_FAST64_MAX __INT_FAST64_MAX__
#endif

// #define USE_NEW_WHICH_FACE

// [DS] defining min and max as macros makes it hard to #include this file
// down the line
// #define min(a, b) ((a) < (b) ? (a) : (b))
// #define max(a, b) ((a) > (b) ? (a) : (b))
using duration = std::chrono::duration<double, std::ratio<1>>;

inline std::chrono::time_point<std::chrono::high_resolution_clock> now () {
  return std::chrono::high_resolution_clock::now();
}


template<typename T>
T max (T a, T b) {
  return a > b ? a : b;
}

template<typename T>
T min (T a, T b) {
  return a < b ? a : b;
}

template<typename T=double>
struct limit_map {
  static constexpr T eps = std::numeric_limits<T>::epsilon();
};

// Check if ray and triangle intersect
template<typename T=double>
int RayTriangleIntersects(T O[3], T D[3], T V0[3], T V1[3], T V2[3], T *t);

inline int RayTriangleIntersects_alt(
  const float O[3], const float D[3],
  const float V0[3], const float V1[3],
  const float V2[3], float *t);

// Structure to hold information of a photon-packet
template<typename T=double>
struct Photon
{
  T pos[3], dir[3];
  int_fast64_t curel, nextel, curface, nextface;
  T weight;
  T phase;
};

// Class for 3D Optical Monte Carlo
template<typename T=double>
class MC3D
{
public:
  MC3D();

  // Nothing need to be done, Arrays will kill themselves when it's time
  ~MC3D() {}
  MC3D &operator=(const MC3D &ref);

  // Random number generation related
  void InitRand();
  T UnifClosed();   // [0, 1]
  T UnifOpen();     // ]0 ,1[
  T UnifHalfDown(); // ]0, 1]
  T UnifHalfUp();   // [0, 1[

  // Volume of element el
  T ElementVolume(int_fast64_t el);
  // Area of boundary element ib, or volumetric element el face f
  T ElementArea(int_fast64_t ib);
  T ElementArea(int_fast64_t el, long f);

  // Normal of boundary element ib -- inward
  void Normal(int_fast64_t ib, T *normal);
  // Normal of face f on element el
  void Normal(int_fast64_t el, long f, T *normal);

  // Initializes the MC3D after all the problem definition parameters have been given
  // Ie. constructs missing parameters
  void Init();

  // Build neighbourhood for volumetric elements
  void BuildNeighbourhoods();

  // Build light sources based on boundary conditions
  void BuildLightSource();

  // Given position, direction, current element & current face photon is on,
  // will return nonzero if the photon will hit a face in element, and also the distance to the point of intersection
  int WhichFace(Photon<T> *phot, T *dist);
  int WhichFace_alt(Photon<T> *phot, T *dist);

  // Create, scatter, mirror & propagate a photon
  void CreatePhoton(Photon<T> *phot);
  void ScatterPhoton(Photon<T> *phot);
  void MirrorPhoton(Photon<T> *phot, int_fast64_t ib);
  void MirrorPhoton(Photon<T> *phot, int_fast64_t el, long f);
  int FresnelPhoton(Photon<T> *phot);
  void PropagatePhoton(Photon<T> *phot, bool use_alt=false);
  int PropagatePhoton_SingleStep_alt(
    Photon<T>* phot,
    T* prop,
    T* dist,
    T* ds,
    int_fast64_t* ib
  );

  int PropagatePhoton_SingleStep(
    Photon<T>* phot,
    T* prop,
    T* dist,
    T* ds,
    int_fast64_t* ib
  );

  void PropagatePhoton_SingleStep_UnmodulatedLight_alt(
    Photon<T>* phot,
    T* ds);

  void PropagatePhoton_SingleStep_UnmodulatedLight(
    Photon<T>* phot,
    T* ds);

  void PropagatePhoton_SingleStep_ModulatedLight_alt (
    Photon<T>* phot,
    T* ds);

  void PropagatePhoton_SingleStep_ModulatedLight (
    Photon<T>* phot,
    T* ds);

  int PropagatePhoton_SingleStep_TestBoundary(
    Photon<T>* phot,
    int_fast64_t* ib);

  // Perform MonteCarlo computation
  void MonteCarlo(bool (*progress)(T) = NULL, void (*finalchecks)(int, int) = NULL, bool use_alt=false);
  // [AL] Check if the arrays seem valid
  void ErrorChecks();

  // [AL] Function to help construct the neighborhood matrix
  // [DS] Moving inline specifier to implementation, see https://isocpp.org/wiki/faq/inline-functions#where-to-put-inline-keyword
  void search_neighbor(std::vector<int_fast64_t> &neighborlist, int_fast64_t element);

public:

  void compute_Vn ();

  std::vector<T> Vn;

  // Geometry
  Array<int_fast64_t> H, HN, BH; // Topology, Neigbourhood, Boundary
  Array<T> r;               // Grid nodes

  // Material parameters for each Element
  Array<T> mua, mus, g, n; // Absorption, Scattering & Scattering inhomogeneity, Index of refraction
  Array<T> k, g2;          // Wave number = omega / c * n, square of g

  // Boundary definitions for each Boundary triangle
  Array<char> BCType;
  Array<T> BCLNormal;
  Array<T> BCn;
  Array<T> BCIntensity; // [AL] Sets the intensity of the light source

  // BCType   =   a  -  Absorbing boundary condition
  //              m  -  Mirror boundary condition
  //              l  -  Laser light source boundary condition
  //              L  -  Mirror with Laser light source boundary condition
  //              i  -  Isotropic light source
  //              I  -  Mirror with Isotropic light source
  //              c  -  Cosinic light source
  //              C  -  Mirror with Cosinic light source
  // BCLNormal = Main direction the light propagates towards
  // BCn = Index of refraction on the outside of the boundary

  Array<char> BCLightDirectionType; // [AL]

  // BCLightDirectionType =   n - No direction provided, use surface normal
  //                          a - Absolute
  //                          r - Relative to surface

  // Frequency and angular frequency of amplitude modulation of the light source
  T f, omega, phase0; //[AL] phase0

  // Number of photons to compute
  int_fast64_t Nphoton;

  // Speed of light (mm / ps)
  T c0;

  // Calculatable parameters
  Array<T> ER, EI;     // Absorbed power density in the volumetric elements (real & imaginary)
  Array<T> EBR, EBI;   // Absorbed power density in the boundary elements (real & imaginary)
  Array<T> DEBR, DEBI; // Absorbed power density in the boundary elements (real & imaginary) weighted by the dot product
                            // of the direction of the photon packets and the boundary normal

  // Light source likelyhood & creation variables
  Array<int> LightSources;
  Array<int> LightSourcesMother;
  Array<T> LightSourcesCDF;

  // Model parameters
  T weight0, chance; // Weight when to commence the roulette & the chance of revitalizing the photon
  long loss;              // Number of photons lost

  // Thread-safe random number generator
  mt_rng rng;

  // Seed id
  unsigned long seed;

  // OpeMP & MPI related variables
  int threadcount;
  int rank, nodecount;
  int totalthreads;
};

// Constuctor, set some default values for Monte Carlo
template<typename T>
MC3D<T>::MC3D()
{
  c0 = 2.99792458e11;

  Nphoton = 1;
  f = omega = 0.0;
  weight0 = 0.001;
  chance = 0.1;
  loss = 0;

  seed = 5489UL;

  rank = 0;
  threadcount = nodecount = totalthreads = 1;
}



// Assingment operator:
//  This will copy references to geometrical and parametric variables
//  Only new variables in the left hand side will be ER/EI, EBR/EBI, VER/VEI
template<typename T>
MC3D<T>& MC3D<T>::operator=(const MC3D &ref)
{
  if (this != &ref)
  {

    H = ref.H;
    HN = ref.HN;
    BH = ref.BH;
    r = ref.r;
    mua = ref.mua;
    mus = ref.mus;
    g = ref.g;
    n = ref.n;
    k = ref.k;
    g2 = ref.g2;
    BCType = ref.BCType;
    BCLNormal = ref.BCLNormal;
    BCIntensity = ref.BCIntensity;

    BCLightDirectionType = ref.BCLightDirectionType;
    BCn = ref.BCn;
    f = ref.f;
    omega = ref.omega;
    phase0 = ref.phase0; // [AL]
    Nphoton = ref.Nphoton;
    c0 = ref.c0;

    ER.resize(ref.ER.N);
    EI.resize(ref.EI.N);
    EBR.resize(ref.EBR.N);
    EBI.resize(ref.EBI.N);
    long ii;

    for (ii = 0; ii < ER.N; ii++)
      ER[ii] = EI[ii] = 0.0;
    for (ii = 0; ii < EBR.N; ii++)
      EBR[ii] = EBI[ii] = 0.0;

    DEBR.resize(ref.DEBR.N); // [AL]
    DEBI.resize(ref.DEBI.N); // [AL]


    for (ii = 0; ii < DEBR.N; ii++) // [AL]
      DEBR[ii] = DEBI[ii] = 0.0;    // [AL]


    // Initialize BCIntensity to one if not given
    if (!BCIntensity.N)
    {
      BCIntensity.resize(BCType.N);
      int ii;
      for (ii = 0; ii < BCIntensity.N; ii++)
        BCIntensity[ii] = 1.0;
    }


    LightSources = ref.LightSources;
    LightSourcesMother = ref.LightSourcesMother;
    LightSourcesCDF = ref.LightSourcesCDF;
    weight0 = ref.weight0;
    chance = ref.chance;
    loss = ref.loss;

    threadcount = ref.threadcount;
    rank = ref.rank;
    nodecount = ref.nodecount;
    totalthreads = ref.totalthreads;

    seed = ref.seed;
    InitRand();

    Vn = ref.Vn;
  }

  return (*this);
}

// Initialize random number generator
template<typename T>
void MC3D<T>::InitRand()
{
  rng.Seed(seed);
}

// Draw random number on [0, 1]
template<typename T>
T MC3D<T>::UnifClosed()
{
  return static_cast<T>(rng.drand_closed());
}

// Draw random number on ]0, 1[
template<typename T>
T MC3D<T>::UnifOpen()
{
  return static_cast<T>(rng.drand_open());
}

// Draw random number ]0, 1]
template<typename T>
T MC3D<T>::UnifHalfDown()
{
  return static_cast<T>(rng.drand_open_down());
}

// Draw random number [0, 1[
template<typename T>
T MC3D<T>::UnifHalfUp()
{
  return static_cast<T>(rng.drand_open_up());
}

// Volume of element el
template<typename T>
T MC3D<T>::ElementVolume(int_fast64_t el)
{
  T ax, ay, az;
  T bx, by, bz;
  T cx, cy, cz;
  T dx, dy, dz;
  T vol;

  ax = r(H(el, 0), 0);
  ay = r(H(el, 0), 1);
  az = r(H(el, 0), 2);
  bx = r(H(el, 1), 0);
  by = r(H(el, 1), 1);
  bz = r(H(el, 1), 2);
  cx = r(H(el, 2), 0);
  cy = r(H(el, 2), 1);
  cz = r(H(el, 2), 2);
  dx = r(H(el, 3), 0);
  dy = r(H(el, 3), 1);
  dz = r(H(el, 3), 2);

  vol = (ax - dx) * ((by - dy) * (cz - dz) - (bz - dz) * (cy - dy)) - (ay - dy) * ((bx - dx) * (cz - dz) - (bz - dz) * (cx - dx)) + (az - dz) * ((bx - dx) * (cy - dy) - (by - dy) * (cx - dx));

  vol = fabs(vol) / 6.0;

  return (vol);
}

// Area of boundary element ib
template<typename T>
T MC3D<T>::ElementArea(int_fast64_t ib)
{
  T a, b, c, area;
  a = sqrt(pow(r(BH(ib, 1), 0) - r(BH(ib, 0), 0), 2) + pow(r(BH(ib, 1), 1) - r(BH(ib, 0), 1), 2) + pow(r(BH(ib, 1), 2) - r(BH(ib, 0), 2), 2));
  b = sqrt(pow(r(BH(ib, 2), 0) - r(BH(ib, 0), 0), 2) + pow(r(BH(ib, 2), 1) - r(BH(ib, 0), 1), 2) + pow(r(BH(ib, 2), 2) - r(BH(ib, 0), 2), 2));
  c = sqrt(pow(r(BH(ib, 1), 0) - r(BH(ib, 2), 0), 2) + pow(r(BH(ib, 1), 1) - r(BH(ib, 2), 1), 2) + pow(r(BH(ib, 1), 2) - r(BH(ib, 2), 2), 2));
  area = sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4.0;
  return (area);
}

// Area of face f of element el
template<typename T>
T MC3D<T>::ElementArea(int_fast64_t el, long f)
{
  T a, b, c, area;
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
    return (0.0);

  a = sqrt(pow(r(H(el, i1), 0) - r(H(el, i0), 0), 2) + pow(r(H(el, i1), 1) - r(H(el, i0), 1), 2) + pow(r(H(el, i1), 2) - r(H(el, i0), 2), 2));
  b = sqrt(pow(r(H(el, i2), 0) - r(H(el, i0), 0), 2) + pow(r(H(el, i2), 1) - r(H(el, i0), 1), 2) + pow(r(H(el, i2), 2) - r(H(el, i0), 2), 2));
  c = sqrt(pow(r(H(el, i1), 0) - r(H(el, i2), 0), 2) + pow(r(H(el, i1), 1) - r(H(el, i2), 1), 2) + pow(r(H(el, i1), 2) - r(H(el, i2), 2), 2));

  area = sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4.0;
  return (area);
}

// Normal of boundary element ib
template<typename T>
void MC3D<T>::Normal(int_fast64_t ib, T *normal)
{
  T x1, y1, z1, x2, y2, z2, nx, ny, nz, norm;

  // Two edges of the face
  x1 = r(BH(ib, 1), 0) - r(BH(ib, 0), 0);
  y1 = r(BH(ib, 1), 1) - r(BH(ib, 0), 1);
  z1 = r(BH(ib, 1), 2) - r(BH(ib, 0), 2);
  x2 = r(BH(ib, 2), 0) - r(BH(ib, 0), 0);
  y2 = r(BH(ib, 2), 1) - r(BH(ib, 0), 1);
  z2 = r(BH(ib, 2), 2) - r(BH(ib, 0), 2);
  // Face normal by a cross product
  nx = y1 * z2 - z1 * y2;
  ny = z1 * x2 - x1 * z2;
  nz = x1 * y2 - y1 * x2;
  norm = sqrt(nx * nx + ny * ny + nz * nz);
  normal[0] = nx / norm;
  normal[1] = ny / norm;
  normal[2] = nz / norm;
}

// Normal of face f on element el
template<typename T>
void MC3D<T>::Normal(int_fast64_t el, long f, T *normal)
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

  T x1, y1, z1, x2, y2, z2, nx, ny, nz, norm;
  // Two edges of the face
  x1 = r(H(el, i1), 0) - r(H(el, i0), 0);
  y1 = r(H(el, i1), 1) - r(H(el, i0), 1);
  z1 = r(H(el, i1), 2) - r(H(el, i0), 2);
  x2 = r(H(el, i2), 0) - r(H(el, i0), 0);
  y2 = r(H(el, i2), 1) - r(H(el, i0), 1);
  z2 = r(H(el, i2), 2) - r(H(el, i0), 2);
  // Face normal by cross product
  nx = y1 * z2 - z1 * y2;
  ny = z1 * x2 - x1 * z2;
  nz = x1 * y2 - y1 * x2;
  norm = sqrt(nx * nx + ny * ny + nz * nz);
  normal[0] = nx / norm;
  normal[1] = ny / norm;
  normal[2] = nz / norm;
}

// Perform errorchecking and throw an error
template<typename T>
void MC3D<T>::ErrorChecks()
{
  /* SANITY CHECKS */
  // Check that
  // row size of H and g are equal
  // row size of H and mus are equal
  // row size of H and mua are equal
  // row size of H and n are equal
  // row size of BH and BCn are equal
  // row size of BH and BCType are equal
  // if given,  row size of BH and BCLNormal are equal
  // row size of BH and BCLightDirectionType are equal
  // H contains an index that cannot be found in r
  // BH contains an index that cannot be found in r
  // std::cerr << "MC3D<T>::ErrorChecks()" << std::endl;
  if (g.Nx != H.Nx)
  {
    throw SIZE_MISMATCH_G;
  }

  // row size of H and mua are equal
  if (mua.Nx != H.Nx)
  {
    throw SIZE_MISMATCH_MUA;
  }

  // row size of H and mus are equal
  if (mus.Nx != H.Nx)
  {
    throw SIZE_MISMATCH_MUS;
  }

  // row size of H and n are equal
  if (n.Nx != H.Nx)
  {
    throw SIZE_MISMATCH_N;
  }

  // Sanity checks for BCn
  if (!BCn.N)
  {
    throw MISSING_BCN;
  }

  //row size of BH and BCn are equal
  if (BCn.Nx != BH.Nx)
  {
    throw SIZE_MISMATCH_BCN;
  }

  //row size of BH and BCType are equal
  if (BCType.Nx != BH.Nx)
  {
    throw SIZE_MISMATCH_BCTYPE;
  }

  // Make sure a light source exists
  bool has_lightsource = false;
  for (int ii = 0; ii < BCType.N; ii++)
  {
    if (BCType[ii] != 'a')
      has_lightsource = true;
  }

  if (!has_lightsource)
    throw NO_LIGHTSOURCE;

  // Sanity checks for BCLNormal
  if (BCLNormal.N)
  {
    if (BCLNormal.Nx != BH.Nx)
    {
      throw SIZE_MISMATCH_LIGHT_DIRECTION;
    }
  }
  if (BCLightDirectionType.N)
  {
    if (BCLightDirectionType.Nx != BH.Nx)
    {
      throw SIZE_MISMATCH_LIGHT_DIRECTION_TYPE;
    }
  }
  // H contains an index that cannot be found in r
  // BH contains an index that cannot be found in r

  for (int ii = 0; ii < H.N; ii++)
  {
    // std::cerr << "(" << r.Nx << ", " << H[ii] << ") ";
    if (H[ii] < 0 || H[ii] >= r.Nx)
    {
      throw INCONSISTENT_H;
    }
  }

  for (int ii = 0; ii < BH.N; ii++)
  {
    if (BH[ii] < 0 || BH[ii] >= r.Nx)
    {
      throw INCONSISTENT_BH;
    }
  }
}

// Initialize Monte Carlo after geometry & material parameters have been assigned
// Under MPI also communicates relevant parameters to other computers and initializes
// mersenne twister with consequetive seed numbers
template<typename T>
void MC3D<T>::Init()
{
#ifdef USE_OMP
  threadcount = omp_get_max_threads();
#endif

#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nodecount);

  MPI_Bcast(&seed, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  seed += rank;

#ifdef USE_OMP
  MPI_Allreduce(&threadcount, &totalthreads, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif



  DistributeArray(H);
  DistributeArray(HN);
  DistributeArray(BH);
  DistributeArray(r);
  DistributeArray(mua);
  DistributeArray(mus);
  DistributeArray(g);
  DistributeArray(n);
  DistributeArray(BCType);
  DistributeArray(BCLNormal);
  DistributeArray(BCn);
  MPI_Bcast(&f, 1, MPI_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Nphoton, 1, MPI_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&weight0, 1, MPI_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&chance, 1, MPI_T, 0, MPI_COMM_WORLD);
  Nphoton /= nodecount;
#endif

  omega = 2.0 * M_PI * f;
  // Build neigborhood topology if required
  BuildNeighbourhoods();

  // Reserve memory for output variables
  int_fast64_t ii;
  ER.resize(H.Nx);
  EI.resize(H.Nx);

  for (ii = 0; ii < H.Nx; ii++)
    ER[ii] = EI[ii] = 0.0;
  EBR.resize(BH.Nx);
  EBI.resize(BH.Nx);

  for (ii = 0; ii < BH.Nx; ii++)
    EBR[ii] = EBI[ii] = 0.0;

  DEBR.resize(BH.Nx);
  DEBI.resize(BH.Nx);

  for (ii = 0; ii < BH.Nx; ii++)
    DEBR[ii] = DEBI[ii] = 0.0;


  // Initialize BCIntensity to one if not given
  if (!BCIntensity.N)
  {
    BCIntensity.resize(BCType.N);
    int ii;
    for (ii = 0; ii < BCIntensity.N; ii++)
      BCIntensity[ii] = 1.0;
  }

  // [AL] Initialize BCLightDirectionType if not given
  if (!BCLightDirectionType.N)
  {
    int ii;
    BCLightDirectionType.resize(BCType.N);
    for (ii = 0; ii < BCLightDirectionType.N; ii++)
    {
      BCLightDirectionType[ii] = 'n';
    }
  }

  // Build lightsources
  BuildLightSource();

  // Compute wavenumber and g squared to speed computations
  k.resize(n.N);
  for (ii = 0; ii < n.N; ii++)
    k[ii] = omega / c0 * n[ii];

  g2.resize(g.N);
  for (ii = 0; ii < g.N; ii++)
    g2[ii] = pow(g[ii], 2);

  InitRand();

  // Normalize BCLNormal if given
  if (BCLNormal.N)
  {
    int ii;
    T norm;
    for (ii = 0; ii < BCLNormal.Nx; ii++)
    {
      norm = sqrt(pow(BCLNormal(ii, 0), 2) + pow(BCLNormal(ii, 1), 2) + pow(BCLNormal(ii, 2), 2));
      // do not normalize coordinates
      if (BCLightDirectionType.Nx == BCLNormal.Nx)
      {
        if (BCLightDirectionType(ii) != 'n')
          continue;
      }
      BCLNormal(ii, 0) /= norm;
      BCLNormal(ii, 1) /= norm;
      BCLNormal(ii, 2) /= norm;
    }
  }

  // [AL] Change BCLNormal coordinates to relative if needed
  T n[3], e1[3], e2[3], norm, dotprodn, dotprod1;

  for (ii = 0; ii < BCLightDirectionType.N; ii++)
  {
    Normal(ii, &n[0]);
    if (BCLightDirectionType[ii] == 'r')
    {
      e1[0] = r(BH(ii, 1), 0) - r(BH(ii, 0), 0);
      e1[1] = r(BH(ii, 1), 1) - r(BH(ii, 0), 1);
      e1[2] = r(BH(ii, 1), 2) - r(BH(ii, 0), 2);
      e2[0] = r(BH(ii, 2), 0) - r(BH(ii, 0), 0);
      e2[1] = r(BH(ii, 2), 1) - r(BH(ii, 0), 1);
      e2[2] = r(BH(ii, 2), 2) - r(BH(ii, 0), 2);

      // Form tangential vectors for BCLNormal, based on vectors e1, e2 by performing
      // Gram-Schmidt orthogonalization on e1 & e2
      dotprodn = n[0] * e1[0] + n[1] * e1[1] + n[2] * e1[2];
      e1[0] -= dotprodn * n[0];
      e1[1] -= dotprodn * n[1];
      e1[2] -= dotprodn * n[2];
      norm = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
      e1[0] /= norm;
      e1[1] /= norm;
      e1[2] /= norm;

      dotprodn = n[0] * e2[0] + n[1] * e2[1] + n[2] * e2[2];
      dotprod1 = e1[0] * e2[0] + e1[1] * e2[1] + e1[2] * e2[2];
      e2[0] -= dotprodn * n[0] + dotprod1 * e1[0];
      e2[1] -= dotprodn * n[1] + dotprod1 * e1[1];
      e2[2] -= dotprodn * n[2] + dotprod1 * e1[2];

      norm = sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
      e2[0] /= norm;
      e2[1] /= norm;
      e2[2] /= norm;

      Normal(ii, &n[0]);
      T old_bclx = BCLNormal(ii, 0);
      T old_bcly = BCLNormal(ii, 1);
      T old_bclz = BCLNormal(ii, 2);
      BCLNormal(ii, 0) = old_bclx * e1[0] + old_bcly * e2[0] + old_bclz * n[0];
      BCLNormal(ii, 1) = old_bclx * e1[1] + old_bcly * e2[1] + old_bclz * n[1];
      BCLNormal(ii, 2) = old_bclx * e1[2] + old_bcly * e2[2] + old_bclz * n[2];
    }
  }

  compute_Vn ();

  return;
}


template<typename T>
void MC3D<T>::compute_Vn ()
{
  // const int_fast64_t H_phot_curel_0 = H(phot_curel, 0);
  // const int_fast64_t H_phot_curel_1 = H(phot_curel, 1);
  // const int_fast64_t H_phot_curel_2 = H(phot_curel, 2);
  // const int_fast64_t H_phot_curel_3 = H(phot_curel, 3);
  //
  // T V0[3] = {r(H_phot_curel_0, 0), r(H_phot_curel_0, 1), r(H_phot_curel_0, 2)};
  // T V1[3] = {r(H_phot_curel_1, 0), r(H_phot_curel_1, 1), r(H_phot_curel_1, 2)};
  // T V2[3] = {r(H_phot_curel_2, 0), r(H_phot_curel_2, 1), r(H_phot_curel_2, 2)};
  // T V3[3] = {r(H_phot_curel_3, 0), r(H_phot_curel_3, 1), r(H_phot_curel_3, 2)};

  // std::cerr << "MC3D::compute_Vn" << std::endl;
  Vn.resize(12*H.Nx);

  #if USE_OMP
  #pragma omp parallel for
  #endif
  for (unsigned idx=0; idx<H.Nx; idx++) {
    for (unsigned idy=0; idy<4; idy++) {
      int_fast64_t H_curel = H(idx, idy);
      for (unsigned idz=0; idz<3; idz++) {
        // Vn[12*idx + 3*idy][idz] = r(H_curel, idz);
        Vn[12*idx + 3*idy + idz] = r(H_curel, idz);
      }
    }
  }
  // std::cerr << "MC3D::compute_Vn done" << std::endl;

}

template<typename T>
void MC3D<T>::search_neighbor(std::vector<int_fast64_t> &neighborlist, int_fast64_t element)
{
  for (unsigned int i = 0; i < neighborlist.size(); i++)
  {
    int_fast64_t neighbor = neighborlist[i];
    // Test for Face 0, 1, 2, 3
    if (neighbor != element)
    {
      if (neighbor >= 0)
      {
      // ispart

#define ISIN3(a, b) (H(element, a) == H(neighbor, b))
#define ISPART3(a) (ISIN3(a, 0) || ISIN3(a, 1) || ISIN3(a, 2) || ISIN3(a, 3))
        if (ISPART3(0) && ISPART3(1) && ISPART3(2))
        {
          // Another, different, entry is already written to the neighborhood matrix.
          // Something must be wrong with the mesh.
          if (HN(element, 0) != INT_FAST64_MAX && HN(element, 0) != neighbor)
          {
            throw INCONSISTENT_MESH_DUPLICATE_NEIGHBORS;
          }
          HN(element, 0) = neighbor;
        }
        if (ISPART3(0) && ISPART3(1) && ISPART3(3))
        {
          if (HN(element, 1) != INT_FAST64_MAX && HN(element, 1) != neighbor)
          {
            std::cout << "HN : " << HN(element, 1) << " \n";
            throw INCONSISTENT_MESH_DUPLICATE_NEIGHBORS;
          }
          HN(element, 1) = neighbor;
        }
        if (ISPART3(0) && ISPART3(2) && ISPART3(3))
        {
          if (HN(element, 2) != INT_FAST64_MAX && HN(element, 2) != neighbor)
          {
            std::cout << "HN : " << HN(element, 2) << " \n";
            throw INCONSISTENT_MESH_DUPLICATE_NEIGHBORS;
          }
          HN(element, 2) = neighbor;
        }
        if (ISPART3(1) && ISPART3(2) && ISPART3(3))
        {
          if (HN(element, 3) != INT_FAST64_MAX && HN(element, 3) != neighbor)
          {
            std::cout << "HN : " << HN(element, 3) << " \n";
            throw INCONSISTENT_MESH_DUPLICATE_NEIGHBORS;
          }
          HN(element, 3) = neighbor;
        }
      }
      else
      {
      // the neighbor is a boundary element
#define ISIN4(a, b) (H(element, a) == BH(-(neighbor + 1), b))
#define ISPART4(a) (ISIN4(a, 0) || ISIN4(a, 1) || ISIN4(a, 2))
        if (ISPART4(0) && ISPART4(1) && ISPART4(2))
        {
          if (HN(element, 0) != INT_FAST64_MAX && HN(element, 0) != neighbor)
          {
            throw INCONSISTENT_MESH_DUPLICATE_NEIGHBORS;
          }
          HN(element, 0) = neighbor;
        }
        if (ISPART4(0) && ISPART4(1) && ISPART4(3))
        {
          if (HN(element, 1) != INT_FAST64_MAX && HN(element, 1) != neighbor)
          {
            throw INCONSISTENT_MESH_DUPLICATE_NEIGHBORS;
          }
          HN(element, 1) = neighbor;
        }
        if (ISPART4(0) && ISPART4(2) && ISPART4(3))
        {
          if (HN(element, 2) != INT_FAST64_MAX && HN(element, 2) != neighbor)
          {
            throw INCONSISTENT_MESH_DUPLICATE_NEIGHBORS;
          }
          HN(element, 2) = neighbor;
        }
        if (ISPART4(1) && ISPART4(2) && ISPART4(3))
        {
          if (HN(element, 3) != INT_FAST64_MAX && HN(element, 3) != neighbor)
          {
            throw INCONSISTENT_MESH_DUPLICATE_NEIGHBORS;
          }
          HN(element, 3) = neighbor;
        }
      }
    }
  }
}

// Build neigbourhood HN for volumetric topology H
template<typename T>
void MC3D<T>::BuildNeighbourhoods()
{
  // std::cerr << "MC3D<T>::BuildNeighbourhoods" << std::endl;
#define NEW_NHOOD
#ifdef NEW_NHOOD
  if (HN.N != H.N)
  {
    // std::cerr << "MC3D<T>::BuildNeighbourhoods: HN.N != H.N" << std::endl;
    HN.resize(H.Nx, 4);
    std::vector<std::vector<int_fast64_t> > parent;
    parent.resize((int)r.Nx);
    // Build a vector (parent)
    // that contains all tetrahedrons shared by a vertex
    for (int_fast64_t i = 0; i < H.Nx; i++)
    {
      parent[(int)H(i, 0)].push_back(i);
      parent[(int)H(i, 1)].push_back(i);
      parent[(int)H(i, 2)].push_back(i);
      parent[(int)H(i, 3)].push_back(i);
    }
    //
    for (int_fast64_t i = 0; i < BH.Nx; i++)
    {
      parent[(int)BH(i, 0)].push_back(-1 - i);
      parent[(int)BH(i, 1)].push_back(-1 - i);
      parent[(int)BH(i, 2)].push_back(-1 - i);
    }
    int_fast64_t istart = 0, iend = H.Nx;
    for (int_fast64_t ii = istart; ii < iend; ii++)
    {
      HN(ii, 0) = HN(ii, 1) = HN(ii, 2) = HN(ii, 3) = INT_FAST64_MAX;
      // Search neighbors from the tetrahedrons shared by the corner vertices
      search_neighbor(parent[(int)H(ii, 0)], ii);
      search_neighbor(parent[(int)H(ii, 1)], ii);
      search_neighbor(parent[(int)H(ii, 2)], ii);
      search_neighbor(parent[(int)H(ii, 3)], ii);
      // if (istart < 10) {
      //   std::cerr << "[ " <<  HN(ii, 0) << ", " << HN(ii, 1) << ", "
      //     <<  HN(ii, 2) << ", " << HN(ii, 3) << "] ";
      //   std::cerr << "[ " <<  H(ii, 0) << ", " << H(ii, 1) << ", "
      //     <<  H(ii, 2) << ", " << H(ii, 3) << "] " << std::endl;
      // }
      //      printf("%li %li %li %li\n", HN(ii, 0), HN(ii, 1), HN(ii, 2), HN(ii, 3));
    }
  }
#else
  if (HN.N != H.N)
  {
    HN.resize(H.Nx, 4);

    int_fast64_t istart = 0, iend = H.Nx;
    int_fast64_t ii, jj;
#ifdef USE_MPI
    istart = rank * H.Nx / nodecount;
    iend = (rank + 1) * H.Nx / nodecount;
    for (ii = 0; ii < HN.N; ii++)
      HN[ii] = 0;
#endif

#ifdef USE_OMP
#pragma omp parallel for private(ii, jj)
#endif
    for (ii = istart; ii < iend; ii++)
    {
      HN(ii, 0) = HN(ii, 1) = HN(ii, 2) = HN(ii, 3) = -1;
      for (jj = 0; jj < H.Nx; jj++)
      {
        if (ii == jj)
          continue;
#define ISIN(a, b) (H(ii, a) == H(jj, b))
#define ISPART(a) (ISIN(a, 0) || ISIN(a, 1) || ISIN(a, 2) || ISIN(a, 3))
        // Test for Face 0, 1, 2, 3
        if (ISPART(0) && ISPART(1) && ISPART(2))
          HN(ii, 0) = jj;
        if (ISPART(0) && ISPART(1) && ISPART(3))
          HN(ii, 1) = jj;
        if (ISPART(0) && ISPART(2) && ISPART(3))
          HN(ii, 2) = jj;
        if (ISPART(1) && ISPART(2) && ISPART(3))
          HN(ii, 3) = jj;
      }

      // Fill -1's with appropriate boundary element from BH
      if ((HN(ii, 0) == -1) || (HN(ii, 1) == -1) || (HN(ii, 2) == -1) || (HN(ii, 3) == -1))
      {
        for (jj = 0; jj < BH.Nx; jj++)
        {
#define ISIN2(a, b) (H(ii, a) == BH(jj, b))
#define ISPART2(a) (ISIN2(a, 0) || ISIN2(a, 1) || ISIN2(a, 2))
          if ((HN(ii, 0) == -1) && (ISPART2(0) && ISPART2(1) && ISPART2(2)))
            HN(ii, 0) = -1 - jj;
          if ((HN(ii, 1) == -1) && (ISPART2(0) && ISPART2(1) && ISPART2(3)))
            HN(ii, 1) = -1 - jj;
          if ((HN(ii, 2) == -1) && (ISPART2(0) && ISPART2(2) && ISPART2(3)))
            HN(ii, 2) = -1 - jj;
          if ((HN(ii, 3) == -1) && (ISPART2(1) && ISPART2(2) && ISPART2(3)))
            HN(ii, 3) = -1 - jj;
        }
      }
      //printf("%li %li %li %li\n", HN(ii, 0), HN(ii, 1), HN(ii, 2), HN(ii, 3));
    }

#ifdef USE_MPI
    // Distribute neigbourhood to other computers
    AllReduceArray(HN, MPI_SUM);
#endif
  }
#endif
}

// Build light sources based on boundary conditions
//   LightSources will contain index to the boundary element in BH acting as lightsource
//   LightSourcesMother will contain index to volumetric element H for which BH is attached
//   LightSourcesCDF will be a cumulative/normalized sum of areas of all the lightsources, this will ease randomizing the creation of photons
template<typename T>
void MC3D<T>::BuildLightSource()
{
  int_fast64_t ii, jj, kk, ib, NLightSource;

  if (LightSources.N != 0)
    return;

  // Loop over HN and determine the number of lightsources (boundary elements)
  NLightSource = 0;
  for (ii = 0; ii < HN.Nx; ii++)
  {
    for (jj = 0; jj < 4; jj++)
    {
      if (HN(ii, jj) <= -1)
      {
        ib = -1 - HN(ii, jj);
        if ((BCType[ib] == 'l') || (BCType[ib] == 'L') ||
            (BCType[ib] == 'i') || (BCType[ib] == 'I') ||
            (BCType[ib] == 'c') || (BCType[ib] == 'C') || (BCType[ib] == 'p'))
          NLightSource++;
      }
    }
  }

  // Allocate space for light sources
  LightSources.resize(NLightSource);
  LightSourcesMother.resize(NLightSource);
  LightSourcesCDF.resize(NLightSource);

  // Compute area of each boundary element acting as a lightsource & assemble lightsource motherhood
  kk = 0;
  for (ii = 0; ii < HN.Nx; ii++)
  {
    for (jj = 0; jj < 4; jj++)
    {
      if (HN(ii, jj) <= -1)
      {
        ib = -1 - HN(ii, jj);
        if ((BCType[ib] == 'l') || (BCType[ib] == 'L') ||
            (BCType[ib] == 'i') || (BCType[ib] == 'I') ||
            (BCType[ib] == 'c') || (BCType[ib] == 'C') || BCType[ib] == 'p')
        {

          LightSources[kk] = (int)ib;
          LightSourcesMother[kk] = (int)ii;

          LightSourcesCDF[kk] = ElementArea(ib);
          kk++;
        }
      }
    }
  }

  for (ii = 0; ii < NLightSource; ii++)
    LightSourcesCDF[ii] *= BCIntensity[LightSources[ii]]; // [AL]

  // Compute cumsum of LightSourcesCDF and normalize -- Ie. form cumulated distribution function
  for (ii = 1; ii < NLightSource; ii++)
    LightSourcesCDF[ii] += LightSourcesCDF[ii - 1];
  for (ii = 0; ii < NLightSource; ii++)
    LightSourcesCDF[ii] /= LightSourcesCDF[NLightSource - 1];
}

template<typename T>
int MC3D<T>::WhichFace_alt(Photon<T> *phot, T *dist)
{
  const int_fast64_t phot_curel = phot->curel;
  const int_fast64_t phot_curface = phot->curface;

  T* pos = phot->pos;
  T* dir = phot->dir;

  T* Vn_ptr = Vn.data() + 12*phot_curel;

  // for (unsigned idx=0; idx<3; idx++) {
  //   Vn_curel[0][idx] = Vn_ptr[0 + idx];
  //   Vn_curel[1][idx] = Vn_ptr[3 + idx];
  //   Vn_curel[2][idx] = Vn_ptr[6 + idx];
  //   Vn_curel[3][idx] = Vn_ptr[9 + idx];
  // }

  // Vn_curel[0] = {Vn_ptr[0], Vn_ptr[1], Vn_ptr[2]};
  // Vn_curel[1] = {Vn_ptr[3], Vn_ptr[4], Vn_ptr[5]};
  // Vn_curel[2] = {Vn_ptr[6], Vn_ptr[7], Vn_ptr[8]};
  // Vn_curel[3] = {Vn_ptr[9], Vn_ptr[10], Vn_ptr[11]};

  T V0[3] = {Vn_ptr[0], Vn_ptr[1], Vn_ptr[2]};
  T V1[3] = {Vn_ptr[3], Vn_ptr[4], Vn_ptr[5]};
  T V2[3] = {Vn_ptr[6], Vn_ptr[7], Vn_ptr[8]};
  T V3[3] = {Vn_ptr[9], Vn_ptr[10], Vn_ptr[11]};

  // T* V0 = Vn_ptr;
  // T* V1 = Vn_ptr + 3;
  // T* V2 = Vn_ptr + 6;
  // T* V3 = Vn_ptr + 9;

  if (phot_curface != 0) {
    if (RayTriangleIntersects(pos, dir, V0, V1, V2, dist)) {
      if (*dist > 0.0) {
        phot->nextface = 0;
        phot->nextel = HN(phot_curel, phot->nextface);
        return 0;
      }
    }
  }

  if (phot_curface != 1) {
    if (RayTriangleIntersects(pos, dir, V0, V1, V3, dist)) {
      if (*dist > 0.0) {
        phot->nextface = 1;
        phot->nextel = HN(phot_curel, phot->nextface);
        return 1;
      }
    }
  }

  if (phot_curface != 2) {
    if (RayTriangleIntersects(pos, dir, V0, V2, V3, dist)) {
      if (*dist > 0.0) {
        phot->nextface = 2;
        phot->nextel = HN(phot_curel, phot->nextface);
        return 2;
      }
    }
  }

  if (phot_curface != 3) {
    if (RayTriangleIntersects(pos, dir, V1, V2, V3, dist)) {
      if (*dist > 0.0) {
        phot->nextface = 3;
        phot->nextel = HN(phot_curel, phot->nextface);
        return 3;
      }
    }
  }


    // int_fast64_t H_Nx = H.Nx;
    //
    // int_fast64_t* H_phot_curel_0 = H_data + phot_curel;
    // int_fast64_t* H_phot_curel_1 = H_phot_curel_0 + H_Nx;
    // int_fast64_t* H_phot_curel_2 = H_phot_curel_1 + H_Nx;
    // int_fast64_t* H_phot_curel_3 = H_phot_curel_2 + H_Nx;
    //
    // T V0[3] = {r(*H_phot_curel_0, 0), r(*H_phot_curel_0, 1), r(*H_phot_curel_0, 2)};
    // T V1[3] = {r(*H_phot_curel_1, 0), r(*H_phot_curel_1, 1), r(*H_phot_curel_1, 2)};
    // T V2[3] = {r(*H_phot_curel_2, 0), r(*H_phot_curel_2, 1), r(*H_phot_curel_2, 2)};
    // T V3[3] = {r(*H_phot_curel_3, 0), r(*H_phot_curel_3, 1), r(*H_phot_curel_3, 2)};


    // std::vector<T*> faces = {
    //   V0, V1, V2,
    //   V0, V1, V3,
    //   V0, V2, V3,
    //   V1, V2, V3
    // };

    // for (int_fast64_t idx=0; idx<4; idx++) {
    //   if (phot->curface != idx) {
    //     // t0 = now();
    //     int res = RayTriangleIntersects<T>(
    //       phot->pos, phot->dir,
    //       faces[3*idx], faces[3*idx + 1], faces[3*idx + 2], dist);
    //     // WhichFace_ray_intersects += (now() - t0);
    //     if (res) {
    //       if (*dist > 0.0) {
    //         // t0 = now();
    //         phot->nextface = idx;
    //         phot->nextel = HN(phot_curel, phot->nextface);
    //         // WhichFace_phot_prop += (now() - t0);
    //         return idx;
    //       }
    //     }
    //   }
    // }



    // int_fast64_t* H_phot_curel = H.data + phot_curel;
    // int_fast64_t H_Nx = H.Nx;
    //
    // T* r_data;
    // int_fast64_t r_Nx = r.Nx;
    //
    // // std::array<T[3], 4> Vn;
    //
    // for (unsigned idx=0; idx<4; idx++) {
    //   r_data = r.data + *H_phot_curel;
    //   // std::cerr << "idx=" << idx << std::endl;
    //   for (unsigned idy=0; idy<3; idy++) {
    //     Vn[idx][idy] = *r_data;
    //     r_data += r_Nx;
    //     // std::cerr << "idy=" << idy << std::endl;
    //   }
    //   H_phot_curel += H_Nx;
    // }

    // if (phot->curface != 0) {
    //   if (RayTriangleIntersects<T>(phot->pos, phot->dir, Vn[0], Vn[1], Vn[2], dist)) {
    //     if (*dist > 0.0) {
    //       phot->nextface = 0;
    //       phot->nextel = HN(phot_curel, phot->nextface);
    //       return 0;
    //     }
    //   }
    // }
    //
    // if (phot->curface != 1) {
    //   if (RayTriangleIntersects<T>(phot->pos, phot->dir, Vn[0], Vn[1], Vn[3], dist)) {
    //     if (*dist > 0.0) {
    //       phot->nextface = 1;
    //       phot->nextel = HN(phot_curel, phot->nextface);
    //       return 1;
    //     }
    //   }
    // }
    //
    // if (phot->curface != 2) {
    //   if (RayTriangleIntersects<T>(phot->pos, phot->dir, Vn[0], Vn[2], Vn[3], dist)) {
    //     if (*dist > 0.0) {
    //       phot->nextface = 2;
    //       phot->nextel = HN(phot_curel, phot->nextface);
    //       return 2;
    //     }
    //   }
    // }
    //
    // if (phot->curface != 3) {
    //   if (RayTriangleIntersects<T>(phot->pos, phot->dir, Vn[1], Vn[2], Vn[3], dist)) {
    //     if (*dist > 0.0) {
    //       phot->nextface = 3;
    //       phot->nextel = HN(phot_curel, phot->nextface);
    //       return 3;
    //     }
    //   }
    // }
  return -1;
}


// Determine which face a photon will exit a volumetric element from
//int MC3D<T>::WhichFace(T curpos[3], T dir[3], int el, int face, T *dist){
template<typename T>
int MC3D<T>::WhichFace(Photon<T> *phot, T *dist)
{

  // duration WhichFace_array_access;
  // duration WhichFace_ray_intersects;
  // duration WhichFace_phot_prop;

  // phot - photon under test
  // dist - distance the photon can travel before hitting the face
  //
  // Return values:
  // -1 if no hit
  // 0, 1, 2 3 for faces formed by (V0, V1, V2), (V0, V1, V3), (V0, V2, V3), (V1, V2, V3) respectively

  // const int_fast64_t phot_curel = phot->curel;
  //
  // const int_fast64_t H_phot_curel_0 = H(phot_curel, 0);
  // const int_fast64_t H_phot_curel_1 = H(phot_curel, 1);
  // const int_fast64_t H_phot_curel_2 = H(phot_curel, 2);
  // const int_fast64_t H_phot_curel_3 = H(phot_curel, 3);
  //
  // T V0[3] = {r(H_phot_curel_0, 0), r(H_phot_curel_0, 1), r(H_phot_curel_0, 2)};
  // T V1[3] = {r(H_phot_curel_1, 0), r(H_phot_curel_1, 1), r(H_phot_curel_1, 2)};
  // T V2[3] = {r(H_phot_curel_2, 0), r(H_phot_curel_2, 1), r(H_phot_curel_2, 2)};
  // T V3[3] = {r(H_phot_curel_3, 0), r(H_phot_curel_3, 1), r(H_phot_curel_3, 2)};

  T V0[3] = {r(H(phot->curel, 0), 0), r(H(phot->curel, 0), 1), r(H(phot->curel, 0), 2)};
  T V1[3] = {r(H(phot->curel, 1), 0), r(H(phot->curel, 1), 1), r(H(phot->curel, 1), 2)};
  T V2[3] = {r(H(phot->curel, 2), 0), r(H(phot->curel, 2), 1), r(H(phot->curel, 2), 2)};
  T V3[3] = {r(H(phot->curel, 3), 0), r(H(phot->curel, 3), 1), r(H(phot->curel, 3), 2)};

  if (phot->curface != 0) {
    if (RayTriangleIntersects<T>(phot->pos, phot->dir, V0, V1, V2, dist)) {
      if (*dist > 0.0) {
        phot->nextface = 0;
        phot->nextel = HN(phot->curel, phot->nextface);
        return 0;
      }
    }
  }

  if (phot->curface != 1) {
    if (RayTriangleIntersects<T>(phot->pos, phot->dir, V0, V1, V3, dist)) {
      if (*dist > 0.0) {
        phot->nextface = 1;
        phot->nextel = HN(phot->curel, phot->nextface);
        return 1;
      }
    }
  }

  if (phot->curface != 2) {
    if (RayTriangleIntersects<T>(phot->pos, phot->dir, V0, V2, V3, dist)) {
      if (*dist > 0.0) {
        phot->nextface = 2;
        phot->nextel = HN(phot->curel, phot->nextface);
        return 2;
      }
    }
  }

  if (phot->curface != 3) {
    if (RayTriangleIntersects<T>(phot->pos, phot->dir, V1, V2, V3, dist)) {
      if (*dist > 0.0) {
        phot->nextface = 3;
        phot->nextel = HN(phot->curel, phot->nextface);
        return 3;
      }
    }
  }
  return -1;
}

// Create a new photon based on LightSources, LightSourcesMother and LighSourcesCDF
template<typename T>
void MC3D<T>::CreatePhoton(Photon<T> *phot)
{
  T xi = UnifClosed();

  T n[3], t[3], norm;

  // Find boundary element index that will create this photon
  int ib;
  for (ib = 0; ib < LightSources.Nx; ib++)
    if (xi < LightSourcesCDF[ib])
      break;
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
  T w0 = UnifOpen(), w1 = UnifOpen(), w2 = UnifOpen();
  phot->pos[0] = (w0 * r(BH(ib, 0), 0) + w1 * r(BH(ib, 1), 0) + w2 * r(BH(ib, 2), 0)) / (w0 + w1 + w2);
  phot->pos[1] = (w0 * r(BH(ib, 0), 1) + w1 * r(BH(ib, 1), 1) + w2 * r(BH(ib, 2), 1)) / (w0 + w1 + w2);
  phot->pos[2] = (w0 * r(BH(ib, 0), 2) + w1 * r(BH(ib, 1), 2) + w2 * r(BH(ib, 2), 2)) / (w0 + w1 + w2);

  // Face normal
  Normal(ib, n);

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
      T dot, r[3], theta, u;
      do
      {
        theta = 2.0 * M_PI * UnifHalfUp();
        u = 2.0 * UnifClosed() - 1.0;
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
      T phi, theta, dotprodn, dotprod1;
      T f[3], e1[3], e2[3];
      // Two edges of the face
      e1[0] = r(BH(ib, 1), 0) - r(BH(ib, 0), 0);
      e1[1] = r(BH(ib, 1), 1) - r(BH(ib, 0), 1);
      e1[2] = r(BH(ib, 1), 2) - r(BH(ib, 0), 2);
      e2[0] = r(BH(ib, 2), 0) - r(BH(ib, 0), 0);
      e2[1] = r(BH(ib, 2), 1) - r(BH(ib, 0), 1);
      e2[2] = r(BH(ib, 2), 2) - r(BH(ib, 0), 2);
      // Cosinically distributed spherical coordinates
      phi = asin(2.0 * UnifOpen() - 1.0);
      theta = 2.0 * M_PI * UnifClosed();
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
      T dot, r[3], theta, u;
      do
      {
        theta = 2.0 * M_PI * UnifHalfUp();
        u = 2.0 * UnifClosed() - 1.0;
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
      T phi, theta, dotprodn, dotprod1;
      T f[3], e1[3], e2[3];
      // Two edges of the face
      e1[0] = r(BH(ib, 1), 0) - r(BH(ib, 0), 0);
      e1[1] = r(BH(ib, 1), 1) - r(BH(ib, 0), 1);
      e1[2] = r(BH(ib, 1), 2) - r(BH(ib, 0), 2);
      e2[0] = r(BH(ib, 2), 0) - r(BH(ib, 0), 0);
      e2[1] = r(BH(ib, 2), 1) - r(BH(ib, 0), 1);
      e2[2] = r(BH(ib, 2), 2) - r(BH(ib, 0), 2);
      // Cosinically distributed spherical coordinates
      phi = asin(2.0 * UnifOpen() - 1.0);
      theta = 2.0 * M_PI * UnifClosed();
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

// Scatter a photon
template<typename T>
void MC3D<T>::ScatterPhoton(Photon<T> *phot)
{
  T xi, theta, phi;
  T dxn, dyn, dzn;

  // Henye-Greenstein scattering
  if (g[phot->curel] != 0.0)
  {
    xi = UnifClosed();
    if ((0.0 < xi) && (xi < 1.0))
      theta = acos((1.0 + g2[phot->curel] - pow((1.0 - g2[phot->curel]) / (1.0 - g[phot->curel] * (1.0 - 2.0 * xi)), 2)) / (2.0 * g[phot->curel]));
    else
      theta = (1.0 - xi) * M_PI;
  }
  else
    theta = acos(2.0 * UnifClosed() - 1.0);

  phi = 2.0 * M_PI * UnifClosed();

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

  T norm = sqrt(dxn * dxn + dyn * dyn + dzn * dzn);
  dxn /= norm;
  dyn /= norm;
  dzn /= norm;

  phot->dir[0] = dxn;
  phot->dir[1] = dyn;
  phot->dir[2] = dzn;

  // This is to prevent RayTriangleIntersects from misbehaving after scattering event in the PropagatePhoton
  phot->curface = -1;
}

// Mirror photons propagation with respect to boundary element ib
template<typename T>
void MC3D<T>::MirrorPhoton(Photon<T> *phot, int_fast64_t ib)
{
  T n[3], cdot;
  Normal(ib, n);
  cdot = n[0] * phot->dir[0] + n[1] * phot->dir[1] + n[2] * phot->dir[2];
  phot->dir[0] -= 2.0 * cdot * n[0];
  phot->dir[1] -= 2.0 * cdot * n[1];
  phot->dir[2] -= 2.0 * cdot * n[2];
}

// Mirror photon with respect to face f of element el
template<typename T>
void MC3D<T>::MirrorPhoton(Photon<T> *phot, int_fast64_t el, long f)
{
  T n[3], cdot;
  Normal(el, f, n);
  cdot = n[0] * phot->dir[0] + n[1] * phot->dir[1] + n[2] * phot->dir[2];
  phot->dir[0] -= 2.0 * cdot * n[0];
  phot->dir[1] -= 2.0 * cdot * n[1];
  phot->dir[2] -= 2.0 * cdot * n[2];
}

// Fresnel transmission / reflection of a photon
template<typename T>
int MC3D<T>::FresnelPhoton(Photon<T> *phot)
{
  // Likelyhood of reflection:
  //   R = 0.5 ( sin^2(theta_i - theta_t) / sin^2(theta_i + theta_t) + tan^2(theta_i - theta_t) / tan^2(theta_i + theta_t))
  //
  // For theta_i + theta_t < limit_map<T>::eps:
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

  // Normal of the tranmitting face
  T nor[3];
  Normal((int)phot->curel, (int)phot->nextface, nor);

  T nipnt;
  // Check special case where the photon escapes through the boundary
  if (phot->nextel < 0)
    nipnt = n[phot->curel] / BCn[-1 - phot->nextel];
  else
    nipnt = n[phot->curel] / n[phot->nextel];

  T costhi = -(phot->dir[0] * nor[0] + phot->dir[1] * nor[1] + phot->dir[2] * nor[2]);

  if (1.0 - pow(nipnt, 2) * (1.0 - pow(costhi, 2)) <= 0.0)
  {
    // Total reflection due to critical angle of Snell's law
    phot->dir[0] += 2.0 * costhi * nor[0];
    phot->dir[1] += 2.0 * costhi * nor[1];
    phot->dir[2] += 2.0 * costhi * nor[2];
    phot->curface = phot->nextface;
    return (1);
  }

  T costht = sqrt(1.0 - pow(nipnt, 2) * (1.0 - pow(costhi, 2)));

  T thi;
  if (costhi > 0.0)
    thi = acos(costhi);
  else
    thi = acos(-costhi);
  T tht = acos(costht);
  T R;
  if (!(sin(thi + tht) > limit_map<T>::eps))
    R = pow((nipnt - 1.0) / (nipnt + 1.0), 2);
  else
    R = 0.5 * (pow(sin(thi - tht) / sin(thi + tht), 2) + pow(tan(thi - tht) / tan(thi + tht), 2));
  T xi = UnifClosed();

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

  return (0);
}



template<typename T>
void MC3D<T>::PropagatePhoton_SingleStep_UnmodulatedLight_alt(
  Photon<T>* phot,
  T* ds
)
{
  // std::cerr << "MC3D::PropagatePhoton_SingleStep_UnmodulatedLight" << std::endl;
  const int_fast64_t phot_curel = phot->curel;
  if (mua[phot_curel] > 0.0) {
    ER[phot_curel] += (1.0 - exp(-mua[phot_curel] * (*ds))) * phot->weight;
  } else {
    ER[phot_curel] += phot->weight * (*ds);
  }
}

template<typename T>
void MC3D<T>::PropagatePhoton_SingleStep_UnmodulatedLight(
  Photon<T>* phot,
  T* ds
)
{
  // std::cerr << "MC3D::PropagatePhoton_SingleStep_UnmodulatedLight" << std::endl;
  if (mua[phot->curel] > 0.0) {
    ER[phot->curel] += (1.0 - exp(-mua[phot->curel] * (*ds))) * phot->weight;
  } else {
    ER[phot->curel] += phot->weight * (*ds);
  }
}

template<typename T>
void MC3D<T>::PropagatePhoton_SingleStep_ModulatedLight_alt (
  Photon<T>* phot,
  T* ds
)
{

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
  const int_fast64_t phot_curel = phot->curel;

  // std::cerr << "MC3D::PropagatePhoton_SingleStep_ModulatedLight" << std::endl;
  ER[phot_curel] += phot->weight * (cos(phot->phase) - cos(-phot->phase - k[phot_curel] * (*ds)) * exp(-mua[phot_curel] * (*ds)));
  EI[phot_curel] += phot->weight * (-sin(phot->phase) + sin(phot->phase + k[phot_curel] * (*ds)) * exp(-mua[phot_curel] * (*ds)));

  phot->phase += k[phot_curel] * (*ds);

}

template<typename T>
void MC3D<T>::PropagatePhoton_SingleStep_ModulatedLight (
  Photon<T>* phot,
  T* ds
)
{

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
  // std::cerr << "MC3D::PropagatePhoton_SingleStep_ModulatedLight" << std::endl;
  ER[phot->curel] += phot->weight * (cos(phot->phase) - cos(-phot->phase - k[phot->curel] * (*ds)) * exp(-mua[phot->curel] * (*ds)));
  EI[phot->curel] += phot->weight * (-sin(phot->phase) + sin(phot->phase + k[phot->curel] * (*ds)) * exp(-mua[phot->curel] * (*ds)));

  phot->phase += k[phot->curel] * (*ds);

}

template<typename T>
int MC3D<T>::PropagatePhoton_SingleStep_TestBoundary(
  Photon<T>* phot,
  int_fast64_t* ib
)
{
  // Boundary element index
  *ib = -1 - phot->nextel;
  char BCType_ib = BCType[*ib];
  if ((BCType_ib == 'm') || (BCType_ib == 'L') || (BCType_ib == 'I') || (BCType_ib == 'C')) {
    // Mirror boundary condition -- Reflect the photon
    MirrorPhoton(phot, *ib);
    phot->curface = phot->nextface;
    return 2;
  } else {
    // Absorbing (a, l, i and c)
    // Check for mismatch between inner & outer index of refraction causes Fresnel transmission
    if (BCn[(*ib)] > 0.0) {
      if (FresnelPhoton(phot)) {
        return 2;
      }
    }

    if (omega <= 0.0) {
      EBR[(*ib)] += phot->weight;
    } else {
      EBR[(*ib)] += phot->weight * cos(phot->phase);
      EBI[(*ib)] -= phot->weight * sin(phot->phase);
    }
    // Photon propagation will terminate
    return 0;
  }
}

template<typename T>
int MC3D<T>::PropagatePhoton_SingleStep(
  Photon<T>* phot,
  T* prop,
  T* dist,
  T* ds,
  int_fast64_t* ib
)
{
  // std::cerr << "MC3D::PropagatePhoton_SingleStep" << std::endl;
  // Check through which face the photon will exit the current element
  if (WhichFace(phot, dist) == -1) {
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
    PropagatePhoton_SingleStep_UnmodulatedLight(phot, ds);
  } else {
    // Modulated light
    PropagatePhoton_SingleStep_ModulatedLight(phot, ds);
  }

  // Upgrade photon weight
  phot->weight *= exp(-mua[phot->curel] * (*ds));

  // Photon has reached a situation where it has to be scattered
  *prop -= *ds;
  if (*prop <= 0.0) {
    return 1;
  }

  // Otherwise the photon will continue to pass through the boundaries of the current element

  // Test for boundary conditions
  if (phot->nextel < 0) {
    return PropagatePhoton_SingleStep_TestBoundary(phot, ib);
  }

  // Test transmission from vacuum -> scattering media
  if ((mus[phot->curel] <= 0.0) && (mus[phot->nextel] > 0.0))
  {
    // Draw new propagation distance -- otherwise photon might travel without scattering
    *prop = -log(UnifOpen()) / mus[phot->nextel];
  }

  // Test for surival of the photon via roulette
  if (phot->weight < weight0)
  {
    if (UnifClosed() > chance) {
      return 0;
    }
    phot->weight /= chance;
  }

  // Fresnel transmission/reflection
  if (n[phot->curel] != n[phot->nextel])
  {
    if (FresnelPhoton(phot)) {
      return 2;
    }
  }

  // Upgrade remaining photon propagation lenght in case it is transmitted to different mus domain
  *prop *= mus[phot->curel] / mus[phot->nextel];



  // Update current face of the photon to that face which it will be on in the next element
  if (HN(phot->nextel, 0) == phot->curel) {
    phot->curface = 0;
  } else if (HN(phot->nextel, 1) == phot->curel) {
    phot->curface = 1;
  } else if (HN(phot->nextel, 2) == phot->curel) {
    phot->curface = 2;
  } else if (HN(phot->nextel, 3) == phot->curel) {
    phot->curface = 3;
  } else {
    return 0;
  }

  // Update current element of the photon
  phot->curel = phot->nextel;
  return 2;
}
/**
 * Return 0 for dead photon, 1 if it needs to be scattered, 2 if propagation continues
 * @param phot   [description]
 * @param [name] [description]
 */
template<typename T>
int MC3D<T>::PropagatePhoton_SingleStep_alt(
  Photon<T>* phot,
  T* prop,
  T* dist,
  T* ds,
  int_fast64_t* ib
)
{
  // std::cerr << "MC3D::PropagatePhoton_SingleStep" << std::endl;
  // Check through which face the photon will exit the current element
  if (WhichFace_alt(phot, dist) == -1) {
    return 0;
  }
  const int_fast64_t phot_curel = phot->curel;
  const int_fast64_t phot_nextel = phot->nextel;
  // Travel distance -- Either propagate to the boundary of the element, or to the end of the leap, whichever is closer
  *ds = fmin(*prop, *dist);

  // Move photon
  for (unsigned idx=0; idx<3; idx++) {
    phot->pos[idx] += phot->dir[idx] * (*ds);
  }
  // phot->pos[0] += phot->dir[0] * (*ds);
  // phot->pos[1] += phot->dir[1] * (*ds);
  // phot->pos[2] += phot->dir[2] * (*ds);

  // Upgrade element fluence
  if (omega <= 0.0) {
    // Unmodulated light
    PropagatePhoton_SingleStep_UnmodulatedLight_alt(phot, ds);
  } else {
    // Modulated light
    PropagatePhoton_SingleStep_ModulatedLight_alt(phot, ds);
  }

  // Upgrade photon weight
  phot->weight *= exp(-mua[phot_curel] * (*ds));

  // Photon has reached a situation where it has to be scattered
  *prop -= *ds;
  if (*prop <= 0.0) {
    return 1;
  }

  // Otherwise the photon will continue to pass through the boundaries of the current element

  // Test for boundary conditions
  if (phot_nextel < 0) {
    return PropagatePhoton_SingleStep_TestBoundary(phot, ib);
  }

  // Test transmission from vacuum -> scattering media
  if ((mus[phot_curel] <= 0.0) && (mus[phot_nextel] > 0.0))
  {
    // Draw new propagation distance -- otherwise photon might travel without scattering
    *prop = -log(UnifOpen()) / mus[phot_nextel];
  }

  // Test for surival of the photon via roulette
  if (phot->weight < weight0)
  {
    if (UnifClosed() > chance) {
      return 0;
    }
    phot->weight /= chance;
  }

  // Fresnel transmission/reflection
  if (n[phot_curel] != n[phot_nextel])
  {
    if (FresnelPhoton(phot)) {
      return 2;
    }
  }

  // Upgrade remaining photon propagation lenght in case it is transmitted to different mus domain
  *prop *= mus[phot_curel] / mus[phot_nextel];

  int_fast64_t* HN_data = HN.data + phot_nextel;
  int_fast64_t HN_Nx = HN.Nx;

  // Update current face of the photon to that face which it will be on in the next element
  bool curface_set = false;
  for (int_fast64_t idx=0; idx<4; idx++) {
    if (*HN_data == phot_curel) {
      phot->curface = idx;
      curface_set = true;
      break;
    }
    HN_data += HN_Nx;
  }

  if (!curface_set) {
    return 0;
  }

  // if (HN(phot_nextel, 0) == phot_curel) {
  //   phot->curface = 0;
  // } else if (HN(phot_nextel, 1) == phot_curel) {
  //   phot->curface = 1;
  // } else if (HN(phot_nextel, 2) == phot_curel) {
  //   phot->curface = 2;
  // } else if (HN(phot_nextel, 3) == phot_curel) {
  //   phot->curface = 3;
  // } else {
  //   return 0;
  // }

  // Update current element of the photon
  phot->curel = phot_nextel;
  return 2;
}




// Propagate a photon until it dies
template<typename T>
void MC3D<T>::PropagatePhoton(Photon<T> *phot, bool use_alt)
{

  if (use_alt) {
    T prop, dist, ds;
    int_fast64_t ib;
    int single_step_res;

    prop = -log(UnifOpen()) / mus[phot->curel];

    bool alive = true;

    while (alive)
    {
      // if (use_alt) {
      single_step_res = PropagatePhoton_SingleStep_alt(phot, &prop, &dist, &ds, &ib);
      // } else {
      //   single_step_res = PropagatePhoton_SingleStep(phot, &prop, &dist, &ds, &ib);
      // }

      if (single_step_res == 0) {
        loss++;
        alive = false;
      } else if (single_step_res == 1) {
        if (mus[phot->curel] > 0.0) {
          ScatterPhoton(phot);
        }
        prop = -log(UnifOpen()) / mus[phot->curel];
      }
    }
  } else {
    T prop, dist, ds;
    int_fast64_t ib;
    // Propagate until the photon dies
    while (1)
    {
      // Draw the propagation distance
      prop = -log(UnifOpen()) / mus[phot->curel];

      // Propagate until the current propagation distance runs out (and a scattering will occur)
      while (1)
      {
        // Check through which face the photon will exit the current element
        if (WhichFace(phot, &dist) == -1)
        {
          loss++;
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
            ER[phot->curel] += (1.0 - exp(-mua[phot->curel] * ds)) * phot->weight;
          }
          else
          {
            ER[phot->curel] += phot->weight * ds;
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

          ER[phot->curel] += phot->weight * (cos(phot->phase) - cos(-phot->phase - k[phot->curel] * ds) * exp(-mua[phot->curel] * ds));
          EI[phot->curel] += phot->weight * (-sin(phot->phase) + sin(phot->phase + k[phot->curel] * ds) * exp(-mua[phot->curel] * ds));

          phot->phase += k[phot->curel] * ds;
        }

        // Upgrade photon weight
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
          char BCType_ib = BCType[ib];
          if ((BCType_ib == 'm') || (BCType_ib == 'L') || (BCType_ib == 'I') || (BCType_ib == 'C'))
          {
            // Mirror boundary condition -- Reflect the photon
            MirrorPhoton(phot, ib);
            phot->curface = phot->nextface;
            continue;
          }
          else
          {
            // Absorbing (a, l, i and c)
            // Check for mismatch between inner & outer index of refraction causes Fresnel transmission
            if (BCn[ib] > 0.0)
              if (FresnelPhoton(phot))
                continue;

            if (omega <= 0.0)
            {
              EBR[ib] += phot->weight;
            }
            else
            {
              EBR[ib] += phot->weight * cos(phot->phase);
              EBI[ib] -= phot->weight * sin(phot->phase);
            }
            // Photon propagation will terminate
            return;
          }
        }

        // Test transmission from vacuum -> scattering media
        if ((mus[phot->curel] <= 0.0) && (mus[phot->nextel] > 0.0))
        {
          // Draw new propagation distance -- otherwise photon might travel without scattering
          prop = -log(UnifOpen()) / mus[phot->nextel];
        }

        // Test for surival of the photon via roulette
        if (phot->weight < weight0)
        {
          if (UnifClosed() > chance)
            return;
          phot->weight /= chance;
        }

        // Fresnel transmission/reflection
        if (n[phot->curel] != n[phot->nextel])
        {
          if (FresnelPhoton(phot))
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
          loss++;
          return;
        }

        // Update current element of the photon
        phot->curel = phot->nextel;
      }

      // Scatter photon
      if (mus[phot->curel] > 0.0)
        ScatterPhoton(phot);
    }
  }




}

// Run Monte Carlo
template<typename T>
void MC3D<T>::MonteCarlo(bool (*progress)(T), void (*finalchecks)(int,int), bool use_alt)
{
#ifdef USE_OMP
  // std::cerr << "Using OpenMP implementation with " << omp_get_max_threads() << " threads" << std::endl;

  // OpenMP implementation

  // Spawn new MC3D classes with Nphoton' = Nphoton / ThreadCount, and initialize mt_rng seed
  int_fast64_t ii, jj, nthread = omp_get_max_threads();
  int_fast64_t *ticks = new int_fast64_t[(int)nthread];
  MC3D<T> *MCS = new MC3D<T>[(int)nthread];
  bool abort_computation = false;
  // [AL] the progress bar gets an update after every TICK_VAL photons
#define TICK_VAL 1000
  for (ii = 0; ii < nthread; ii++)
  {
    MCS[ii] = *this;
    MCS[ii].Nphoton = Nphoton / nthread;
    MCS[ii].seed = (unsigned long) (MCS[ii].seed * totalthreads + ii);
    MCS[ii].InitRand();
    ticks[ii] = 0;
  }

  // [AL] if remainder of nphoton / nthread is non-zero, total photon count is not the same as Nphoton
  // therefore add the remaining photons to the last thread.
  long realnphot = 0;
  for (ii = 0; ii < nthread; ii++)
    realnphot += MCS[ii].Nphoton;
  MCS[nthread - 1].Nphoton += Nphoton - realnphot;
  // Compute Monte Carlo on each thread separetely
#pragma omp parallel
  {
    int_fast64_t iphoton, thread = omp_get_thread_num();
    Photon<T> phot;
    for (iphoton = 1; iphoton <= MCS[thread].Nphoton; iphoton++)
    {
      ticks[thread] = iphoton;
      if (iphoton % TICK_VAL == 0)
      {
#pragma omp critical
        if (thread == 0)
        {
          int_fast64_t jj, csum = 0;
          {
            for (jj = 0; jj < nthread; jj++)
            {
              csum += ticks[jj];
            }
            if (!progress(100 * ((T)csum / (T)Nphoton)))
            {
              abort_computation = true;
            }
          }
        }
        if (abort_computation)
          break;
      }
      MCS[thread].CreatePhoton(&phot);
      MCS[thread].PropagatePhoton(&phot, use_alt);
    }
  }
#ifdef VALOMC_MEX
  int_fast64_t csum = 0;
  for (jj = 0; jj < nthread; jj++)
  {
    csum += ticks[jj];
  }

  finalchecks(csum, Nphoton);

#endif
#pragma omp barrier

  // Sum up the results to first instance and delete MCS
  Nphoton = 0;
  loss = 0;
  for (jj = 0; jj < nthread; jj++)
  {
    Nphoton += MCS[jj].Nphoton;
    loss += MCS[jj].loss;
  }
  for (ii = 0; ii < H.Nx; ii++)
  {
    ER[ii] = EI[ii] = 0.0;
    for (jj = 0; jj < nthread; jj++)
    {
      ER[ii] += MCS[jj].ER[ii];
      EI[ii] += MCS[jj].EI[ii];
    }
  }
  for (ii = 0; ii < BH.Nx; ii++)
  {
    EBR[ii] = EBI[ii] = 0.0;
    for (jj = 0; jj < nthread; jj++)
    {
      EBR[ii] += MCS[jj].EBR[ii];
      EBI[ii] += MCS[jj].EBI[ii];
    }
  }

  for (ii = 0; ii < BH.Nx; ii++) // [AL]
  {
    DEBR[ii] = DEBI[ii] = 0.0;
    for (jj = 0; jj < nthread; jj++)
    {
      DEBR[ii] += MCS[jj].DEBR[ii];
      DEBI[ii] += MCS[jj].DEBI[ii];
    }
  }

  //  delete[] itick;
  delete[] ticks;
  delete[] MCS;

#else
  // std::cerr << "Using single thread implementation" << std::endl;
  // Single thread implementation
  long ii;
  long itick = max(static_cast<int_fast64_t>(1), Nphoton / 100);
  int percentage = 0;
  long iphoton;
  Photon<T> phot;
  for (iphoton = 0; iphoton < Nphoton; iphoton++)
  {
    if ((iphoton % itick == 0))
    {
      percentage = ((T)100.0 * iphoton / (T)Nphoton);
      if (!progress(percentage))
        break;
    }
    CreatePhoton(&phot);
    PropagatePhoton(&phot, use_alt);
  }

#endif

#ifdef USE_MPI
  // Sum up computation from each computer & normalize
  AllReduceArray(ER, MPI_SUM);
  AllReduceArray(EI, MPI_SUM);
  AllReduceArray(EBR, MPI_SUM);
  AllReduceArray(EBI, MPI_SUM);
  // Sum up computed photons
  long tmplong;
  MPI_Allreduce(&Nphoton, &tmplong, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  Nphoton = tmplong;
  // Sum up lost photons
  MPI_Allreduce(&loss, &tmplong, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  loss = tmplong;
#endif

  // Normalize output variables
  if (omega <= 0.0)
  {
    for (ii = 0; ii < H.Nx; ii++)
    {
      if (mua[ii] > 0.0)
        ER[ii] /= mua[ii] * ElementVolume(ii) * (T)Nphoton;
      else
        ER[ii] /= ElementVolume(ii) * (T)Nphoton;
    }
    for (ii = 0; ii < BH.Nx; ii++)
      EBR[ii] /= (T)Nphoton * ElementArea(ii);
  }
  else
  {
    for (ii = 0; ii < H.Nx; ii++)
    {
      T a = ER[ii], b = EI[ii];
      ER[ii] = (b * k[ii] + a * mua[ii]) / (pow(k[ii], 2) + pow(mua[ii], 2)) / (T)Nphoton / ElementVolume(ii);
      EI[ii] = -(a * k[ii] - b * mua[ii]) / (pow(k[ii], 2) + pow(mua[ii], 2)) / (T)Nphoton / ElementVolume(ii);
    }
    for (ii = 0; ii < BH.Nx; ii++)
    {
      EBR[ii] /= (T)Nphoton * ElementArea(ii);
      EBI[ii] /= (T)Nphoton * ElementArea(ii);
    }
  }

  if (progress != NULL)
    progress(100);
}

inline __m128 cross_sse (const __m128 v1, const __m128 v2)
{
  // __m128 a = _mm_setr_ps(v1[0], v1[1], v1[2], 0);
  // __m128 b = _mm_setr_ps(v2[0], v2[1], v2[2], 0);
  return _mm_sub_ps(
    _mm_mul_ps(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(3, 1, 0, 2))),
    _mm_mul_ps(_mm_shuffle_ps(v1, v1, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(3, 0, 2, 1)))
  );
  // float* c_ptr = reinterpret_cast<float*>(&c);
  // dest[0] = *c_ptr;
  // dest[1] = *(c_ptr + 1);
  // dest[2] = *(c_ptr + 2);

}

inline float dot_sse (const __m128 v1, const __m128 v2)
{
  // __m128 mulRes, shufReg, sumsReg;
  //  mulRes = _mm_mul_ps(v1, v2);
  //
  //  // Calculates the sum of SSE Register - https://stackoverflow.com/a/35270026/195787
  //  shufReg = _mm_movehdup_ps(mulRes);        // Broadcast elements 3,1 to 2,0
  //  sumsReg = _mm_add_ps(mulRes, shufReg);
  //  shufReg = _mm_movehl_ps(shufReg, sumsReg); // High Half -> Low Half
  //  sumsReg = _mm_add_ss(sumsReg, shufReg);
  //  return  _mm_cvtss_f32(sumsReg); // Result in the lower part of the SSE Register

  // __m128 a = _mm_setr_ps(v1[0], v1[1], v1[2], 0);
  // __m128 b = _mm_setr_ps(v2[0], v2[1], v2[2], 0);
  // __m128 r1 = _mm_mul_ps(v1, v2);
  // __m128 shuf = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));
  // __m128 sums = _mm_add_ps(r1, shuf);
  // shuf = _mm_movehl_ps(shuf, sums);
  // sums = _mm_add_ss(sums, shuf);
  // return _mm_cvtss_f32(sums);
  // return _mm_dp_ps (v1, v2, 0);
  __m128 mul_res = _mm_mul_ps(v1, v2);
  return mul_res[0] + mul_res[1] + mul_res[2];
  // return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

inline __m128 sub_sse (const __m128 v1, const __m128 v2)
{
  // __m128 a = _mm_setr_ps(v1[0], v1[1], v1[2], 0);
  // __m128 b = _mm_setr_ps(v2[0], v2[1], v2[2], 0);
  // __m128 c = _mm_sub_ps(a, b);
  // float* c_ptr = reinterpret_cast<float*>(&c);
  //
  // dest[0] = *c_ptr;
  // dest[1] = *(c_ptr + 1);
  // dest[2] = *(c_ptr + 2);
  return _mm_sub_ps(v1, v2);
}

// template<typename T>
int RayTriangleIntersects_alt(const float* O, const float* D, const float* V0, const float* V1, const float* V2, float *t) //, T* scratch)
{
// O is the origin of line, D is the direction of line
// V0, V1, V2 are the corners of the triangle.
//   If the line intersects the triangle, will return nonzero and t will be set to value such that
// O + t D will equal the intersection point
//
// Source:
//   Fast, Minimum Storage Ray/Triangle Intersection
//   Tomas Moeller and Ben Trumbore
//   Journal of Graphics Tools, 2(1):21--28, 1997.
  const float eps = limit_map<float>::eps;
  // The algorithm
  __m128 O_128 = _mm_setr_ps(O[0], O[1], O[2], 0.0f);
  __m128 D_128 = _mm_setr_ps(D[0], D[1], D[2], 0.0f);
  __m128 V0_128 = _mm_setr_ps(V0[0], V0[1], V0[2], 0.0f);
  __m128 V1_128 = _mm_setr_ps(V1[0], V1[1], V1[2], 0.0f);
  __m128 V2_128 = _mm_setr_ps(V2[0], V2[1], V2[2], 0.0f);

  __m128 edge1, edge2, tvec, pvec, qvec;
  float det, inv_det, u, v;

  edge1 = sub_sse(V1_128, V0_128);
  edge2 = sub_sse(V2_128, V0_128);
  pvec = cross_sse(D_128, edge2);
  det = dot_sse(edge1, pvec);
  if ((-eps < det) && (det < eps)) {
    return 0;
  }
  inv_det = 1.0 / det;
  tvec = sub_sse(O_128, V0_128);
  u = dot_sse(tvec, pvec) * inv_det;
  if ((u < 0.0) || (u > 1.0)) {
    return 0;
  }
  qvec = cross_sse(tvec, edge1);
  v = dot_sse(D_128, qvec) * inv_det;
  if ((v < 0.0) || (u + v > 1.0)) {
    return 0;
  }
  *t = dot_sse(edge2, qvec) * inv_det;

  return 1;
}


// Check if ray and triangle intersect
template<typename T>
int RayTriangleIntersects(T O[3], T D[3], T V0[3], T V1[3], T V2[3], T *t)
{
// O is the origin of line, D is the direction of line
// V0, V1, V2 are the corners of the triangle.
//   If the line intersects the triangle, will return nonzero and t will be set to value such that
// O + t D will equal the intersection point
//
// Source:
//   Fast, Minimum Storage Ray/Triangle Intersection
//   Tomas Moeller and Ben Trumbore
//   Journal of Graphics Tools, 2(1):21--28, 1997.
// Cross product
#define CROSS(dest, v1, v2)                \
  dest[0] = v1[1] * v2[2] - v1[2] * v2[1]; \
  dest[1] = v1[2] * v2[0] - v1[0] * v2[2]; \
  dest[2] = v1[0] * v2[1] - v1[1] * v2[0];
// Dot product
#define DOT(v1, v2) (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
// Vector substraction
#define SUB(dest, v1, v2)  \
  dest[0] = v1[0] - v2[0]; \
  dest[1] = v1[1] - v2[1]; \
  dest[2] = v1[2] - v2[2];

  // The algorithm
  T edge1[3], edge2[3], tvec[3], pvec[3], qvec[3], det, inv_det, u, v;
  SUB(edge1, V1, V0);
  SUB(edge2, V2, V0);
  CROSS(pvec, D, edge2);
  det = DOT(edge1, pvec);
  if ((-limit_map<T>::eps < det) && (det < limit_map<T>::eps))
    return (0);
  inv_det = 1.0 / det;
  SUB(tvec, O, V0);
  u = DOT(tvec, pvec) * inv_det;
  if ((u < 0.0) || (u > 1.0))
    return (0);
  CROSS(qvec, tvec, edge1);
  v = DOT(D, qvec) * inv_det;
  if ((v < 0.0) || (u + v > 1.0))
    return (0);
  *t = DOT(edge2, qvec) * inv_det;

  return (1);
}

#endif
