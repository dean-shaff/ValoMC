#include <stdio.h> // for fprintf, stderr

#include "MC3D_cuda.cuh"
#include "MC3D_util.cuh"

namespace ValoMC {

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
  double* dist,
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



}
