#include <string>

#include "catch.hpp"

#include "MC3D_cuda.cuh"
#include "util.hpp"

static ValoMC::test::util::TestConfig config;

TEST_CASE(
  "MC3DCUDA can calculate the amount of space needed ",
  "[MC3DCUDA][unit][get_total_memory_usage]"
)
{
  config.init_MC3D_from_json();

  ValoMC::MC3DCUDA mc3dcuda(config.get_mc3d(), 1);

  unsigned int usage = mc3dcuda.get_total_memory_usage();
}

// init
// allocate
// h2d
// d2h

// TEST_CASE(
//   "MC3DCUDA can transfer arrays from host to device",
//   "[MC3DCUDA][unit][allocate_attributes]"
// )
// {
//   config.init_MC3D_from_json();
//
//   ValoMC::MC3DCUDA mc3dcuda(mc3d, 1);
//   mc3dcuda.set_mc3d(config.get_mc3d());
//   mc3dcuda.allocate_attributes();
// }
