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

  ValoMC::MC3DCUDA mc3dcuda;
  mc3dcuda.set_mc3d(config.get_mc3d());

  unsigned int usage = mc3dcuda.get_total_memory_usage();
}


TEST_CASE(
  "MC3DCUDA can transfer arrays from host to device",
  "[MC3DCUDA][unit][allocate_attributes]"
)
{
  config.init_MC3D_from_json();

  ValoMC::MC3DCUDA mc3dcuda;
  mc3dcuda.set_mc3d(config.get_mc3d());
  mc3dcuda.allocate_attributes();
}
