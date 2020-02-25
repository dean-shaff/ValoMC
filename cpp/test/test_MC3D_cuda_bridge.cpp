#include <string>

#include "catch.hpp"

#include "MC3D_cuda_bridge.hpp"
#include "util.hpp"

static ValoMC::test::util::TestConfig config;

TEST_CASE(
  "",
  "[MC3DCUDA][unit][get_total_memory_usage]"
)
{

  unsigned states_size = 100;

  config.init_MC3D_from_json();

  ValoMC::MC3DCUDA mc3dcuda(config.get_mc3d(), 0);

  unsigned int baseline = mc3dcuda.get_total_memory_usage();
  mc3dcuda.set_states_size(states_size);
  unsigned int usage = mc3dcuda.get_total_memory_usage();

  REQUIRE(usage - baseline == sizeof(curandState_t)*states_size);
}
