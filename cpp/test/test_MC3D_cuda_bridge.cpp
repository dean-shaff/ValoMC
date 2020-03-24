#include <string>

#include "catch.hpp"

#include "MC3D_cuda_bridge.hpp"
#include "util.hpp"


TEMPLATE_TEST_CASE(
  "Can call bridging code from C++ source file",
  "[bridge][unit][monte_carlo]",
  float, double
)
{
  ValoMC::test::util::TestConfig<TestType> config;

  unsigned states_size = 100;
  unsigned nphotons = 100;

  config.init_MC3D_from_json();

  MC3D<TestType> mc3d = config.get_mc3d();
  mc3d.Nphoton = nphotons;

  REQUIRE_THROWS(ValoMC::monte_carlo(mc3d, states_size));

  mc3d.ErrorChecks();
  mc3d.Init();

  ValoMC::monte_carlo(mc3d, states_size);
}
