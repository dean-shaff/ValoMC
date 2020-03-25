#include <iostream>
#include <fstream>
#include <string>

#include "catch.hpp"

#include "util.hpp"


CATCH_TRANSLATE_EXCEPTION( mcerror& ex ) {
  return std::string(errorstring(ex));
}


TEMPLATE_TEST_CASE (
  "MC3D", "[unit][MC3D]",
  float, double
)
{
  ValoMC::test::util::TestConfig<TestType> config;
  config.init_MC3D_from_json();

  MC3D<TestType> mc3d = config.get_mc3d();
  mc3d.Nphoton = 1000;
  mc3d.ErrorChecks();
  mc3d.Init();

  mc3d.MonteCarlo(
    [](double perc) -> bool {return true;},
    [](int csum, int Nphoton) {
      std::cerr << "csum=" << csum << ", Nphoton=" << Nphoton << std::endl;
    },
    true
  );

  CHECK(mc3d.loss == mc3d.Nphoton);

  mc3d.loss = 0;
  mc3d.MonteCarlo(
    [](double perc) -> bool {return true;},
    [](int csum, int Nphoton) {
      std::cerr << "csum=" << csum << ", Nphoton=" << Nphoton << std::endl;
    },
    false
  );

  CHECK(mc3d.loss == mc3d.Nphoton);



}
