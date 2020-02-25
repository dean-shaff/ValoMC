#include <iostream>
#include <fstream>
#include <string>

#include "catch.hpp"

#include "util.hpp"

static ValoMC::test::util::TestConfig config;

CATCH_TRANSLATE_EXCEPTION( mcerror& ex ) {
  return std::string(errorstring(ex));
}


TEST_CASE ("MC3D", "[unit][MC3D]")
{
  config.init_MC3D_from_json();

  MC3D mc3d = config.get_mc3d();
  mc3d.Nphoton = 100;
  mc3d.ErrorChecks();
  mc3d.Init();

  mc3d.MonteCarlo(
    [](double perc) -> bool {return true;},
    [](int csum, int Nphoton) {
      std::cerr << "csum=" << csum << ", Nphoton=" << Nphoton << std::endl;
    }
  );

}
