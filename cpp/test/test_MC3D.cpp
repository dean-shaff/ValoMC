#include <iostream>
#include <fstream>
#include <string>

#include "catch.hpp"

#include "util.hpp" // have to do this include before MC3D due to max macro definition in MC3D.hpp

#include "MC3D.hpp"

CATCH_TRANSLATE_EXCEPTION( mcerror& ex ) {
  return std::string(errorstring(ex));
}


TEST_CASE ("MC3D", "[unit]")
{
  const std::string json_file_path = "/home/dean/work/freelance/Yoav-Mor_28-01-2020_CUDA-Matlab-ValoMC/ValoMC/cpp/test/MC3Dmex.input.json";

  MC3D mc3d;

  ValoMC::test::util::init_MC3D_from_json<MC3D>(
    json_file_path,
    mc3d
  );
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
