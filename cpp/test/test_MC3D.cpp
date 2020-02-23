#include <iostream>
#include <fstream>
#include <string>

#include "catch.hpp"

#include "util.hpp"

const std::string json_file_path = ValoMC::test::util::get_env_var(
  "MC3D_TEST_DATA_PATH", "./../cpp/test/MC3Dmex.input.json");

TEST_CASE ("MC3D", "[unit][MC3D]")
{
  MC3D mc3d;

  ValoMC::test::util::init_MC3D_from_json(
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
