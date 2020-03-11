// benchmark different implementations of MonteCarlo algorithm.

#include <iostream>

#include "3d/MC3D.hpp"
#include "test/util.hpp"
#include "bench/util.hpp"


int main (int argc, char *argv[]) {
  unsigned nphotons = 1000;
  unsigned niter = 10;
  if (argc > 1) {
    nphotons = std::stoi(argv[1]);
    niter = std::stoi(argv[2]);
  }
  std::cerr << "Using " << nphotons << " photons" << std::endl;

  auto t0 = ValoMC::bench::util::now();
  static ValoMC::test::util::TestConfig config;
  config.init_MC3D_from_json();
  ValoMC::bench::util::duration delta_io = ValoMC::bench::util::now() - t0;
  std::cerr << "Took " << delta_io.count() << " s to load data from disk" << std::endl;

  MC3D mc3d = config.get_mc3d();

  mc3d.Nphoton = nphotons;
  mc3d.ErrorChecks();
  mc3d.Init();

  t0 = ValoMC::bench::util::now();
  for (unsigned iter=0; iter<niter; iter++) {
    mc3d.MonteCarlo(
      [](double perc) -> bool {return true;},
      [](int csum, int Nphoton) {
        std::cerr << "csum=" << csum << ", Nphoton=" << Nphoton << std::endl;
      }
    );
  }
  ValoMC::bench::util::duration delta_cpu = ValoMC::bench::util::now() - t0;

  std::cerr << "CPU version took " << delta_cpu.count() << " s, " << delta_cpu.count() / niter << " s per loop" << std::endl;
}
