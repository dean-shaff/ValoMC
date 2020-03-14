// benchmark different implementations of MonteCarlo algorithm.

#include <iostream>

#include "3d/MC3D.hpp"
#include "test/util.hpp"
#include "bench/util.hpp"

template<typename T>
ValoMC::bench::util::duration benchmark (unsigned nphotons, unsigned niter) {
  auto t0 = ValoMC::bench::util::now();
  ValoMC::test::util::TestConfig<T> config;
  config.init_MC3D_from_json();
  ValoMC::bench::util::duration delta_io = ValoMC::bench::util::now() - t0;
  std::cerr << "Took " << delta_io.count() << " s to load data from disk" << std::endl;

  MC3D<T> mc3d = config.get_mc3d();

  mc3d.Nphoton = nphotons;
  mc3d.ErrorChecks();
  mc3d.Init();

  t0 = ValoMC::bench::util::now();
  for (unsigned iter=0; iter<niter; iter++) {
    mc3d.MonteCarlo(
      [](T perc) -> bool {return true;},
      [](int csum, int Nphoton) {
        std::cerr << "csum=" << csum << ", Nphoton=" << Nphoton << std::endl;
      }
    );
  }
  ValoMC::bench::util::duration delta = ValoMC::bench::util::now() - t0;
  return delta;
}


int main (int argc, char *argv[]) {
  unsigned nphotons = 1000;
  unsigned niter = 10;
  if (argc > 1) {
    nphotons = std::stoi(argv[1]);
    niter = std::stoi(argv[2]);
  }
  std::cerr << "Using " << nphotons << " photons" << std::endl;

  auto delta_double = benchmark<double>(nphotons, niter);
  auto delta_float = benchmark<float>(nphotons, niter);

  std::cerr << "CPU double version took " << delta_double.count() << " s, " << delta_double.count() / niter << " s per loop" << std::endl;
  std::cerr << "CPU float version took " << delta_float.count() << " s, " << delta_float.count() / niter << " s per loop" << std::endl;

  std::cerr << "CPU float version " << delta_double / delta_float << " times faster" << std::endl;
}
