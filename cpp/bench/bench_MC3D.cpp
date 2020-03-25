// benchmark different implementations of MonteCarlo algorithm.

#include <iostream>

#include "3d/MC3D.hpp"
#include "test/util.hpp"
#include "bench/util.hpp"

template<typename T>
void benchmark (unsigned nphotons, unsigned niter) {
  auto t0 = ValoMC::bench::util::now();
  ValoMC::test::util::TestConfig<T> config;
  config.init_MC3D_from_json();
  ValoMC::bench::util::duration delta_io = ValoMC::bench::util::now() - t0;
  std::cerr << "Took " << delta_io.count() << " s to load data from disk" << std::endl;

  MC3D<T> mc3d = config.get_mc3d();

  mc3d.seed = 1584769671;
  std::cerr << "mc3d.H.Nx=" << mc3d.H.Nx << std::endl;
  mc3d.Nphoton = nphotons;
  mc3d.ErrorChecks();
  mc3d.Init();

  ValoMC::bench::util::duration delta_alt;
  ValoMC::bench::util::duration delta_no_alt;

  t0 = ValoMC::bench::util::now();
  auto t1 = ValoMC::bench::util::now();
  for (unsigned iter=0; iter<niter; iter++) {
    mc3d.InitRand();
    t0 = now();
    mc3d.MonteCarlo(
      [](double perc) -> bool {return true;},
      [](int csum, int Nphoton) {
        std::cerr << "csum=" << csum << ", Nphoton=" << Nphoton << std::endl;
      },
      false
    );
    delta_no_alt += (ValoMC::bench::util::now() - t0);
    std::cerr << "Lost " << mc3d.loss << " photons" << std::endl;
    mc3d.InitRand();
    t1 = now();
    mc3d.MonteCarlo(
      [](double perc) -> bool {return true;},
      [](int csum, int Nphoton) {
        std::cerr << "csum=" << csum << ", Nphoton=" << Nphoton << std::endl;
      },
      true
    );
    delta_alt += (ValoMC::bench::util::now() - t1);
    std::cerr << "Lost " << mc3d.loss << " photons" << std::endl;


  }
  // ValoMC::bench::util::duration delta = ValoMC::bench::util::now() - t0;
  std::cerr << "Alt version Took " << delta_alt.count() << " s, " << delta_alt.count() / niter << " s per loop" << std::endl;
  std::cerr << "no Alt version Took " << delta_no_alt.count() << " s, " << delta_no_alt.count() / niter << " s per loop" << std::endl;

  if (delta_no_alt > delta_alt) {
    std::cerr << "Alternate version " << delta_no_alt.count() / delta_alt.count() << " times faster"  << std::endl;
  } else {
    std::cerr << "Normal version " << delta_alt.count() / delta_no_alt.count() << " times faster"  << std::endl;
  }
  // return delta;
}


int main (int argc, char *argv[]) {
  unsigned nphotons = 1000;
  unsigned niter = 10;
  if (argc > 1) {
    nphotons = std::stoi(argv[1]);
    niter = std::stoi(argv[2]);
  }
  std::cerr << "Using " << nphotons << " photons" << std::endl;
  // std::cerr << "double" << std::endl;
  // benchmark<double>(nphotons, niter);
  std::cerr << "float" << std::endl;
  benchmark<float>(nphotons, niter);
  // auto delta_float = benchmark<float>(nphotons, niter);

  // std::cerr << "CPU double version took " << delta_double.count() << " s, " << delta_double.count() / niter << " s per loop" << std::endl;
  // std::cerr << "CPU float version took " << delta_float.count() << " s, " << delta_float.count() / niter << " s per loop" << std::endl;

  // std::cerr << "CPU float version " << delta_double / delta_float << " times faster" << std::endl;
}
