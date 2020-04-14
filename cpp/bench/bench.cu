#include <iostream>
#include <chrono>

#include "MC3D_cuda.cuh"
#include "test/util.hpp"
#include "util.hpp"


int main (int argc, char *argv[]) {
  unsigned nphotons = 1000;
  unsigned niter = 10;
  if (argc > 1) {
    nphotons = std::stoi(argv[1]);
    niter = std::stoi(argv[2]);
  }
  unsigned states_size = nphotons;
  // if (states_size > 1000) {
  //   states_size = 1000;
  // }
  std::cerr << "Using " << nphotons << " photons" << std::endl;
  std::cerr << "Using " << states_size << " curandState_t states" << std::endl;

  auto t0 = ValoMC::bench::util::now();
  static ValoMC::test::util::TestConfig<double> config;
  config.init_MC3D_from_json();
  ValoMC::bench::util::duration delta_io = ValoMC::bench::util::now() - t0;
  std::cerr << "Took " << delta_io.count() << " s to load data from disk" << std::endl;

  MC3D<double> mc3d = config.get_mc3d();

  mc3d.Nphoton = nphotons;
  mc3d.ErrorChecks();
  mc3d.Init();

  ValoMC::MC3DCUDA<double> mc3dcuda (mc3d, states_size);
  std::cerr << "mc3dcuda.get_max_block_size_init_state()=" << mc3dcuda.get_max_block_size_init_state() << std::endl;
  std::cerr << "mc3dcuda.get_max_block_size_monte_carlo()=" << mc3dcuda.get_max_block_size_monte_carlo() << std::endl;

  t0 = ValoMC::bench::util::now();
  mc3dcuda.allocate();
  mc3dcuda.h2d();
  cudaDeviceSynchronize();
  ValoMC::bench::util::duration delta_h2d = ValoMC::bench::util::now() - t0;


  std::cerr << "Took " << delta_h2d.count() << " s to transfer to GPU" << std::endl;

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

  t0 = ValoMC::bench::util::now();
  for (unsigned iter=0; iter<niter; iter++) {
    mc3dcuda.monte_carlo(false);
    cudaDeviceSynchronize();
  }
  ValoMC::bench::util::duration delta_gpu = ValoMC::bench::util::now() - t0;

  t0 = ValoMC::bench::util::now();
  for (unsigned iter=0; iter<niter; iter++) {
    mc3dcuda.monte_carlo(true);
    cudaDeviceSynchronize();
  }
  ValoMC::bench::util::duration delta_gpu_alt = ValoMC::bench::util::now() - t0;


  std::cerr << "CPU version took " << delta_cpu.count() << " s, " << delta_cpu.count() / niter << " s per loop" << std::endl;
  std::cerr << "GPU version took " << delta_gpu.count() << " s, " << delta_gpu.count() / niter << " s per loop"<< std::endl;
  std::cerr << "Alt GPU version took " << delta_gpu_alt.count() << " s, " << delta_gpu_alt.count() / niter << " s per loop"<< std::endl;

  if (delta_gpu > delta_cpu) {
    std::cerr << "CPU version " << delta_gpu.count() / delta_cpu.count() << " times faster" << std::endl;
  } else {
    std::cerr << "GPU version " << delta_cpu.count() / delta_gpu.count() << " times faster" << std::endl;
  }

}
