#include <iostream>
#include <chrono>

#include "MC3D_cuda.cuh"
#include "test/util.hpp"

using duration = std::chrono::duration<double, std::ratio<1>>;

std::chrono::time_point<std::chrono::high_resolution_clock> now () {
  return std::chrono::high_resolution_clock::now();
}


int main (int argc, char *argv[]) {
  unsigned nphotons = 1000;
  if (argc > 1) {
    nphotons = std::stoi(argv[1]);
  }

  auto t0 = now();
  static ValoMC::test::util::TestConfig config;
  config.init_MC3D_from_json();
  duration delta_io = now() - t0;
  std::cerr << "Took " << delta_io.count() << " s to load data from disk" << std::endl;

  MC3D mc3d = config.get_mc3d();

  mc3d.Nphoton = nphotons;
  mc3d.ErrorChecks();
  mc3d.Init();

  unsigned states_size = nphotons;
  if (states_size > 1000) {
    states_size = 1000;
  }

  t0 = now();
  ValoMC::MC3DCUDA mc3dcuda (mc3d, states_size);
  mc3dcuda.allocate();
  mc3dcuda.h2d();
  cudaDeviceSynchronize();
  duration delta_h2d = now() - t0;

  std::cerr << "Took " << delta_h2d.count() << " s to transfer to GPU" << std::endl;

  t0 = now();
  mc3d.MonteCarlo(
    [](double perc) -> bool {return true;},
    [](int csum, int Nphoton) {
      std::cerr << "csum=" << csum << ", Nphoton=" << Nphoton << std::endl;
    }
  );
  duration delta_cpu = now() - t0;

  t0 = now();
  mc3dcuda.monte_carlo();
  cudaDeviceSynchronize();
  duration delta_gpu = now() - t0;

  std::cerr << "CPU version took " << delta_cpu.count() << " s" << std::endl;
  std::cerr << "GPU version took " << delta_gpu.count() << " s" << std::endl;

  if (delta_gpu > delta_cpu) {
    std::cerr << "CPU version " << delta_gpu.count() / delta_cpu.count() << " times faster" << std::endl;
  } else {
    std::cerr << "GPU version " << delta_cpu.count() / delta_gpu.count() << " times faster" << std::endl;
  }

}
