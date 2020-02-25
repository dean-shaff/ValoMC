#include <string>

#include "catch.hpp"

#include "MC3D_cuda.cuh"
#include "util.hpp"

static ValoMC::test::util::TestConfig config;

TEST_CASE(
  "MC3DCUDA can calculate the amount of space needed ",
  "[MC3DCUDA][unit][get_total_memory_usage]"
)
{
  config.init_MC3D_from_json();

  ValoMC::MC3DCUDA mc3dcuda(config.get_mc3d(), 1);

  unsigned int usage = mc3dcuda.get_total_memory_usage();

  std::cerr << "usage=" << usage << std::endl;

}

TEST_CASE(
  "MC3DCUDA can initialize properties from MC3D object",
  "[MC3DCUDA][unit][init]"
)
{
  config.init_MC3D_from_json();
  MC3D mc3d = config.get_mc3d();

  mc3d.Nphoton = 100;

  ValoMC::MC3DCUDA mc3dcuda(mc3d, 1);

  mc3dcuda.init();

  CHECK(mc3dcuda.get_nphotons() == 100);
}

__global__ void iter_states(
  curandState_t* states,
  unsigned states_size,
  unsigned* result
) {
  const unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx == 0) {
    for (unsigned istate=0; istate<states_size; istate++) {
      states[istate];
      (*result)++;
    }
  }
}


TEST_CASE(
  "MC3DCUDA can allocate memory",
  "[MC3DCUDA][unit][allocate]"
)
{
  unsigned states_size = 10;
  config.init_MC3D_from_json();

  ValoMC::MC3DCUDA mc3dcuda(config.get_mc3d(), states_size);

  mc3dcuda.allocate();
  unsigned result_h = 0;
  unsigned* result_d;
  gpuErrchk(cudaMalloc((void**)&result_d, sizeof(unsigned)));
  gpuErrchk(cudaMemcpy(result_d, &result_h, sizeof(unsigned), cudaMemcpyHostToDevice));
  iter_states<<<1,1>>>(mc3dcuda.get_states(), mc3dcuda.get_states_size(), result_d);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaMemcpy(&result_h, result_d, sizeof(unsigned), cudaMemcpyDeviceToHost));

  REQUIRE(result_h == mc3dcuda.get_states_size());

  gpuErrchk(cudaDeviceSynchronize());
}

TEST_CASE(
  "MC3DCUDA can transfer arrays from MC3D object to internal Array objects",
  "[MC3DCUDA][unit][h2d]"
)
{
  config.init_MC3D_from_json();

  ValoMC::MC3DCUDA mc3dcuda(config.get_mc3d(), 1);

  mc3dcuda.allocate();
  mc3dcuda.h2d();
  gpuErrchk(cudaDeviceSynchronize());

}

TEST_CASE(
  "MC3DCUDA can transfer result arrays to MC3D object",
  "[MC3DCUDA][unit][d2h]"
)
{
  config.init_MC3D_from_json();

  ValoMC::MC3DCUDA mc3dcuda(config.get_mc3d(), 1);

  mc3dcuda.allocate();
  // fill arrays
  mc3dcuda.d2h();
  gpuErrchk(cudaDeviceSynchronize());
}
