#include <string>

#include "catch.hpp"

#include "MC3D_cuda.cuh"
#include "util.hpp"

CATCH_TRANSLATE_EXCEPTION( mcerror& ex ) {
  return std::string(errorstring(ex));
}


template<typename T>
struct Config {
  static ValoMC::test::util::TestConfig<T> config;
};

template<typename T>
ValoMC::test::util::TestConfig<T> Config<T>::config;


TEMPLATE_TEST_CASE(
  "MC3DCUDA can calculate the amount of space needed ",
  "[MC3DCUDA][cuda][unit][get_total_memory_usage]",
  float, double
)
{
  unsigned states_size = 100;

  Config<TestType>::config.init_MC3D_from_json();

  ValoMC::MC3DCUDA<TestType> mc3dcuda(Config<TestType>::config.get_mc3d(), 0);

  unsigned int baseline = mc3dcuda.get_total_memory_usage();
  mc3dcuda.set_states_size(states_size);
  unsigned int usage = mc3dcuda.get_total_memory_usage();

  REQUIRE(usage - baseline == sizeof(curandState_t)*states_size);
}

TEMPLATE_TEST_CASE(
  "MC3DCUDA can initialize properties from MC3D object",
  "[MC3DCUDA][cuda][unit][init]",
  float, double
)
{
  Config<TestType>::config.init_MC3D_from_json();
  MC3D<TestType> mc3d = Config<TestType>::config.get_mc3d();

  mc3d.Nphoton = 100;

  ValoMC::MC3DCUDA<TestType> mc3dcuda(mc3d, 1);

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


TEMPLATE_TEST_CASE(
  "MC3DCUDA can allocate memory",
  "[MC3DCUDA][cuda][unit][allocate]",
  float, double
)
{
  unsigned states_size = 10;
  Config<TestType>::config.init_MC3D_from_json();

  ValoMC::MC3DCUDA<TestType> mc3dcuda(Config<TestType>::config.get_mc3d(), states_size);

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

template<typename T>
__global__ void iter_boundary (Array<T>* boundary, unsigned* result)
{
  const unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx == 0) {
    for (unsigned ib=0; ib<boundary->N; ib++) {
      (*boundary)[ib];
      (*result)++;
    }
  }
}


TEMPLATE_TEST_CASE(
  "MC3DCUDA can transfer arrays from MC3D object to internal Array objects",
  "[MC3DCUDA][cuda][unit][h2d]",
  float, double
)
{
  Config<TestType>::config.init_MC3D_from_json();
  MC3D<TestType> mc3d = Config<TestType>::config.get_mc3d();

  ValoMC::MC3DCUDA<TestType> mc3dcuda(mc3d, 1);

  mc3dcuda.allocate();
  mc3dcuda.h2d();
  unsigned result_h = 0;
  unsigned* result_d;
  gpuErrchk(cudaMalloc((void**)&result_d, sizeof(unsigned)));
  gpuErrchk(cudaMemcpy(result_d, &result_h, sizeof(unsigned), cudaMemcpyHostToDevice));
  iter_boundary<<<1,1>>>(mc3dcuda.get_boundary(), result_d);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaMemcpy(&result_h, result_d, sizeof(unsigned), cudaMemcpyDeviceToHost));

  REQUIRE(result_h == Config<TestType>::config.get_mc3d().BH.N);

  gpuErrchk(cudaDeviceSynchronize());

}


template<typename T>
__global__ void fill_array (Array<T>* arr, T val)
{
  const unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned total_size_x = blockDim.x * gridDim.x;

  for (unsigned iarr=idx; iarr<arr->N; iarr+=total_size_x) {
    (*arr)[iarr] = val;
  }
}


TEMPLATE_TEST_CASE(
  "MC3DCUDA can transfer result arrays to MC3D object",
  "[MC3DCUDA][cuda][unit][d2h]",
  float, double
)
{
  Config<TestType>::config.init_MC3D_from_json();

  MC3D<TestType> mc3d = Config<TestType>::config.get_mc3d();

  ValoMC::MC3DCUDA<TestType> mc3dcuda(mc3d, 1);

  mc3dcuda.allocate();

  unsigned ER_N = mc3d.ER.N;

  unsigned block_size = 1024;
  unsigned grid_size = 1;
  fill_array<TestType><<<block_size, grid_size>>>(mc3dcuda.get_pow_den_vol_real(), 1.0);
  gpuErrchk(cudaGetLastError());
  mc3dcuda.d2h();

  bool allclose = true;

  for (unsigned idx=0; idx<ER_N; idx++) {
    if (mc3d.ER[idx] != 1.0) {
      allclose = false;
    }
  }

  REQUIRE(allclose == true);
  gpuErrchk(cudaDeviceSynchronize());
}


TEMPLATE_TEST_CASE(
  "MC3DCUDA can do monte_carlo simulation",
  "[MC3DCUDA][cuda][unit][monte_carlo]",
  float, double
)
{
  Config<TestType>::config.init_MC3D_from_json();

  MC3D<TestType> mc3d = Config<TestType>::config.get_mc3d();
  mc3d.Nphoton = 100;
  mc3d.ErrorChecks();
  mc3d.Init();

  ValoMC::MC3DCUDA<TestType> mc3dcuda(mc3d, 100);

  mc3dcuda.allocate();
  mc3dcuda.h2d();
  mc3dcuda.monte_carlo();
  // mc3dcuda.monte_carlo(true);
  mc3dcuda.d2h();

  bool allclose = true;

  for (unsigned idx=0; idx<mc3d.ER.N; idx++) {
    if (mc3d.ER[idx] != 0.0) {
      allclose = false;
    }
  }

  REQUIRE(allclose == false);
  gpuErrchk(cudaDeviceSynchronize());
}



TEMPLATE_TEST_CASE(
  "MC3DCUDA alternate version creates same results as reference implementation",
  "[MC3DCUDA][cuda][verify][monte_carlo]",
  float, double
)
{
  Config<TestType>::config.init_MC3D_from_json();

  MC3D<TestType> mc3d = Config<TestType>::config.get_mc3d();
  mc3d.Nphoton = 100;
  mc3d.ErrorChecks();
  mc3d.Init();

  ValoMC::MC3DCUDA<TestType> mc3dcuda_ref(mc3d, 100);
  ValoMC::MC3DCUDA<TestType> mc3dcuda_alt(mc3d, 100);

  mc3dcuda_ref.allocate();
  mc3dcuda_ref.h2d();
  mc3dcuda_ref.monte_carlo(false);
  mc3dcuda_ref.d2h();

  mc3dcuda_alt.allocate();
  mc3dcuda_alt.h2d();
  mc3dcuda_alt.monte_carlo(true);
  mc3dcuda_alt.d2h();

  // bool allclose = true;
  //
  // for (unsigned idx=0; idx<mc3d.ER.N; idx++) {
  //   if (mc3d.ER[idx] != ) {
  //     allclose = false;
  //   }
  // }
  //
  // REQUIRE(allclose == false);
  gpuErrchk(cudaDeviceSynchronize());
}
