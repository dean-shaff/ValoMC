#include <random>
#include <vector>
#include <numeric>

#include "catch.hpp"

#include "MC3D_util.cuh"
#include "MC3D.hpp"

TEST_CASE(
  "Vector Ray Triangle Intersects produces same result",
  "[ray_triangle_intersects][RayTriangleIntersects][unit][util][cuda]"
)
{
  const unsigned ntest = 100;
  // long unsigned seed = static_cast<long unsigned int>(
  //   std::chrono::high_resolution_clock::now().time_since_epoch().count());
  // std::default_random_engine generator(seed);

  // std::random_device rd;
  // std::mt19937 generator(rd());

  std::default_random_engine generator;
  std::uniform_real_distribution<double> real (0.0, 1.0);

  std::vector<std::vector<double>> expected (5, std::vector<double>(3, 0.0));
  std::vector<std::vector<double>> test;
  double t_expected;
  double t_test;

  int nsame = 0;

  for (unsigned itest=0; itest<ntest; itest++) {
    for (unsigned idx=0; idx<expected.size(); idx++) {
      for (unsigned idy=0; idy<expected[idx].size(); idy++) {
        expected[idx][idy] = real(generator);
      }
    }
    test = expected;

    int res_expected = RayTriangleIntersects(
      expected[0].data(), expected[1].data(),
      expected[2].data(), expected[3].data(),
      expected[4].data(), &t_expected
    );
    int res_test = ValoMC::util::ray_triangle_intersects(
      test[0].data(), test[1].data(),
      test[2].data(), test[3].data(),
      test[4].data(), &t_test
    );

    if (res_expected == res_test && t_expected == t_test) {
      nsame++;
    }
    t_expected = 0.0;
    t_test = 0.0;
  }
  REQUIRE(nsame == ntest);
}


TEMPLATE_TEST_CASE(
  "check_ptr determines if pointer was allocated on device",
  "[unit][util][cuda][check_ptr]",
  float, Array<float>
)
{
  TestType* ptr;
  CHECK(ValoMC::util::check_device_ptr(ptr) == false);

  TestType* ptr_d;
  gpuErrchk(cudaMalloc((void**)&ptr_d, sizeof(TestType)));
  CHECK(ValoMC::util::check_device_ptr(ptr_d) == true);
  gpuErrchk(cudaFree(ptr_d));

  TestType* ptr_h;
  gpuErrchk(cudaMallocHost((void**)&ptr_h, sizeof(TestType)));
  CHECK(ValoMC::util::check_device_ptr(ptr_h) == false);
  gpuErrchk(cudaFreeHost(ptr_h));

  gpuErrchk(cudaDeviceSynchronize());
}

class ContainsArray {
public:
  ContainsArray () {
    // std::cerr << "ContainsArray::ContainsArray" << std::endl;
    is_allocated = false;
  }

  ~ContainsArray () {
    // std::cerr << "ContainsArray::~ContainsArray" << std::endl;
  }

  void allocate () {
    gpuErrchk(cudaMalloc((void**)&arr, sizeof(Array<double>)));
    is_allocated = true;
  }

  void destroy () {
    // std::cerr << "ContainsArray::destroy" << std::endl;
    if (is_allocated) {
      gpuErrchk(cudaFree(arr));
      is_allocated = false;
    }
  }

  Array<double>* arr;

private:

  bool is_allocated;
};

// class PseudoContainsArray {
//   char buffer[sizeof(ContainsArray)];
// };
//
//
// __global__ void iter_obj (PseudoContainsArray _con, int* result)
// {
//   ContainsArray &con = *((ContainsArray *)&_con);
//
//   const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
//   Array<double> arr = *(con.arr);
//   if (idx == 0) {
//     for (unsigned istep=0; istep<arr.N; istep++) {
//       arr[istep];
//       *result += 1;
//     }
//   }
// }

__global__ void iter_obj (ContainsArray con, int* result)
{
  const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  Array<double> arr = *(con.arr);
  if (idx == 0) {
    for (unsigned istep=0; istep<arr.N; istep++) {
      arr[istep];
      *result += 1;
    }
  }
}


template<typename T>
__global__ void iter_array (Array<T>* arr, int* result) {
  const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx == 0) {
    for (unsigned istep=0; istep<arr->N; istep++) {
      (*arr)[istep];
      *result += 1;
    }
  }
}

template<typename T>
__global__ void fill_array (Array<T>* arr, T val) {
  const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  // printf("fill_array: val=%f\n", val);
  if (idx > arr->N) {
    return;
  }
  for (unsigned istep=idx; istep<arr->N; istep++) {
    (*arr)[istep] = val;
  }
}

TEMPLATE_TEST_CASE(
  "h2d works as expected",
  "[util][unit][h2d][cuda]",
  float, double
)
{
  Array<TestType> arr_h;
  arr_h.resize(10, 2);
  for (unsigned idx=0; idx<arr_h.Nx; idx++) {
    for (unsigned idy=0; idy<arr_h.Ny; idy++) {
      arr_h(idx, idy) = static_cast<TestType>(idx + arr_h.Nx*idy);
    }
  }

  Array<TestType>* arr;
  gpuErrchk(cudaMalloc((void**)&arr, sizeof(Array<TestType>)));

  SECTION ("h2d works in isolation") {
    ValoMC::util::h2d(arr, &arr_h);
    gpuErrchk(cudaDeviceSynchronize());
  }

  SECTION ("can pass Array pointer to CUDA kernel") {
    ValoMC::util::h2d(arr, &arr_h);
    gpuErrchk(cudaDeviceSynchronize());
    int* result;
    int result_h = 0;
    gpuErrchk(cudaMalloc((void**)&result, sizeof(int)));
    iter_array<TestType><<<1, 1>>>(arr, result);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(&result_h, result, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(result));
    REQUIRE(result_h == 20);
  }
  gpuErrchk(cudaFree(arr));
  gpuErrchk(cudaDeviceSynchronize());
}

TEMPLATE_TEST_CASE(
  "d2h works as expected",
  "[util][unit][d2h][cuda]",
  float, double
)
{
  TestType val = 1.0;
  Array<TestType> arr_h;
  arr_h.resize(10, 2);

  Array<TestType>* arr;
  cudaMalloc((void**)&arr, sizeof(Array<TestType>));
  ValoMC::util::h2d(arr, &arr_h);

  fill_array<TestType><<<1, 1>>>(arr, val);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
  ValoMC::util::d2h(&arr_h, arr);

  bool all_close = true;

  for (unsigned idx=0; idx<arr_h.N; idx++) {
    if (arr_h[idx] != val) {
      all_close = false;
    }
  }
  REQUIRE(all_close == true);

  gpuErrchk(cudaFree(arr));
  gpuErrchk(cudaDeviceSynchronize());
}

TEST_CASE (
  "Can call kernel on object containing pointers to Arrays",
  "[util][ContainsArray][cuda]"
)
{
  ContainsArray con;
  Array<double> arr_h;
  arr_h.resize(10, 2);
  for (unsigned idx=0; idx<arr_h.N; idx++) {
    arr_h[idx] = 1.0;
  }
  con.allocate();
  ValoMC::util::h2d(con.arr, &arr_h);
  int* result;
  int result_h = 0.0;
  gpuErrchk(cudaMalloc((void**)&result, sizeof(int)));
  // iter_obj<<<1, 1>>>(*(PseudoContainsArray* )&con, result);
  iter_obj<<<1, 1>>>(con, result);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaMemcpy(&result_h, result, sizeof(int), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(result));
  gpuErrchk(cudaDeviceSynchronize());
  REQUIRE(result_h == 20);
  con.destroy();
}

// TEST_CASE("Random number generators produce same statistics", "[unit][util][cuda][random]")
// {
//   unsigned nsamples = 100;
//   curandState_t state;
//   util::invoke_init_random(1024, 1, 0, state);
//   std::vector<double> cuda_samples(nsamples);
//
//   SECTION("Uniform Closed") {
//     util::invoke_uniform_closed(state, cuda_samples);
//     double mean = std::accumulate(
//       cuda_samples.begin(), cuda_samples.end(), 0.0) / (double) nsamples;
//     std::cerr << "mean=" << mean << std::endl;
//     // for (unsigned idx=0; idx<cuda_samples.size(); idx++) {
//     //   std::cerr << cuda_samples[idx] << " ";
//     // }
//     // std::cerr << std::endl;
//   }
//
//   // SECTION("Uniform Open") {
//   //
//   // }
//   //
//   // SECTION("Uniform Half Upper") {
//   //
//   // }
//   //
//   // SECTION("Uniform Half Lower") {
//   //
//   // }
// }
