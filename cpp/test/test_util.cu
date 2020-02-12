#include <random>
#include <vector>
#include <numeric>

#include "catch.hpp"

#include "MC3D_util.cuh"
#include "MC3D.hpp"

TEST_CASE("Vector Ray Triangle Intersects produces same result", "[unit][util][cuda]")
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
