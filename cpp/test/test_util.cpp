#include <random>
#include <vector>

#include "catch.hpp"

#include "MC3D_cuda.hpp"
#include "MC3D.hpp"

TEST_CASE("Vector Ray Triangle Intersects produces same result")
{
  std::default_random_engine generator;
  std::uniform_real_distribution<double> real (0.0, 1.0);

  std::vector<std::vector<double>> expected (5, std::vector<double>(3, 0.0));
  for (unsigned idx=0; idx<expected.size(); idx++) {
    for (unsigned idy=0; idy<expected[idx].size(); idy++) {
      expected[idx][idy] = real(generator);
    }
  }
  std::vector<std::vector<double>> real = expected;

  double t_expected;
  double t_test;

  int res_expected = RayTriangleIntersects(
    expected[0].data(), expected[1].data(),
    expected[2].data(), expected[3].data(),
    expected[4].data(), &t_expected
  );
  int res_test = util::ray_triangle_intersects(
    test[0].data(), test[1].data(),
    test[2].data(), test[3].data(),
    test[4].data(), &t_test
  );

  REQUIRE(res_expected == res_test);
  REQUIRE(t_expected == t_test);

}

TEST_CASE("Random number generators produce same statistics")
{


  SECTION("Uniform Closed") {

  }

  SECTION("Uniform Open") {

  }

  SECTION("Uniform Half Upper") {

  }

  SECTION("Uniform Half Lower") {

  }
}
