#include <exception>

#include "MC3D_cuda_bridge.hpp"
#include "MC3D_cuda.cuh"

namespace ValoMC {

  template<typename T>
  void monte_carlo (MC3D<T>& mc3d, unsigned states_size, bool use_alt) {
    // first check if MC3D::Init has been called.
    // If it hasn't then things will not work properly down the line

    if (mc3d.LightSources.N == 0) {
      throw std::runtime_error("Make sure MC3D::Init has been called beforehand!");
    }

    MC3DCUDA<T> mc3dcuda (mc3d, states_size);
    mc3dcuda.allocate();
    mc3dcuda.h2d();
    mc3dcuda.monte_carlo(use_alt); // this is where the meat and potatoes happens!
    mc3dcuda.d2h();
  }

  template void monte_carlo<float> (MC3D<float>& mc3d, unsigned states_size, bool use_alt);
  template void monte_carlo<double> (MC3D<double>& mc3d, unsigned states_size, bool use_alt);

}
