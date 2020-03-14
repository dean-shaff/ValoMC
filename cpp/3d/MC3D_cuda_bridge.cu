#include <exception>

#include "MC3D_cuda_bridge.hpp"
#include "MC3D_cuda.cuh"

namespace ValoMC {

  void monte_carlo (MC3D<double>& mc3d, unsigned states_size) {
    // first check if MC3D::Init has been called.
    // If it hasn't then things will not work properly down the line

    if (mc3d.LightSources.N == 0) {
      throw std::runtime_error("Make sure MC3D::Init has been called beforehand!");
    }

    MC3DCUDA mc3dcuda (mc3d, states_size);
    mc3dcuda.allocate();
    mc3dcuda.h2d();
    mc3dcuda.monte_carlo(); // this is where the meat and potatoes happens!
    mc3dcuda.d2h();
  }


}
