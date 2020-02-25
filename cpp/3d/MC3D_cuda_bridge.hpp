#ifndef __MC3D_cuda_bridge_hpp
#define __MC3D_cuda_bridge_hpp

#include "MC3D.hpp"

namespace ValoMC {

/**
 * Do monte_carlo simulation entirely from C++ land. This will do all the
 * data transfers to and from the GPU, so if multiple runs are desired, this
 * is not the function to use.
 * @param mc3d MC3D object used to construct MC3DCUDA object in implementation
 * @param states_size number of curandState_t states to allocate
 */
void monte_carlo (MC3D& mc3d, unsigned states_size=1000) ;


}



#endif
