#ifndef __bench_util_hpp
#define __bench_util_hpp

#include <chrono>

namespace ValoMC {
namespace bench {
namespace util {

using duration = std::chrono::duration<double, std::ratio<1>>;

inline std::chrono::time_point<std::chrono::high_resolution_clock> now () {
  return std::chrono::high_resolution_clock::now();
}


}
}
}

#endif
