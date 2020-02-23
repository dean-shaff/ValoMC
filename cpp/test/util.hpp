#ifndef __test_util_hpp
#define __test_util_hpp

#include <string>
#include <vector>
#include <ctime>
#include <cmath>

#include "catch.hpp"

#include "MC3D.hpp"
#include "Array.hpp"

CATCH_TRANSLATE_EXCEPTION( mcerror& ex ) {
  return std::string(errorstring(ex));
}

namespace ValoMC {
namespace test {
namespace util {

inline std::string get_env_var (
  const std::string& name,
  const std::string& default_val
)
{
 const char* env_var = std::getenv(name.c_str());
 if (env_var) {
   return std::string(env_var);
 } else {
   return std::string(default_val);
 }
}


template<typename T>
using vec2d = std::vector<std::vector<T>>;

template<typename T>
void flatten(const vec2d<T>& in, std::vector<T>& out) {
  for (unsigned idx=0; idx<in.size(); idx++) {
    out.insert(out.end(), in[idx].begin(), in[idx].end());
  }
}

template<typename T>
void flatten_transpose(const vec2d<T>& in, std::vector<T>& out) {
  unsigned rows = in.size();
  unsigned cols = in[0].size();
  out.resize(rows * cols);
  for (unsigned idx=0; idx<rows; idx++) {
    for (unsigned idy=0; idy<cols; idy++) {
      // out[idx*cols + idy] = in[idx][idy];
      out[idy*rows + idx] = in[idx][idy];
    }
  }
}




template<typename T>
void load_array_from_vector (
  const vec2d<T>& vec, Array<T>& arr
)
{
  if (vec.size() == 0) {
    return;
  }
  std::vector<T> vec_flat;
  ValoMC::test::util::flatten_transpose(vec, vec_flat);

  T* data = arr.resize(vec.size(), vec[0].size());
  std::copy(vec_flat.data(), vec_flat.data() + arr.N, data);
}


template<typename JsonType, typename T>
void load_array_from_json(
  const JsonType& js_obj,
  Array<T>& arr
)
{
  vec2d<T> vec = js_obj.template get<vec2d<T>>();
  ValoMC::test::util::load_array_from_vector(vec, arr);
}


void init_MC3D_from_json(const std::string& file_path, MC3D& mc_obj);


}
}
}


#endif
