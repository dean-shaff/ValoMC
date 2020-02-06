#ifndef __test_util_hpp
#define __test_util_hpp

#include <ctime>
#include <cmath>

#include "json.hpp"

#include "Array.hpp"

using json = nlohmann::json;

namespace ValoMC {
namespace test {
namespace util {

template<typename Type>
using vec2d = std::vector<std::vector<Type>>;

template<typename Type>
void flatten(const vec2d<Type>& in, std::vector<Type>& out) {
  for (unsigned idx=0; idx<in.size(); idx++) {
    out.insert(out.end(), in[idx].begin(), in[idx].end());
  }
}

template<typename Type>
void flatten_transpose(const vec2d<Type>& in, std::vector<Type>& out) {
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



template<typename Type>
void load_array_from_vector (
  const vec2d<Type>& vec, Array<Type>& arr
)
{
  if (vec.size() == 0) {
    return;
  }
  std::vector<Type> vec_flat;
  ValoMC::test::util::flatten_transpose(vec, vec_flat);

  Type* data = arr.resize(vec.size(), vec[0].size());
  std::copy(vec_flat.data(), vec_flat.data() + arr.N, data);
}


template<typename Type>
void load_array_from_json(
  const json& js_obj,
  Array<Type>& arr
)
{
  vec2d<Type> vec = js_obj.get<vec2d<Type>>();
  ValoMC::test::util::load_array_from_vector(vec, arr);
}


// have to use template here because MC3D cannot be included in multiple files
template<typename ValoMCType>
void init_MC3D_from_json(const std::string& file_path, ValoMCType& mc_obj)
{
  std::ifstream ifs(file_path);
  json js_obj;
  ifs >> js_obj;
  ifs.close();
  // for (auto it = js_obj.begin(); it != js_obj.end(); ++it)
  // {
  //   std::cerr  << it.key() << std::endl;
  // }

  // Order in which to load variables:
  // H
  // HN
  // BH

  // r
  // mua
  // mus
  // g
  // n

  // BCType
  // BCIntensity
  // BCLightDirectionType
  // BCLightDirection -> BCLNormal
  // BCn

  // f
  // phase0
  // Nphoton

  // rnseed

  // disable_pbar
  // vec2d<unsigned> H_vec = js_obj["H"].get<vec2d<unsigned>>();
  // for (unsigned idx=0; idx<10; idx++) {
  //   for (unsigned idy=0; idy<H_vec[0].size(); idy++) {
  //     std::cerr << H_vec[idx][idy] << " ";
  //   }
  //   std::cerr << std::endl;
  // }

  ValoMC::test::util::load_array_from_json(js_obj["H"], mc_obj.H);
  ValoMC::test::util::load_array_from_json(js_obj["HN"], mc_obj.HN);
  ValoMC::test::util::load_array_from_json(js_obj["BH"], mc_obj.BH);

  ValoMC::test::util::load_array_from_json(js_obj["r"], mc_obj.r);
  ValoMC::test::util::load_array_from_json(js_obj["mua"], mc_obj.mua);
  ValoMC::test::util::load_array_from_json(js_obj["mus"], mc_obj.mus);
  ValoMC::test::util::load_array_from_json(js_obj["g"], mc_obj.g);
  ValoMC::test::util::load_array_from_json(js_obj["n"], mc_obj.n);

  ValoMC::test::util::load_array_from_json(js_obj["BCType"], mc_obj.BCType);
  ValoMC::test::util::load_array_from_json(js_obj["BCIntensity"], mc_obj.BCIntensity);
  ValoMC::test::util::load_array_from_json(js_obj["BCLightDirectionType"], mc_obj.BCLightDirectionType);
  ValoMC::test::util::load_array_from_json(js_obj["BCLightDirection"], mc_obj.BCLNormal);
  ValoMC::test::util::load_array_from_json(js_obj["BCn"], mc_obj.BCn);

  vec2d<double> f_vec = js_obj["f"].get<vec2d<double>>();
  mc_obj.f = f_vec[0][0];

  vec2d<double> phase0_vec = js_obj["phase0"].get<vec2d<double>>();
  mc_obj.phase0 = phase0_vec[0][0];
  if(mc_obj.phase0 < 0) {
    mc_obj.phase0 += 2*M_PI*std::ceil(-mc_obj.phase0 / (2*M_PI));
  }

  vec2d<double> Nphoton_vec = js_obj["Nphoton"].get<vec2d<double>>();
  mc_obj.Nphoton = Nphoton_vec[0][0];

  vec2d<unsigned long> rnseed_vec = js_obj["rnseed"].get<vec2d<unsigned long>>();

  if ((bool) rnseed_vec[0][1]) {
     mc_obj.seed = (unsigned long) rnseed_vec[0][0];
  } else {
     mc_obj.seed = (unsigned long) std::time(NULL);
  }

}


}
}
}


#endif
