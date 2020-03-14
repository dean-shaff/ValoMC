#include <fstream>

#include "json.hpp"

#include "util.hpp"

using json = nlohmann::json;

namespace ValoMC {
namespace test {
namespace util {

template<typename T>
void TestConfig<T>::init_MC3D_from_json()
{
  if (loaded) {
    return;
  }
  std::ifstream ifs(test_data_file_path);
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
  // vec2d<unsigned> H_vec = js_obj["H"].template get<vec2d<unsigned>>();
  // for (unsigned idx=0; idx<10; idx++) {
  //   for (unsigned idy=0; idy<H_vec[0].size(); idy++) {
  //     std::cerr << H_vec[idx][idy] << " ";
  //   }
  //   std::cerr << std::endl;
  // }

  load_array_from_json(js_obj["H"], mc3d.H);
  load_array_from_json(js_obj["HN"], mc3d.HN);
  load_array_from_json(js_obj["BH"], mc3d.BH);

  load_array_from_json(js_obj["r"], mc3d.r);
  load_array_from_json(js_obj["mua"], mc3d.mua);
  load_array_from_json(js_obj["mus"], mc3d.mus);
  load_array_from_json(js_obj["g"], mc3d.g);
  load_array_from_json(js_obj["n"], mc3d.n);

  load_array_from_json(js_obj["BCType"], mc3d.BCType);
  load_array_from_json(js_obj["BCIntensity"], mc3d.BCIntensity);
  load_array_from_json(js_obj["BCLightDirectionType"], mc3d.BCLightDirectionType);
  load_array_from_json(js_obj["BCLightDirection"], mc3d.BCLNormal);
  load_array_from_json(js_obj["BCn"], mc3d.BCn);

  vec2d<double> f_vec = js_obj["f"].template get<vec2d<double>>();
  mc3d.f = f_vec[0][0];

  vec2d<double> phase0_vec = js_obj["phase0"].template get<vec2d<double>>();
  mc3d.phase0 = phase0_vec[0][0];
  if(mc3d.phase0 < 0) {
    mc3d.phase0 += 2*M_PI*std::ceil(-mc3d.phase0 / (2*M_PI));
  }

  vec2d<double> Nphoton_vec = js_obj["Nphoton"].template get<vec2d<double>>();
  mc3d.Nphoton = Nphoton_vec[0][0];

  vec2d<unsigned long> rnseed_vec = js_obj["rnseed"].template get<vec2d<unsigned long>>();

  if ((bool) rnseed_vec[0][1]) {
    mc3d.seed = (unsigned long) rnseed_vec[0][0];
  } else {
    mc3d.seed = (unsigned long) std::time(NULL);
  }
  loaded = true;
}

template class TestConfig<float>;
template class TestConfig<double>;

}
}
}
