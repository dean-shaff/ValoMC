#include <iostream>
#include <fstream>
#include <string>

#include "catch.hpp"
#include "json.hpp"

#include "MC3D.hpp"
#include "Array.hpp"

using json = nlohmann::json;

template<typename Type>
using vec2d = std::vector<std::vector<Type>>;

template<typename Type>
void flatten(const vec2d<Type>& in, std::vector<Type>& out) {
  for (unsigned idx=0; idx<in.size(); idx++) {
    out.insert(out.end(), in[idx].begin(), in[idx].end());
  }
}


template<typename Type>
void load_array_from_vector (
  const vec2d<Type>& vec, Array<Type>& arr
)
{
  std::vector<Type> vec_flat;
  flatten(vec, vec_flat);

  arr.data = vec_flat.data();
  arr.Nx = vec.size();
  arr.Ny = vec[0].size();
  arr.Nz = 1;
  arr.Nxy = arr.Nx * arr.Ny;
  arr.N = arr.Nx * arr.Ny * arr.Nz;
  arr.IsRef = 1;
  arr.rank = 2;
}

template<typename Type>
void load_array_from_json(
  const json& js_obj,
  Array<Type>& arr
)
{
  vec2d<Type> vec = js_obj.get<vec2d<Type>>();
  load_array_from_vector(vec, arr);
}



void init_MC3D_from_json(const std::string& file_path, MC3D& mc3d)
{
  std::ifstream ifs(file_path);
  json js_obj;
  ifs >> js_obj;
  ifs.close();
  for (auto it = js_obj.begin(); it != js_obj.end(); ++it)
  {
    std::cerr  << it.key() << std::endl;
  }

// BCIntensity
// BCLightDirection
// BCLightDirectionType
// BCType
// BCn
// BH
// H
// HN
// Nphoton
// disable_pbar
// f
// g
// mua
// mus
// n
// phase0
// r
// rnseed
  load_array_from_json(js_obj["H"], mc3d.H);
  // load_array_from_json(js_obj["BH"], mc3d.BH);
  // load_array_from_json(js_obj["HN"], mc3d.HN);
  // load_array_from_json(js_obj["BCType"], mc3d.BCType);
  // load_array_from_json(js_obj["BCLightDirectionType"], mc3d.BCLightDirectionType);
  // load_array_from_json(js_obj["BCIntensity"], mc3d.BCIntensity);
  // load_array_from_json(js_obj["BCn"], mc3d.BCn);
  // load_array_from_json(js_obj["r"], mc3d.r);
  // load_array_from_json(js_obj["mua"], mc3d.mua);
  // load_array_from_json(js_obj["mus"], mc3d.mus);
  // load_array_from_json(js_obj["g"], mc3d.g);
  // load_array_from_json(js_obj["n"], mc3d.n);

}




TEST_CASE ("MC3D", "[unit]")
{
  const std::string json_file_path = "/home/dean/work/freelance/Yoav-Mor_28-01-2020_CUDA-Matlab-ValoMC/ValoMC/cpp/test/MC3Dmex.input.json";

  MC3D mc3d;

  init_MC3D_from_json(
    json_file_path,
    mc3d
  );

}
