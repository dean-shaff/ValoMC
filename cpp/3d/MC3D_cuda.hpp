#ifndef __MC3D_cuda_hpp
#define __MC3D_cuda_hpp

namespace util {
  int invoke_ray_triangle_intersects (
    double O[3],
    double D[3],
    double V0[3],
    double V1[3],
    double V2[3],
    double *t
  );

  void invoke_cross (double dest[3], double v1[3], double v2[3]);

  double invoke_dot (double v1[3], double v2[3]);

  void invoke_sub (double dest[3], double v1[3], double v2[3]);

  template<typename StateType>
  double invoke_uniform_closed (StateType* state);

  template<typename StateType>
  double invoke_uniform_open (StateType* state);

  template<typename StateType>
  double invoke_uniform_half_upper (StateType* state);

  template<typename StateType>
  double invoke_uniform_half_lower (StateType* state);
}


#endif
