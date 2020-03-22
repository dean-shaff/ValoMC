comp_flags='COMPFLAGS=\$COMPFLAGS -fopenmp';
cxx_flags='CXXFLAGS=\$CXXFLAGS -fopenmp';
ld_flags='LDFLAGS=\$LDFLAGS -fopenmp';
if ispc
  % the following line may need to change to point to the location of the
  % directory containing the nvcc executable.
  comp_flags='COMPFLAGS=\$COMPFLAGS /openmp /O2';
  cxx_flags='CXXFLAGS=\$CXXFLAGS';
  ld_flags='LDFLAGS=\$LDFLAGS';
end
% mex('-c', 'cpp/3d/MC3Dmex.cpp', 'NVCC_FLAGS=-dc', '-DHAVE_CUDA', comp_flags, cxx_flags, '-DUSE_OMP');

mex('cpp/2d/MC2Dmex.cpp', comp_flags, cxx_flags, '-DUSE_OMP');
mex('cpp/3d/MC3Dmex.cpp', comp_flags, cxx_flags, '-DUSE_OMP');
mex('cpp/3d/createBH3mex.cpp', comp_flags, cxx_flags, '-DUSE_OMP');
