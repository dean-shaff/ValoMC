function compile_vmc_mex (use_omp)
  if ~exist('use_omp', 'var')
    use_omp=true;
  end
  if use_omp
    fprintf('Compiling with OpenMP flags\n');
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
  else
    fprintf('Compiling without OpenMP flags\n');
    mex('cpp/2d/MC2Dmex.cpp');
    mex('cpp/3d/MC3Dmex.cpp');
    mex('cpp/3d/createBH3mex.cpp');
  end
end
