suffix='o';
comp_flags='COMPFLAGS=\$COMPFLAGS -fopenmp';
cxx_flags='CXXFLAGS=\$CXXFLAGS -fopenmp';
ld_flags='LDFLAGS=\$LDFLAGS -fopenmp';

if ispc
  % the following line may need to change to point to the location of the
  % directory containing the nvcc executable.
  setenv('MW_NVCC_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin');
  suffix='obj';
  comp_flags='COMPFLAGS=\$COMPFLAGS /openmp /O2';
  cxx_flags='CXXFLAGS=\$CXXFLAGS';
  ld_flags='LDFLAGS=\$LDFLAGS';
end

mc3d_mex_obj = sprintf('MC3Dmex.%s', suffix);
mc3d_cuda_obj = sprintf('MC3D_cuda.%s', suffix);
mc3d_util_obj = sprintf('MC3D_util.%s', suffix);
mc3d_cuda_bridge_obj = sprintf('MC3D_cuda_bridge.%s', suffix);
device_link_obj = sprintf('device_link.%s', suffix);

nvcc_cmd_str=sprintf('nvcc -dlink -o %s %s %s %s', ...
  device_link_obj, mc3d_util_obj, mc3d_cuda_obj, mc3d_cuda_bridge_obj);

% build individual object files
fprintf('Compiling %s\n', mc3d_cuda_obj);
mexcuda('-c', 'cpp/3d/MC3D_cuda.cu', 'NVCC_FLAGS=-dc');
fprintf('Compiling %s\n', mc3d_util_obj);
mexcuda('-c', 'cpp/3d/MC3D_util.cu', 'NVCC_FLAGS=-dc');
fprintf('Compiling %s\n', mc3d_cuda_bridge_obj);
mexcuda('-c', 'cpp/3d/MC3D_cuda_bridge.cu', 'NVCC_FLAGS=-dc');
fprintf('Compiling %s\n', mc3d_mex_obj);
mex('-c', 'cpp/3d/MC3Dmex.cpp', 'NVCC_FLAGS=-dc', '-DHAVE_CUDA', comp_flags, cxx_flags, '-DUSE_OMP');
% mexcuda('-c', 'cpp/3d/MC3Dmex.cpp', 'NVCC_FLAGS=-dc', '-DHAVE_CUDA', comp_flags, '-DUSE_OMP');

fprintf('Linking device code\n');
system(nvcc_cmd_str);

% finally build MC3Dmex library
fprintf('Creating mex file\n');
mexcuda(mc3d_mex_obj, mc3d_util_obj, mc3d_cuda_obj, mc3d_cuda_bridge_obj, device_link_obj, ld_flags, cxx_flags);
