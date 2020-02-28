% setenv('MW_NVCC_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin')

suffix='o'

mc3d_mex_obj = sprintf('MC3Dmex.%s', suffix);
mc3d_cuda_obj = sprintf('MC3D_cuda.%s', suffix);
mc3d_util_obj = sprintf('MC3D_util.%s', suffix);
mc3d_cuda_bridge_obj = sprintf('MC3D_cuda_bridge.%s', suffix);
device_link_obj = sprintf('device_link.%s', suffix);
nvcc_cmd_str=sprintf('nvcc -dlink -o %s %s %s %s', ...
  device_link_obj, mc3d_util_obj, mc3d_cuda_obj, mc3d_cuda_bridge_obj);

% build individual object files
% -Xcompiler -fPIC
mexcuda('-c', 'cpp/3d/MC3D_cuda.cu', 'NVCC_FLAGS=-dc')
mexcuda('-c', 'cpp/3d/MC3D_util.cu', 'NVCC_FLAGS=-dc')
mexcuda('-c', 'cpp/3d/MC3D_cuda_bridge.cu', 'NVCC_FLAGS=-dc')
mexcuda('-c', 'cpp/3d/MC3Dmex.cpp', 'NVCC_FLAGS=-dc')

% link everything with nvcc
system(nvcc_cmd_str);

% finally build MC3Dmex library
mexcuda(mc3d_mex_obj, mc3d_util_obj, mc3d_cuda_obj, mc3d_cuda_bridge_obj, device_link_obj)
