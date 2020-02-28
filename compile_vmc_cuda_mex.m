suffix='o'

mc3d_mex_obj = sprintf('MC3Dmex.%s', suffix);
mc3d_cuda_obj = sprintf('MC3D_cuda.%s', suffix);
mc3d_util_obj = sprintf('MC3D_util.%s', suffix);
mc3d_cuda_bridge_obj = sprintf('MC3D_cuda_bridge.%s', suffix);
device_link_obj = sprintf('device_link.%s', suffix);
nvcc_cmd_str=sprintf('nvcc  -Xcompiler -fPIC -dlink -o %s %s %s %s', ...
  device_link_obj, mc3d_util_obj, mc3d_cuda_obj, mc3d_cuda_bridge_obj);

% build individual object files
mexcuda('-c', 'cpp/3d/MC3D_cuda.cu', 'NVCCFLAGS=-Xcompiler -fPIC -dc')
mexcuda('-c', 'cpp/3d/MC3D_util.cu', 'NVCCFLAGS=-Xcompiler -fPIC -dc')
mexcuda('-c', 'cpp/3d/MC3D_cuda_bridge.cu', 'NVCCFLAGS=-Xcompiler -fPIC -dc')
mexcuda('-c', 'cpp/3d/MC3Dmex.cpp', 'NVCCFLAGS=-dc')

% link everything with nvcc
system(nvcc_cmd_str);

% finally build MC3Dmex library 
mexcuda(mc3d_cuda_obj, mc3d_util_obj, mc3d_cuda_obj, mc3d_cuda_bridge_obj, device_link_obj)
