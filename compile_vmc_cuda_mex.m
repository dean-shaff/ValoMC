
% mexcuda('-c', 'cpp/3d/MC3D_cuda.cu', 'NVCCFLAGS=-Xcompiler -fPIC -dc')
% mexcuda('-c', 'cpp/3d/MC3D_util.cu', 'NVCCFLAGS=-Xcompiler -fPIC -dc')
% mexcuda('-c', 'cpp/3d/MC3D_cuda_bridge.cu', 'NVCCFLAGS=-Xcompiler -fPIC -dc')
% mexcuda('-c', 'cpp/3d/MC3Dmex.cpp', 'NVCCFLAGS=-dc')
system('nvcc  -Xcompiler -fPIC -dlink -o device_link.o MC3D_cuda.o MC3D_util.o MC3D_cuda_bridge.o');
mexcuda('MC3Dmex.o', 'MC3D_cuda.o', 'MC3D_util.o', 'MC3D_cuda_bridge.o', 'device_link.o')

% mexcuda('cpp/3d/MC3Dmex.cpp', 'MC3D_cuda.o', 'MC3D_util.o', 'MC')

% mex cpp/3d/MC3Dmex.cpp COMPFLAGS='$COMPFLAGS -O3'
