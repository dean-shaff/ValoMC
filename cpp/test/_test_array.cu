#include "GPUArray.cuh"

int main ()
{
  GPUArray<float> arr;
  arr.IsGPU = 1;
  arr.resize(100);
  std::cerr << "hello" << std::endl;

}
