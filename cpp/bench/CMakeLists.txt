set(BENCH_INCLUDE_DIRECTORIES
  ${CPP_3D_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_SOURCE_DIR}/cpp
)


set(BENCH_SOURCES
  ${CPP_3D_DIR}/MC3D_util.cu
  ${CPP_3D_DIR}/MC3D_cuda.cu
  ${CMAKE_SOURCE_DIR}/cpp/test/util.cpp
  bench.cu
)

add_executable(bench ${BENCH_SOURCES})
target_include_directories(bench PUBLIC ${BENCH_INCLUDE_DIRECTORIES})
target_link_libraries(bench PUBLIC ${CUDART_LIBRARY})
target_compile_options(bench PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
