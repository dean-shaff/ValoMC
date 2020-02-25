set(TEST_LIB_SOURCES
  ${CPP_3D_DIR}/MC3D_util.cu
  ${CPP_3D_DIR}/MC3D_cuda.cu
  ${CPP_3D_DIR}/MC3D_cuda_bridge.cu
)

set(TEST_SOURCES
  util.cpp
  test_util.cu
  test_array.cpp
  test_MC3D.cpp
  test_MC3D_cuda.cu
  test_MC3D_cuda_bridge.cpp
  test_main.cpp
)

add_executable(test_main ${TEST_LIB_SOURCES} ${TEST_SOURCES})
target_include_directories(test_main PUBLIC ${mc3d_INCLUDE_DIRECTORIES})
target_link_libraries(test_main PUBLIC ${mc3d_LINK_LIBRARIES} ${ValoMC_LINK_OPTIONS})
# target_compile_options(test_main PUBLIC ${ValoMC_COMPILE_OPTIONS})
# target_link_libraries(test_main PUBLIC -arch=sm_61)
target_compile_options(test_main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -g -dc>)
