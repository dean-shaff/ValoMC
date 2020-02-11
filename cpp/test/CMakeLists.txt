set(CPP_3D_DIR
  ${CMAKE_SOURCE_DIR}/cpp/3d
)

set(TEST_MAIN_INCLUDE_DIRECTORIES
  ${CPP_3D_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)


set(TEST_SOURCES
  ${CPP_3D_DIR}/MC3D_kernels.cu
  ${CPP_3D_DIR}/MC3D_cuda.cu
  test_util.cpp
  # test_MC3D.cpp
  test_main.cpp
)

add_executable(test_main ${TEST_SOURCES})
target_include_directories(test_main PUBLIC ${TEST_MAIN_INCLUDE_DIRECTORIES})
target_link_libraries(test_main PUBLIC ${CUDART_LIBRARY})
target_compile_options(test_main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -g -dc>)
# target_compile_options(test_main PUBLIC -G -g)
