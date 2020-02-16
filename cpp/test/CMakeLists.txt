set(CPP_3D_DIR
  ${CMAKE_SOURCE_DIR}/cpp/3d
)

set(TEST_MAIN_INCLUDE_DIRECTORIES
  ${CPP_3D_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)


set(TEST_SOURCES
  # ${CPP_3D_DIR}/MC3D_util.cu
  # test_util.cu
  test_array.cu
  # test_MC3D.cpp
  test_main.cpp
)

add_executable(test_main ${TEST_SOURCES})
target_include_directories(test_main PUBLIC ${TEST_MAIN_INCLUDE_DIRECTORIES})
target_link_libraries(test_main PUBLIC ${CUDART_LIBRARY})
target_compile_options(test_main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -g -dc>)

# add_executable(_test_array _test_array.cu)
# target_include_directories(_test_array PUBLIC ${TEST_MAIN_INCLUDE_DIRECTORIES})
# target_link_libraries(_test_array PUBLIC ${CUDART_LIBRARY})
# target_compile_options(_test_array PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -g -dc>)
