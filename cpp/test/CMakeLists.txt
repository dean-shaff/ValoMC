set(CPP_3D_DIR
  ${CMAKE_SOURCE_DIR}/cpp/3d
)

set(TEST_MAIN_INCLUDE_DIRECTORIES
  ${CPP_3D_DIR}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(TEST_LIB_SOURCES
  ${CPP_3D_DIR}/MC3D_util.cu
  ${CPP_3D_DIR}/MC3D_cuda.cu
)

set(TEST_SOURCES
  util.cpp
  test_util.cu
  test_array.cpp
  test_MC3D.cpp
  test_MC3D_cuda.cu
  test_main.cpp
)

# add_library(test_lib SHARED ${TEST_LIB_SOURCES})
# target_include_directories(test_lib PUBLIC ${TEST_MAIN_INCLUDE_DIRECTORIES})
# target_link_libraries(test_lib PUBLIC ${CUDART_LIBRARY})
# target_compile_options(test_lib PUBLIC -G -g -dc -arch=sm_61)

add_executable(test_main ${TEST_LIB_SOURCES} ${TEST_SOURCES})
target_include_directories(test_main PUBLIC ${TEST_MAIN_INCLUDE_DIRECTORIES})
target_link_libraries(test_main PUBLIC ${CUDART_LIBRARY})
# target_link_libraries(test_main PUBLIC -arch=sm_61)
target_compile_options(test_main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -g -dc>)
# target_compile_options(test_main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -g -dc -arch=sm_61>)
# target_link_options(test_main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_61>)
