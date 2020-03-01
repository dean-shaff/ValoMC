if (HAVE_CUDA)
  set(bench_INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/cpp
  )

  set(bench_SOURCES
    ${CPP_3D_DIR}/MC3D_util.cu
    ${CPP_3D_DIR}/MC3D_cuda.cu
    ${CPP_3D_DIR}/MC3D_cuda_bridge.cu
    ${CMAKE_SOURCE_DIR}/cpp/test/util.cpp
    bench.cu
  )

  # add_executable(bench ${bench_SOURCES})
  # target_include_directories(bench PUBLIC ${mc3d_INCLUDE_DIRECTORIES} ${bench_INCLUDE_DIRECTORIES})
  # set_property(TARGET bench PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  # # target_link_libraries(bench PUBLIC ${mc3d_LINK_LIBRARIES} mc3d)
  # target_link_libraries(bench PUBLIC ${mc3d_LINK_LIBRARIES})
  # target_compile_options(bench PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
  # # target_compile_options(bench PUBLIC $<$<COMPILE_LANGUAGE:CXX>:O2>)


  set(array_INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/cpp
  )

  set(array_SOURCES
    ${CMAKE_SOURCE_DIR}/cpp/3d/MC3D_util.cu
    array.cu
  )

  add_executable(array ${array_SOURCES})
  target_include_directories(array PUBLIC ${array_INCLUDE_DIRECTORIES})
  target_link_libraries(array PUBLIC ${mc3d_LINK_LIBRARIES})
  target_compile_options(array PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
  # target_compile_options(array PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-O3>)

endif()
