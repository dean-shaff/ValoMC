if (HAVE_CUDA)
  set(bench_INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/cpp
  )

  set(bench_SOURCES
    ${CMAKE_SOURCE_DIR}/cpp/test/util.cpp
    bench.cu
  )

  add_executable(bench ${bench_SOURCES})
  target_include_directories(bench PUBLIC ${mc3d_INCLUDE_DIRECTORIES} ${bench_INCLUDE_DIRECTORIES})
  target_link_libraries(bench PUBLIC ${mc3d_LINK_LIBRARIES} mc3d)
  target_compile_options(bench PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
endif()
