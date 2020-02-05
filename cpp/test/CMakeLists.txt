
set(CPP_3D_DIR
  ${CMAKE_SOURCE_DIR}/cpp/3d
)

set(TEST_SOURCES
  test_MC3D.cpp
  test_main.cpp
)

add_executable(test_main ${TEST_SOURCES})
target_include_directories(test_main PUBLIC ${CPP_3D_DIR})
