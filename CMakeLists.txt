cmake_minimum_required (VERSION 3.5)
project (ValoMC)
find_package(OpenMP)

set (CMAKE_BUILD_TYPE "Release")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message("64 bit target platform detected.")
else()
    message(FATAL_ERROR "Did not detect a 64 bit target platform. Use \"cmake --help\" to list all available target platforms and select a 64 bit one using e.g. \n cmake . -G \"Visual Studio 15 2017 Win64\" ")
endif()

# for running CPP code in non matlab environment
find_package(Matlab COMPONENTS MEX_COMPILER MX_LIBRARY ENG_LIBRARY)

set(ValoMC_COMPILE_OPTIONS "")
set(ValoMC_LINK_OPTIONS "")

if (OPENMP_FOUND)
  set(ValoMC_COMPILE_OPTIONS
    ${OpenMP_CXX_FLAGS}
  )
  set(ValoMC_LINK_OPTIONS
    ${OpenMP_CXX_FLAGS}
    ${OpenMP_EXE_LINKER_FLAGS}
  )
  # message("ValoMC_LINK_OPTIONS=${ValoMC_LINK_OPTIONS}")
  # message("ValoMC_COMPILE_OPTIONS=${ValoMC_COMPILE_OPTIONS}")
  # set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  # set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  # set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  add_definitions(-DUSE_OMP)
endif()

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )

foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR} )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR} )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR} )
endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR})

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
	set(HAVE_CUDA 1)
else()
  message(STATUS "No CUDA compiler found")
  set(HAVE_CUDA 0)
endif()

set(CPP_3D_DIR
  ${CMAKE_SOURCE_DIR}/cpp/3d
)
set(mc3d_LINK_LIBRARIES "")
set(mc3d_INCLUDE_DIRECTORIES "")

if (HAVE_CUDA)
  find_library(CUDART_LIBRARY cudart HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

  set(mc3d_SOURCES
    ${CPP_3D_DIR}/MC3D_util.cu
    ${CPP_3D_DIR}/MC3D_cuda.cu
    ${CPP_3D_DIR}/MC3D_cuda_bridge.cu
  )

  set(mc3d_LINK_LIBRARIES
    ${CUDART_LIBRARY}
  )
  set(mc3d_INCLUDE_DIRECTORIES
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CPP_3D_DIR}
  )

  add_library(mc3d SHARED ${mc3d_SOURCES})
  target_include_directories(mc3d PUBLIC ${mc3d_INCLUDE_DIRECTORIES})
  target_link_libraries(mc3d PUBLIC ${mc3d_LINK_LIBRARIES} ${ValoMC_LINK_OPTIONS})
  set_property(TARGET mc3d PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(mc3d PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-dc>)

endif()

set(standalone_EXECUTABLES
  MC2D
  MC3D
)

set(MC2D_SOURCES
  cpp/2d/MC2D.cpp
)

set(MC3D_SOURCES
  cpp/3d/MC3D.cpp
)

foreach(exe ${standalone_EXECUTABLES})
  add_executable(${exe} ${${exe}_SOURCES})
  target_link_libraries(${exe} PUBLIC ${ValoMC_LINK_OPTIONS})
  target_compile_options(${exe} PUBLIC ${ValoMC_COMPILE_OPTIONS})
endforeach()

# add_subdirectory(cpp/test)
add_subdirectory(cpp/bench)

if (Matlab_FOUND)
  get_filename_component(Matlab_MX_LIBRARY_PATH ${Matlab_MX_LIBRARY} DIRECTORY)
  find_library(Matlab_UT_LIBRARY NAME libut.lib libut.so HINTS ${Matlab_MX_LIBRARY_PATH})

  set(MC2Dmex_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/2d/MC2Dmex.cpp
  )
  set(MC3Dmex_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/3d/MC3Dmex.cpp
  )
  set(createBH3_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/3d/createBH3mex.cpp
  )

  set(matlab_EXECUTABLES
    MC2Dmex
    MC3Dmex
    createBH3
  )

  matlab_get_mex_suffix(
    ${Matlab_ROOT_DIR}
    mex_suffix
  )

  foreach(exe ${matlab_EXECUTABLES})
    # matlab_add_mex(NAME ${exe} OUTPUT_NAME ${exe} SRC ${${exe}_SOURCES})
    add_library(${exe} SHARED ${${exe}_SOURCES})
    set_target_properties(${exe} PROPERTIES SUFFIX ".${mex_suffix}" PREFIX "")
    # can't use PRIVATE/PUBLIC with matlab_add_mex targets
    target_link_libraries(${exe} ${ValoMC_LINK_OPTIONS} ${Matlab_MX_LIBRARY})
    target_include_directories(${exe} PUBLIC ${Matlab_INCLUDE_DIRS})
    target_compile_options(${exe} PUBLIC ${ValoMC_COMPILE_OPTIONS})
    target_compile_definitions(${exe} PRIVATE MATLAB_MEXCMD_RELEASE=R2017b USE_MEX_CMD MATLAB_MEX_FILE)
    if (Matlab_UT_LIBRARY)
      target_link_libraries(${exe} ${Matlab_UT_LIBRARY})
      target_compile_definitions(${exe} PRIVATE _MAKE_CTRL_C_POSSIBLE_)
    endif()
  endforeach()

  if (HAVE_CUDA)
    target_compile_options(MC3Dmex PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-dc>)
    target_link_libraries(MC3Dmex ${mc3d_LINK_LIBRARIES} mc3d)
  endif()
endif()
