cmake_minimum_required(VERSION 3.8.0)
project(voxel_score)

set (voxel_score_VERSION_MAJOR 0)
set (voxel_score_VERSION_MINOR 1)

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_SOURCE_DIR}/cmake/modules)


## CUDA
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O2")
set (CUDA_ARCH "sm_61" CACHE STRING "Cuda Architecture")
find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-arch=${CUDA_ARCH};-O2;-Xcompiler;-fPIC;-DVERBOSE")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
cuda_include_directories(${CMAKE_SOURCE_DIR}/include)
set(CUDA_ARCH_BIN "61" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")
set(CUDA_HOST_COMPILER g++-5)
include(CudaComputeTargetFlags)
APPEND_TARGET_ARCH_FLAGS()
#if(CMAKE_COMPILER_IS_GNUCXX)
    #add_definitions(-fPIC)
    #add_definitions(-O3)
    #add_definitions(-g)
    #add_definitions(-std=c++11)
    #add_definitions(-Wall)
#endif()


find_package(PCL COMPONENTS common gpu_containers)
#if (OPTIONAL_FOUND)
    #add_definitions(-DUSE_OPTIONAL)
#else()
    #message(STATUS "No optional package found. optional support will be disabled.")
#endif()


file (GLOB_RECURSE cpp_files RELATIVE "${PROJECT_SOURCE_DIR}" "src/*.cpp")
file (GLOB_RECURSE cu_files RELATIVE "${PROJECT_SOURCE_DIR}" "cuda/*.cu")

if (CUDA_FOUND AND PCL_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${PCL_INCLUDE_DIRS})
#    if(OPTIONAL_FOUND)
#        include_directories(${OPTIONAL_INCLUDE_DIRS})
#    else()
#        set(OPTIONAL_LIBRARIES "")
#    endif()

    cuda_compile(cuda_objs ${cu_files})
    cuda_add_library(voxel_score SHARED ${cpp_files} ${cuda_objs})
    target_link_libraries(voxel_score ${PCL_LIBRARIES})

    add_executable(test_main "${PROJECT_SOURCE_DIR}/apps/main.cpp")
    target_link_libraries(test_main voxel_score)

	# install binary
    #install (TARGETS voxel_score DESTINATION lib)
	# install header
    #install (DIRECTORY include/ DESTINATION include/voxel_score)
endif()
