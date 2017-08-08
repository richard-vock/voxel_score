cmake_minimum_required(VERSION 3.8.0)
project(voxel_score)

set (voxel_score_VERSION_MAJOR 0)
set (voxel_score_VERSION_MINOR 1)

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")

install (DIRECTORY include/ DESTINATION include/voxel_score)

if (NOT WIN32)
    file (GLOB find_modules RELATIVE "${PROJECT_SOURCE_DIR}" "VoxelScoreConfig.cmake")
    install (FILES ${find_modules} DESTINATION share/VoxelScore)
endif(NOT WIN32)
