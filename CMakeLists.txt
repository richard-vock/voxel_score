cmake_minimum_required(VERSION 3.8.0)
project(voxel_score)

set (voxel_score_VERSION_MAJOR 0)
set (voxel_score_VERSION_MINOR 1)

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")

install (DIRECTORY include/ DESTINATION include/voxel_score)
