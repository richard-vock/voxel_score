###############################################################################
# Find VoxelScore
#
# This sets the following variables:
# VOXEL_SCORE_FOUND - True if VOXEL_SCORE was found.
# VOXEL_SCORE_INCLUDE_DIRS - Directories containing the VOXEL_SCORE include files.
# VOXEL_SCORE_LIBRARY_DIRS - Directories containing the VOXEL_SCORE library.
# VOXEL_SCORE_LIBRARIES - VOXEL_SCORE library files.

if(WIN32)
    find_path(VOXEL_SCORE_INCLUDE_DIR voxel_score PATHS "/usr/include" "/usr/local/include" "/usr/x86_64-w64-mingw32/include" "$ENV{PROGRAMFILES}" NO_DEFAULT_PATHS)
else(WIN32)
    find_path(VOXEL_SCORE_INCLUDE_DIR voxel_score PATHS "/usr/include" "/usr/local/include" "$ENV{PROGRAMFILES}" NO_DEFAULT_PATHS)
endif(WIN32)

set(VOXEL_SCORE_INCLUDE_DIRS ${VOXEL_SCORE_INCLUDE_DIR})
set(VOXEL_SCORE_LIBRARY_DIRS "")
set(VOXEL_SCORE_LIBRARIES "")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VOXEL_SCORE DEFAULT_MSG VOXEL_SCORE_INCLUDE_DIR VOXEL_SCORE_LIBRARY VOXEL_SCORE_LIBRARY_DIR)

mark_as_advanced(VOXEL_SCORE_INCLUDE_DIR)
mark_as_advanced(VOXEL_SCORE_LIBRARY_DIR)
mark_as_advanced(VOXEL_SCORE_LIBRARY)
mark_as_advanced(VOXEL_SCORE_LIBRARY_PATH)
