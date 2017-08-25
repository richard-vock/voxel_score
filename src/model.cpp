#include <model>
#include <impl/model.hpp>

namespace voxel_score {

namespace detail {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template mat3f_t pca<type>(typename pcl::PointCloud<type>::ConstPtr cloud);
#include "pcl_point_types.def"

}  // namespace detail

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct model<type>;
#include "pcl_point_types.def"

}  // namespace voxel_score
