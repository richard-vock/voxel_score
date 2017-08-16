#include <model>
#include <impl/model.hpp>

namespace voxel_score {

namespace detail {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template mat3f_t pca<type>(typename pcl::PointCloud<type>::ConstPtr cloud);
#include "pcl_point_types.def"

}  // namespace detail

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct model<type>::impl; \
    template model<type>::model(); \
    template model<type>::~model(); \
    template const mat4f_t& model<type>::projection() const; \
    template const vec3i_t& model<type>::extents() const; \
    template gpu::future<void> model<type>::init(typename pcl::PointCloud<type>::ConstPtr cloud, gpu_state::sptr_t state, int max_dim_size); \
    template typename model<type>::gpu_data_t& model<type>::device_data(); \
    template const typename model<type>::gpu_data_t& model<type>::device_data() const; \
    template const std::vector<uint8_t>& model<type>::host_data() const;
#include "pcl_point_types.def"

}  // namespace voxel_score
