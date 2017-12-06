#include <scene>
#include <impl/scene.hpp>

namespace voxel_score {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct scene<type>::impl; \
    template scene<type>::scene(); \
    template scene<type>::~scene(); \
    template void scene<type>::init(typename pcl::PointCloud<type>::ConstPtr cloud, gpu_state::sptr_t state, const subset_t&); \
    template typename scene<type>::gpu_data_t& scene<type>::device_data(); \
    template const typename scene<type>::gpu_data_t& scene<type>::device_data() const; \
    template gpu::vector<int>& scene<type>::device_mask(); \
    template const gpu::vector<int>& scene<type>::device_mask() const; \
    template void scene<type>::reset_device_mask(gpu_state::sptr_t, const subset_t&); \
    template void scene<type>::update_device_mask(gpu_state::sptr_t, const std::vector<int>&);
#include "pcl_point_types.def"

}  // namespace voxel_score

