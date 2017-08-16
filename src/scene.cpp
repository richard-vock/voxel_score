#include <scene>
#include <impl/scene.hpp>

namespace voxel_score {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct scene<type>::impl; \
    template scene<type>::scene(); \
    template scene<type>::~scene(); \
    template gpu::future<void> scene<type>::init(typename pcl::PointCloud<type>::ConstPtr cloud, gpu_state::sptr_t state); \
    template typename scene<type>::gpu_data_t& scene<type>::device_data(); \
    template const typename scene<type>::gpu_data_t& scene<type>::device_data() const;
#include "pcl_point_types.def"

}  // namespace voxel_score

