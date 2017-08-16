#include <score_functor>
#include <impl/score_functor.hpp>

namespace voxel_score {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct score_functor<type, type>::impl; \
    template score_functor<type, type>::score_functor(gpu_state::sptr_t gstate); \
    template score_functor<type, type>::~score_functor(); \
    template void score_functor<type, type>::set_scene(typename pcl::PointCloud<type>::ConstPtr scene_cloud); \
    template void score_functor<type, type>::set_model(typename pcl::PointCloud<type>::ConstPtr model_cloud, int max_dim_size); \
    template float score_functor<type, type>::operator()(const mat4f_t& transform); \
    template typename score_functor<type, type>::subset_t score_functor<type, type>::correspondences(const mat4f_t& transform, float max_normalized_dist); \
    template const model<type>& score_functor<type, type>::get_model() const;
#include "pcl_point_types.def"

} // voxel_score
