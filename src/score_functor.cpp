#include <score_functor>
#include <impl/score_functor.hpp>

namespace voxel_score {

#define INSTANTIATE_PCL_POINT_TYPE(type) \
    template struct score_functor<type, type>;
#include "pcl_point_types.def"

} // voxel_score
