#include <boost/compute/interop/eigen.hpp>

namespace voxel_score {

namespace detail {

constexpr char score_kernel_source[] =
    "__kernel void score_kernel(__global const float4* pnts, int n, __global const uchar* dist_voxel, int sx, int sy, int sz, __global const float* trans, int max_dist, __global long* output) {"
    "    const uint index = get_global_id(0);"
    "    if (index > n) {"
    "        return;"
    "    }"
    "    float4 pnt = pnts[index];"
    "    float4 tp = (float4)("
    "        trans[0] * pnt.x + trans[4] * pnt.y + trans[ 8] * pnt.z + trans[12] * pnt.w,"
    "        trans[1] * pnt.x + trans[5] * pnt.y + trans[ 9] * pnt.z + trans[13] * pnt.w,"
    "        trans[2] * pnt.x + trans[6] * pnt.y + trans[10] * pnt.z + trans[14] * pnt.w,"
    "        trans[3] * pnt.x + trans[7] * pnt.y + trans[11] * pnt.z + trans[15] * pnt.w"
    "    );"
    "    int x = (int)tp.x;"
    "    int y = (int)tp.y;"
    "    int z = (int)tp.z;"
    "    long dist = (long)max_dist;"
    "    if (x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz) {"
    "        dist = (long)(dist_voxel[x*sy*sz + y*sz + z]);"
    "    }"
    "    output[index] = dist;"
    "}";

constexpr char corrs_kernel_source[] =
    "__kernel void corrs_kernel(__global const long* scores, int n, int max_dist, __global int* output) {"
    "    const int index = get_global_id(0);"
    "    if (index > n) {"
    "        return;"
    "    }"
    "    output[index] = (int)(scores[index]) <= max_dist ? index : -1;"
    "}";

constexpr char icp_kernel_source[] =
    "__kernel void icp_average_kernel(__global const float4* pnts, int n,"
    "                                 __global const uchar* dist_voxel,"
    "                                 float cx, float cy, float cz,"
    "                                 int sx, int sy, int sz,"
    "                                 __global const float* proj,"
    "                                 __global const float* guess,"
    "                                 int max_dist,"
    "                                 int max_score_dist,"
    "                                 __global float* out_x,"
    "                                 __global float* out_y,"
    "                                 __global float* out_z,"
    "                                 __global int* out_indices) {"
    "    const uint index = get_global_id(0);"
    "    if (index > n) {"
    "        return;"
    "    }"
    "    float4 pnt = pnts[index];"
    "    float4 aligned = (float4)("
    "        guess[0] * pnt.x + guess[4] * pnt.y + guess[ 8] * pnt.z + guess[12] * pnt.w,"
    "        guess[1] * pnt.x + guess[5] * pnt.y + guess[ 9] * pnt.z + guess[13] * pnt.w,"
    "        guess[2] * pnt.x + guess[6] * pnt.y + guess[10] * pnt.z + guess[14] * pnt.w,"
    "        guess[3] * pnt.x + guess[7] * pnt.y + guess[11] * pnt.z + guess[15] * pnt.w"
    "    );"
    "    float4 loc = (float4)(aligned.x - cx, aligned.y - cy, aligned.z - cz, aligned.w);"
    "    float4 tp = (float4)("
    "        proj[0] * aligned.x + proj[4] * aligned.y + proj[ 8] * aligned.z + proj[12] * aligned.w,"
    "        proj[1] * aligned.x + proj[5] * aligned.y + proj[ 9] * aligned.z + proj[13] * aligned.w,"
    "        proj[2] * aligned.x + proj[6] * aligned.y + proj[10] * aligned.z + proj[14] * aligned.w,"
    "        proj[3] * aligned.x + proj[7] * aligned.y + proj[11] * aligned.z + proj[15] * aligned.w"
    "    );"
    "    int x = (int)tp.x;"
    "    int y = (int)tp.y;"
    "    int z = (int)tp.z;"
    "    if (x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz && ((int)(dist_voxel[x*sy*sz + y*sz + z]) < max_score_dist)) {"
    "       out_indices[index] = index;"
    "       out_x[index] = loc.x;"
    "       out_y[index] = loc.y;"
    "       out_z[index] = loc.z;"
    "    } else {"
    "       out_indices[index] = -1;"
    "       out_x[index] = 0.f;"
    "       out_y[index] = 0.f;"
    "       out_z[index] = 0.f;"
    "    }"
    "}";

constexpr int threads_per_block = 512;

} // detail

template <typename PointSceneT, typename PointModelT>
struct score_functor<PointSceneT, PointModelT>::impl {
    impl(gpu_state::sptr_t gstate) : gstate_(gstate), projection_(16, gstate->context), transform_(16, gstate->context) {
    }

    void init() {
        if (!has_scene_ || !has_model_) {
            throw std::runtime_error("Score functor needs both, a scene and a model set.");
        }

        ready_scene_.wait();
        ready_model_dist_.wait();
        ready_model_corr_.wait();

        // prepare kernel (all args except transform)
        uint32_t n = this->scene_->device_data().size();
        this->score_output_ = gpu::vector<cl_long>(n, this->gstate_->context);
        this->corrs_output_ = gpu::vector<int>(n, this->gstate_->context);
        this->icp_x_output_ = gpu::vector<float>(n, this->gstate_->context);
        this->icp_y_output_ = gpu::vector<float>(n, this->gstate_->context);
        this->icp_z_output_ = gpu::vector<float>(n, this->gstate_->context);
        score_program_ = gpu::program::create_with_source(detail::score_kernel_source, gstate_->context);
        score_program_.build();
        corrs_program_ = gpu::program::create_with_source(detail::corrs_kernel_source, gstate_->context);
        corrs_program_.build();
        icp_program_ = gpu::program::create_with_source(detail::icp_kernel_source, gstate_->context);
        icp_program_.build();
        score_kernel_ = gpu::kernel(score_program_, "score_kernel");
        score_kernel_.set_arg(0, this->scene_->device_data().get_buffer());
        score_kernel_.set_arg(1, n);
        score_kernel_.set_arg(2, this->model_->device_distance_data().get_buffer());
        score_kernel_.set_arg(3, static_cast<int>(this->model_->extents()[0]));
        score_kernel_.set_arg(4, static_cast<int>(this->model_->extents()[1]));
        score_kernel_.set_arg(5, static_cast<int>(this->model_->extents()[2]));
        score_kernel_.set_arg(7, static_cast<int>(max_integer_dist));
        score_kernel_.set_arg(8, this->score_output_.get_buffer());
        corrs_kernel_ = gpu::kernel(corrs_program_, "corrs_kernel");
        corrs_kernel_.set_arg(0, this->score_output_.get_buffer());
        corrs_kernel_.set_arg(1, n);
        corrs_kernel_.set_arg(3, this->corrs_output_.get_buffer());
        icp_average_kernel_ = gpu::kernel(icp_program_, "icp_average_kernel");
        icp_average_kernel_.set_arg(0, this->scene_->device_data().get_buffer());
        icp_average_kernel_.set_arg(1, n);
        icp_average_kernel_.set_arg(2, this->model_->device_distance_data().get_buffer());
        icp_average_kernel_.set_arg(3, static_cast<float>(this->model_->centroid()[0]));
        icp_average_kernel_.set_arg(4, static_cast<float>(this->model_->centroid()[1]));
        icp_average_kernel_.set_arg(5, static_cast<float>(this->model_->centroid()[2]));
        icp_average_kernel_.set_arg(6, static_cast<int>(this->model_->extents()[0]));
        icp_average_kernel_.set_arg(7, static_cast<int>(this->model_->extents()[1]));
        icp_average_kernel_.set_arg(8, static_cast<int>(this->model_->extents()[2]));
        mat4f_t proj = this->model_->projection();
        gpu::eigen_copy_matrix_to_buffer(proj, this->projection_.begin(), this->gstate_->queue);
        score_kernel_.set_arg(9, this->projection_.get_buffer());
        // 10 guess (dynamic input)
        icp_average_kernel_.set_arg(11, static_cast<int>(max_integer_dist));
        // 12 max score distance threshold (dynamic input)
        icp_average_kernel_.set_arg(13, this->icp_x_output_.get_buffer());
        icp_average_kernel_.set_arg(14, this->icp_y_output_.get_buffer());
        icp_average_kernel_.set_arg(15, this->icp_z_output_.get_buffer());
        icp_average_kernel_.set_arg(16, this->corrs_output_.get_buffer());
        global_threads_ = n;
        int res = n % detail::threads_per_block;
        if (res) {
            global_threads_ += detail::threads_per_block - res;
        }

//  0 __global const float4* pnts,
//  1 int n,
//  2 __global const uchar* dist_voxel,
//  3 float cx,
//  4 float cy,
//  5 float cz,
//  6 int sx,
//  7 int sy,
//  8 int sz,
//  9 __global const float* proj,
// 10 __global const float* guess,
// 11 int max_dist,
// 12 int max_score_dist,
// 13 __global float* out_x,
// 14 __global float* out_y,
// 15 __global float* out_z,
// 16 __global int* out_indices

        ready_ = true;
    }

    float score(const mat4f_t& t) {
        if (!ready_) {
            this->init();
        }

        // upload transform
        mat4f_t pt = this->model_->projection() * t;
        gpu::eigen_copy_matrix_to_buffer(pt, this->transform_.begin(), this->gstate_->queue);
        score_kernel_.set_arg(6, this->transform_.get_buffer());
        this->gstate_->queue.enqueue_1d_range_kernel(score_kernel_, 0, global_threads_, detail::threads_per_block);
        cl_long sum = 0;
        gpu::reduce(score_output_.begin(), score_output_.end(), &sum, gpu::plus<cl_long>(), this->gstate_->queue);
        return 1.f - static_cast<float>(static_cast<double>(sum) / (scene_->device_data().size() * max_integer_dist));
    }

    mat4f_t icp(const mat4f_t& guess, float max_ndist) {
        if (!ready_) {
            this->init();
        }

        // upload transform and threshold
        int max_int_distance = static_cast<int>(max_ndist * max_integer_dist);
        gpu::eigen_copy_matrix_to_buffer(guess, this->transform_.begin(), this->gstate_->queue);
        icp_average_kernel_.set_arg(10, this->transform_.get_buffer());
        icp_average_kernel_.set_arg(12, max_int_distance);

        // get correspondence count
        using gpu::lambda::_1;
        int n = 0;
        boost::compute::function<int (int)> indicator =
            boost::compute::make_function_from_source<int (int)>(
                "indicator",
                "int indicator(int x) { return x >= 0 ? 1 : 0; }"
            );
        gpu::transform_reduce(corrs_output_.begin(), corrs_output_.end(), &n, indicator, gpu::plus<int>(), this->gstate_->queue);

        // compute correspondence centroid
        this->gstate_->queue.enqueue_1d_range_kernel(icp_average_kernel_, 0, global_threads_, detail::threads_per_block);
        float x_sum = 0.f, y_sum = 0.f, z_sum = 0.f;
        gpu::reduce(icp_x_output_.begin(), icp_x_output_.end(), &x_sum, gpu::plus<float>(), this->gstate_->queue);
        gpu::reduce(icp_y_output_.begin(), icp_y_output_.end(), &y_sum, gpu::plus<float>(), this->gstate_->queue);
        gpu::reduce(icp_z_output_.begin(), icp_z_output_.end(), &z_sum, gpu::plus<float>(), this->gstate_->queue);
        //vec3f_t corr_centroid = vec3f_t(x_sum / n, y_sum / n, z_sum / n) + model_->centroid();
        return mat4f_t::Identity();
    }

    subset_t correspondences(const mat4f_t& t, float max_ndist) {
        if (!ready_) {
            this->init();
        }

        mat4f_t pt = this->model_->projection() * t.inverse();
        gpu::eigen_copy_matrix_to_buffer(pt, this->transform_.begin(), this->gstate_->queue);
        score_kernel_.set_arg(6, this->transform_.get_buffer());
        this->gstate_->queue.enqueue_1d_range_kernel(score_kernel_, 0, global_threads_, detail::threads_per_block);

        int max_int_distance = static_cast<int>(max_ndist * max_integer_dist);
        corrs_kernel_.set_arg(2, max_int_distance);
        this->gstate_->queue.enqueue_1d_range_kernel(corrs_kernel_, 0, global_threads_, detail::threads_per_block);
        subset_t subset(this->scene_->device_data().size());
        gpu::vector<int> dev_subset(this->scene_->device_data().size(), this->gstate_->context);
        using gpu::lambda::_1;
        auto dev_end = gpu::copy_if(corrs_output_.begin(), corrs_output_.end(), dev_subset.begin(), _1 >= 0, this->gstate_->queue);
        auto host_end = gpu::copy(dev_subset.begin(), dev_end, subset.begin(), this->gstate_->queue);
        subset.resize(std::distance(subset.begin(), host_end));
        return subset;
    }

    void
    set_scene(typename pcl::PointCloud<PointSceneT>::ConstPtr scene_cloud) {
        this->scene_ = std::make_unique<::voxel_score::scene<PointSceneT>>();
        this->ready_scene_ = scene_->init(scene_cloud, gstate_);
        this->has_scene_ = true;
    }

    void
    set_model(typename pcl::PointCloud<PointModelT>::ConstPtr model_cloud, int max_dim_size) {
        this->model_ = std::make_unique<::voxel_score::model<PointModelT>>();
        std::tie(this->ready_model_dist_, this->ready_model_corr_) =
            model_->init(model_cloud, this->gstate_, max_dim_size);
        this->has_model_ = true;
    }

    const model<PointModelT>&
    get_model() const {
        return *model_;
    }

    bool has_scene_ = false;
    bool has_model_ = false;
    bool ready_ = false;

    gpu_state::sptr_t gstate_;

    gpu::vector<float> projection_;
    gpu::vector<float> transform_;
    gpu::vector<cl_long> score_output_;
    gpu::vector<int> corrs_output_;
    gpu::vector<float> icp_x_output_;
    gpu::vector<float> icp_y_output_;
    gpu::vector<float> icp_z_output_;

    gpu::program score_program_;
    gpu::program corrs_program_;
    gpu::program icp_program_;
    gpu::kernel  score_kernel_;
    gpu::kernel  corrs_kernel_;
    gpu::kernel  icp_average_kernel_;
    int global_threads_;

    typename scene<PointSceneT>::uptr_t scene_;
    typename model<PointModelT>::uptr_t model_;

    gpu::future<void>   ready_scene_;
    gpu::future<void>   ready_model_dist_;
    gpu::future<void>   ready_model_corr_;
};

template <typename PointSceneT, typename PointModelT>
inline
score_functor<PointSceneT, PointModelT>::score_functor(gpu_state::sptr_t gstate) : impl_(new impl(gstate)) {
}

template <typename PointSceneT, typename PointModelT>
inline
score_functor<PointSceneT, PointModelT>::~score_functor() {
}

template <typename PointSceneT, typename PointModelT>
inline void
score_functor<PointSceneT, PointModelT>::set_scene(typename scene_cloud_t::ConstPtr scene_cloud) {
    impl_->set_scene(scene_cloud);
}

template <typename PointSceneT, typename PointModelT>
inline void
score_functor<PointSceneT, PointModelT>::set_model(typename model_cloud_t::ConstPtr model_cloud, int max_dim_size) {
    impl_->set_model(model_cloud, max_dim_size);
}

template <typename PointSceneT, typename PointModelT>
inline float
score_functor<PointSceneT, PointModelT>::operator()(const mat4f_t& transform) {
    return impl_->score(transform);
}

template <typename PointSceneT, typename PointModelT>
inline typename score_functor<PointSceneT, PointModelT>::subset_t
score_functor<PointSceneT, PointModelT>::correspondences(const mat4f_t& transform, float max_normalized_dist) {
    return impl_->correspondences(transform, max_normalized_dist);
}

template <typename PointSceneT, typename PointModelT>
inline const model<PointModelT>&
score_functor<PointSceneT, PointModelT>::get_model() const {
    return impl_->get_model();
}

} // voxel_score
