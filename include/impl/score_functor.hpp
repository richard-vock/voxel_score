#include <boost/compute/interop/eigen.hpp>

namespace voxel_score {

namespace detail {

constexpr char score_kernel_source[] =
    "__kernel void score_kernel(__global const float4* pnts, int n, __global const uchar* voxel, int sx, int sy, int sz, __global const float* trans, int max_dist, __global long* output) {"
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
    "        dist = (long)(voxel[x*sy*sz + y*sz + z]);"
    "    }"
    "    output[index] = dist;"
    "}";

constexpr int threads_per_block = 512;

} // detail

template <typename PointSceneT, typename PointModelT>
struct score_functor<PointSceneT, PointModelT>::impl {
    impl(gpu_state::sptr_t gstate) : gstate_(gstate), transform_(16, gstate->context) {
    }

    void init() {
        if (!has_scene_ || !has_model_) {
            throw std::runtime_error("Score functor needs both, a scene and a model set.");
        }

        ready_scene_.wait();
        ready_model_.wait();

        // prepare kernel (all args except transform)
        uint32_t n = this->scene_->device_data().size();
        this->output_ = gpu::vector<cl_long>(n, this->gstate_->context);
        program_ = gpu::program::create_with_source(detail::score_kernel_source, gstate_->context);
        program_.build();
        kernel_ = gpu::kernel(program_, "score_kernel");
        kernel_.set_arg(0, this->scene_->device_data().get_buffer());
        kernel_.set_arg(1, n);
        kernel_.set_arg(2, this->model_->device_data().get_buffer());
        kernel_.set_arg(3, static_cast<int>(this->model_->extents()[0]));
        kernel_.set_arg(4, static_cast<int>(this->model_->extents()[1]));
        kernel_.set_arg(5, static_cast<int>(this->model_->extents()[2]));
        kernel_.set_arg(7, static_cast<int>(max_integer_dist));
        kernel_.set_arg(8, this->output_.get_buffer());
        global_threads_ = n;
        int res = n % detail::threads_per_block;
        if (res) {
            global_threads_ += detail::threads_per_block - res;
        }

        ready_ = true;
    }

    float score(const mat4f_t& t) {
        if (!ready_) {
            this->init();
        }

        // upload transform
        mat4f_t pt = this->model_->projection() * t;
        gpu::eigen_copy_matrix_to_buffer(pt, this->transform_.begin(), this->gstate_->queue);
        kernel_.set_arg(6, this->transform_.get_buffer());
        this->gstate_->queue.enqueue_1d_range_kernel(kernel_, 0, global_threads_, detail::threads_per_block);
        cl_long sum = 0;
        gpu::reduce(output_.begin(), output_.end(), &sum, gpu::plus<cl_long>(), this->gstate_->queue);
        return 1.f - static_cast<float>(static_cast<double>(sum) / (scene_->device_data().size() * max_integer_dist));
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
        this->ready_model_ = model_ ->init(model_cloud, this->gstate_, max_dim_size);
        this->has_model_ = true;
    }

    bool has_scene_ = false;
    bool has_model_ = false;
    bool ready_ = false;

    gpu_state::sptr_t gstate_;

    gpu::vector<float> transform_;
    gpu::vector<cl_long> output_;

    gpu::program program_;
    gpu::kernel  kernel_;
    int global_threads_;

    typename scene<PointSceneT>::uptr_t scene_;
    typename model<PointModelT>::uptr_t model_;

    gpu::future<void>   ready_scene_;
    gpu::future<void>   ready_model_;
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

} // voxel_score
