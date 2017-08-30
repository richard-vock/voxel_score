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
    "}"
    "__kernel void corr_count_kernel(__global const float4* pnts, int n, __global const uchar* dist_voxel, int sx, int sy, int sz, __global const float* trans, int max_dist, int max_score_dist, __global int* output) {"
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
    "    output[index] = dist < max_score_dist ? 1 : 0;"
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
    "__kernel void icp_projection(__global const float4* pnts, int n,\n"
    "                             __global const long* dist_voxel,\n"
    "                             int sx, int sy, int sz,\n"
    "                             __global const float* proj,\n"
    "                             __global const float* guess,\n"
    "                             int max_score_dist,\n"
    "                             __global float4* out_positions,\n"
    "                             __global int* model_indices,\n"
    "                             __global int* scene_indices) {\n"
    "    const uint index = get_global_id(0);\n"
    "    if (index > n) {\n"
    "        return;\n"
    "    }\n"
    "    float4 pnt = pnts[index];\n"
    "    float4 aligned = (float4)(\n"
    "        guess[0] * pnt.x + guess[4] * pnt.y + guess[ 8] * pnt.z + guess[12] * pnt.w,\n"
    "        guess[1] * pnt.x + guess[5] * pnt.y + guess[ 9] * pnt.z + guess[13] * pnt.w,\n"
    "        guess[2] * pnt.x + guess[6] * pnt.y + guess[10] * pnt.z + guess[14] * pnt.w,\n"
    "        guess[3] * pnt.x + guess[7] * pnt.y + guess[11] * pnt.z + guess[15] * pnt.w\n"
    "    );\n"
    "    int x = (int)(proj[0] * aligned.x + proj[4] * aligned.y + proj[ 8] * aligned.z + proj[12] * aligned.w);\n"
    "    int y = (int)(proj[1] * aligned.x + proj[5] * aligned.y + proj[ 9] * aligned.z + proj[13] * aligned.w);\n"
    "    int z = (int)(proj[2] * aligned.x + proj[6] * aligned.y + proj[10] * aligned.z + proj[14] * aligned.w);\n"
    "    int idx = x*sy*sz + y*sz + z;\n"
    "    if (x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz && ((int)(dist_voxel[idx]) < max_score_dist)) {\n"
    "       model_indices[index] = idx;\n"
    "       scene_indices[index] = index;\n"
    "       out_positions[index] = (float4)(aligned.x, aligned.y, aligned.z, 1.f);\n"
    "    } else {\n"
    "       model_indices[index] = -1;\n"
    "       scene_indices[index] = -1;\n"
    "       out_positions[index] = (float4)(0.f, 0.f, 0.f, 1.f);\n"
    "    }\n"
    "}\n"
    "__kernel void icp_correlation(__global const float4* scene,\n"
    "                              __global const float4* model,\n"
    "                              __global const int* indices_scene,\n"
    "                              __global const int* indices_model,\n"
    "                              int n,\n"
    "                              float4 centroid_scene,\n"
    "                              float4 centroid_model,\n"
    "                              __global float16* output) {\n"
    "    const uint index = get_global_id(0);\n"
    "    if (index > n) {\n"
    "        return;\n"
    "    }\n"
    "    float4 spt = scene[indices_scene[index]] - centroid_scene;\n"
    "    float4 mpt = model[indices_model[index]] - centroid_model;\n"
    "    float norm = 1.f / (n-1);"
    "    output[index].s0 = spt.x * mpt.x * norm;\n"
    "    output[index].s1 = spt.y * mpt.x * norm;\n"
    "    output[index].s2 = spt.z * mpt.x * norm;\n"
    "    output[index].s3 = spt.x * mpt.y * norm;\n"
    "    output[index].s4 = spt.y * mpt.y * norm;\n"
    "    output[index].s5 = spt.z * mpt.y * norm;\n"
    "    output[index].s6 = spt.x * mpt.z * norm;\n"
    "    output[index].s7 = spt.y * mpt.z * norm;\n"
    "    output[index].s8 = spt.z * mpt.z * norm;\n"
    "}\n"
    ;

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

        score_program_ = gpu::program::create_with_source(detail::score_kernel_source, gstate_->context);
        score_program_.build();
        corrs_program_ = gpu::program::create_with_source(detail::corrs_kernel_source, gstate_->context);
        corrs_program_.build();
        icp_program_ = gpu::program::create_with_source(detail::icp_kernel_source, gstate_->context);
        icp_program_.build();

        // buffers
        uint32_t n = this->scene_->device_data().size();
        score_output_ = gpu::vector<cl_long>(n, this->gstate_->context);
        model_corrs_ = gpu::vector<int>(n, this->gstate_->context);
        scene_corrs_ = gpu::vector<int>(n, this->gstate_->context);
        unique_indices_ = gpu::vector<int>(n, this->gstate_->context);
        positions_ = gpu::vector<gpu::float4_>(n, this->gstate_->context);
        corr_matrices_ = gpu::vector<gpu::float16_>(n, this->gstate_->context);

        // kernels
        icp_projection_kernel_ = gpu::kernel(icp_program_, "icp_projection");
        icp_projection_kernel_.set_arg(0, this->scene_->device_data().get_buffer());
        icp_projection_kernel_.set_arg(1, n);
        icp_projection_kernel_.set_arg(2, model_->device_distance_data().get_buffer());
        icp_projection_kernel_.set_arg(3, static_cast<int>(model_->extents()[0]));
        icp_projection_kernel_.set_arg(4, static_cast<int>(model_->extents()[1]));
        icp_projection_kernel_.set_arg(5, static_cast<int>(model_->extents()[2]));
        mat4f_t proj = model_->projection();
        gpu::eigen_copy_matrix_to_buffer(proj, projection_.begin(), gstate_->queue);
        icp_projection_kernel_.set_arg(6, projection_.get_buffer());
        // 7 -> guess (dynamic input)
        // 8 -> max score distance threshold (dynamic input)
        icp_projection_kernel_.set_arg(9, positions_.get_buffer());
        icp_projection_kernel_.set_arg(10, model_corrs_.get_buffer());
        icp_projection_kernel_.set_arg(11, scene_corrs_.get_buffer());

        icp_correlation_kernel_ = gpu::kernel(icp_program_, "icp_correlation");
        icp_correlation_kernel_.set_arg(0, positions_.get_buffer());
        icp_correlation_kernel_.set_arg(1, model_->device_correspondence_data().get_buffer());
        icp_correlation_kernel_.set_arg(2, scene_corrs_.get_buffer());
        icp_correlation_kernel_.set_arg(3, model_corrs_.get_buffer());
        // 4 -> correlation count
        // 5 -> scene centroid
        // 6 -> model centroid
        icp_correlation_kernel_.set_arg(7, corr_matrices_.get_buffer());

        score_kernel_ = gpu::kernel(score_program_, "score_kernel");
        score_kernel_.set_arg(0, this->scene_->device_data().get_buffer());
        score_kernel_.set_arg(1, n);
        score_kernel_.set_arg(2, this->model_->device_distance_data().get_buffer());
        score_kernel_.set_arg(3, static_cast<int>(this->model_->extents()[0]));
        score_kernel_.set_arg(4, static_cast<int>(this->model_->extents()[1]));
        score_kernel_.set_arg(5, static_cast<int>(this->model_->extents()[2]));
        // 6 -> transform
        score_kernel_.set_arg(7, static_cast<int>(max_integer_dist));
        score_kernel_.set_arg(8, this->score_output_.get_buffer());

        corr_count_kernel_ = gpu::kernel(score_program_, "corr_count_kernel");
        corr_count_kernel_.set_arg(0, this->scene_->device_data().get_buffer());
        corr_count_kernel_.set_arg(1, n);
        corr_count_kernel_.set_arg(2, this->model_->device_distance_data().get_buffer());
        corr_count_kernel_.set_arg(3, static_cast<int>(this->model_->extents()[0]));
        corr_count_kernel_.set_arg(4, static_cast<int>(this->model_->extents()[1]));
        corr_count_kernel_.set_arg(5, static_cast<int>(this->model_->extents()[2]));
        // 6 -> transform
        corr_count_kernel_.set_arg(7, static_cast<int>(max_integer_dist));
        // 8 -> max_dist
        corr_count_kernel_.set_arg(9, this->model_corrs_.get_buffer());

        corrs_kernel_ = gpu::kernel(corrs_program_, "corrs_kernel");
        corrs_kernel_.set_arg(0, this->score_output_.get_buffer());
        corrs_kernel_.set_arg(1, n);
        corrs_kernel_.set_arg(3, this->model_corrs_.get_buffer());

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
        score_kernel_.set_arg(6, this->transform_.get_buffer());
        this->gstate_->queue.enqueue_1d_range_kernel(score_kernel_, 0, global_threads_, detail::threads_per_block);
        cl_long sum = 0;
        gpu::reduce(score_output_.begin(), score_output_.end(), &sum, gpu::plus<cl_long>(), this->gstate_->queue);
        return 1.f - static_cast<float>(static_cast<double>(sum) / (scene_->device_data().size() * max_integer_dist));
    }

    uint32_t correspondence_count(const mat4f_t& t, float max_ndist) {
        if (!ready_) {
            this->init();
        }

        // upload transform and max dist
        int max_int_distance = static_cast<int>(max_ndist * max_integer_dist);
        mat4f_t pt = this->model_->projection() * t;
        gpu::eigen_copy_matrix_to_buffer(pt, this->transform_.begin(), this->gstate_->queue);
        corr_count_kernel_.set_arg(6, this->transform_.get_buffer());
        corr_count_kernel_.set_arg(8, max_int_distance);
        this->gstate_->queue.enqueue_1d_range_kernel(corr_count_kernel_, 0, global_threads_, detail::threads_per_block);
        int sum = 0;
        gpu::reduce(model_corrs_.begin(), model_corrs_.end(), &sum, gpu::plus<int>(), this->gstate_->queue);
        return sum;
    }

    template <typename InputIterator>
    auto get_unique_indices_(InputIterator begin, InputIterator end) {
        auto copied_end = gpu::copy(begin, end, unique_indices_.begin(), this->gstate_->queue);
        gpu::sort(unique_indices_.begin(), copied_end, this->gstate_->queue);
        return gpu::unique(unique_indices_.begin(), copied_end,
                           this->gstate_->queue);
    }

    template <typename IndexIterator>
    gpu::float4_ compute_centroid_(gpu::vector<gpu::float4_>& data, IndexIterator begin, IndexIterator end) {
        gpu::float4_ result;
        gpu::reduce(
            gpu::make_permutation_iterator(data.begin(), begin),
            gpu::make_permutation_iterator(gpu::make_buffer_iterator<gpu::float4_>(data.get_buffer(), end.get_index()), end),
            &result,
            this->gstate_->queue
        );
        result[0] /= result[3];
        result[1] /= result[3];
        result[2] /= result[3];
        return result;
    }

    mat4f_t icp(const mat4f_t& guess, float max_ndist, std::vector<vec3f_t>& debug) {
        if (!ready_) {
            this->init();
        }

        // upload transform and threshold
        int max_int_distance = static_cast<int>(max_ndist * max_integer_dist);
        gpu::eigen_copy_matrix_to_buffer(guess, this->transform_.begin(), this->gstate_->queue);
        icp_projection_kernel_.set_arg(7, this->transform_.get_buffer());
        icp_projection_kernel_.set_arg(8, max_int_distance);

        // compute correspondence centroid
        this->gstate_->queue.enqueue_1d_range_kernel(icp_projection_kernel_, 0, global_threads_, detail::threads_per_block);

        // generate index sets
        auto valid_model = gpu::stable_partition(
            model_corrs_.begin(), model_corrs_.end(),
            boost::compute::lambda::_1 >= 0, this->gstate_->queue);
        auto valid_scene = gpu::stable_partition(
            scene_corrs_.begin(), scene_corrs_.end(),
            boost::compute::lambda::_1 >= 0, this->gstate_->queue);

        // model indices probably are redundant - for centroid computation we
        // need unique indices (copied into unique_indices_)
        auto unique_model = get_unique_indices_(model_corrs_.begin(), valid_model);

        uint32_t n_model = unique_model.get_index();
        uint32_t n_scene = valid_scene.get_index();
        if (!n_model) {
            // no correspondences
            return mat4f_t::Identity();
        }
        // compute centroids
        gpu::float4_ centroid_scene =
            compute_centroid_(positions_, scene_corrs_.begin(), valid_scene);
        gpu::float4_ centroid_model =
            compute_centroid_(this->model_->device_correspondence_data(),
                              unique_indices_.begin(), unique_model);

        std::cout << "scene centroid: " << centroid_scene << "\n";
        std::cout << "model centroid: " << centroid_model << "\n";

        // compute correlation matrices
        icp_correlation_kernel_.set_arg(4, n_scene);
        icp_correlation_kernel_.set_arg(5, centroid_scene);
        icp_correlation_kernel_.set_arg(6, centroid_model);
        int threads = n_scene;
        int res = n_scene % detail::threads_per_block;
        if (res) {
            threads += detail::threads_per_block - res;
        }
        this->gstate_->queue.enqueue_1d_range_kernel(icp_correlation_kernel_, 0, threads, detail::threads_per_block);
        gpu::float16_ corr_16;
        gpu::reduce(corr_matrices_.begin(), gpu::make_buffer_iterator<gpu::float16_>(corr_matrices_.get_buffer(), n_scene), &corr_16, this->gstate_->queue);
        mat3f_t corr_matrix;
        for (int i = 0; i < 9; ++i) {
            corr_matrix.data()[i] = corr_16[i];
        }

        Eigen::JacobiSVD<mat3f_t> svd(corr_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        mat4f_t icp_mat = mat4f_t::Identity();
        icp_mat.topLeftCorner<3,3>() = svd.matrixV() * svd.matrixU().transpose();
        std::cout << icp_mat << "\n";
        vec3f_t cm(centroid_model[0], centroid_model[1], centroid_model[2]);
        vec3f_t cs(centroid_scene[0], centroid_scene[1], centroid_scene[2]);
        icp_mat.block<3,1>(0,3) = cm - icp_mat.topLeftCorner<3,3>() * cs;
        return icp_mat * guess;
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
        auto dev_end = gpu::copy_if(model_corrs_.begin(), model_corrs_.end(), dev_subset.begin(), _1 >= 0, this->gstate_->queue);
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
    gpu::vector<int> model_corrs_;
    gpu::vector<int> scene_corrs_;
    gpu::vector<int> unique_indices_;
    gpu::vector<gpu::float4_> positions_;
    gpu::vector<gpu::float16_> corr_matrices_;

    gpu::program score_program_;
    gpu::program corrs_program_;
    gpu::program icp_program_;
    gpu::kernel  score_kernel_;
    gpu::kernel  corr_count_kernel_;
    gpu::kernel  corrs_kernel_;
    gpu::kernel  icp_projection_kernel_;
    gpu::kernel  icp_correlation_kernel_;
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
inline uint32_t
score_functor<PointSceneT, PointModelT>::correspondence_count(const mat4f_t& transform, float max_normalized_dist) {
    return impl_->correspondence_count(transform, max_normalized_dist);
}

template <typename PointSceneT, typename PointModelT>
inline typename score_functor<PointSceneT, PointModelT>::subset_t
score_functor<PointSceneT, PointModelT>::correspondences(const mat4f_t& transform, float max_normalized_dist) {
    return impl_->correspondences(transform, max_normalized_dist);
}

template <typename PointSceneT, typename PointModelT>
inline mat4f_t
score_functor<PointSceneT, PointModelT>::icp(const mat4f_t& guess, float max_normalized_dist, std::vector<vec3f_t>& debug) {
    return impl_->icp(guess, max_normalized_dist, debug);
}

template <typename PointSceneT, typename PointModelT>
inline const model<PointModelT>&
score_functor<PointSceneT, PointModelT>::get_model() const {
    return impl_->get_model();
}

} // voxel_score
