#include <boost/compute/interop/eigen.hpp>
#include "timer.hpp"

namespace voxel_score {

namespace detail {

//constexpr char score_kernel_source[] =
//    "__kernel void score_kernel(__global const float4* pnts, int n, __global const uchar* dist_voxel, int sx, int sy, int sz, __global const float* trans, int max_dist, __global long* output) {"
//    "    const uint index = get_global_id(0);"
//    "    if (index > n) {"
//    "        return;"
//    "    }"
//    "    float4 pnt = pnts[index];"
//    "    float4 tp = (float4)("
//    "        trans[0] * pnt.x + trans[4] * pnt.y + trans[ 8] * pnt.z + trans[12] * pnt.w,"
//    "        trans[1] * pnt.x + trans[5] * pnt.y + trans[ 9] * pnt.z + trans[13] * pnt.w,"
//    "        trans[2] * pnt.x + trans[6] * pnt.y + trans[10] * pnt.z + trans[14] * pnt.w,"
//    "        trans[3] * pnt.x + trans[7] * pnt.y + trans[11] * pnt.z + trans[15] * pnt.w"
//    "    );"
//    "    int x = (int)tp.x;"
//    "    int y = (int)tp.y;"
//    "    int z = (int)tp.z;"
//    "    long dist = (long)max_dist;"
//    "    if (x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz) {"
//    "        dist = (long)(dist_voxel[x*sy*sz + y*sz + z]);"
//    "    }"
//    "    output[index] = dist;"
//    "}"
//    "__kernel void corr_count_kernel(__global const float4* pnts, int n, __global const uchar* dist_voxel, int sx, int sy, int sz, __global const float* trans, int max_dist, int max_score_dist, __global int* output) {"
//    "    const uint index = get_global_id(0);"
//    "    if (index > n) {"
//    "        return;"
//    "    }"
//    "    float4 pnt = pnts[index];"
//    "    float4 tp = (float4)("
//    "        trans[0] * pnt.x + trans[4] * pnt.y + trans[ 8] * pnt.z + trans[12] * pnt.w,"
//    "        trans[1] * pnt.x + trans[5] * pnt.y + trans[ 9] * pnt.z + trans[13] * pnt.w,"
//    "        trans[2] * pnt.x + trans[6] * pnt.y + trans[10] * pnt.z + trans[14] * pnt.w,"
//    "        trans[3] * pnt.x + trans[7] * pnt.y + trans[11] * pnt.z + trans[15] * pnt.w"
//    "    );"
//    "    int x = (int)tp.x;"
//    "    int y = (int)tp.y;"
//    "    int z = (int)tp.z;"
//    "    long dist = (long)max_dist;"
//    "    if (x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz) {"
//    "        dist = (long)(dist_voxel[x*sy*sz + y*sz + z]);"
//    "    }"
//    "    output[index] = dist < max_score_dist ? 1 : 0;"
//    "}";
//
//constexpr char corrs_kernel_source[] =
//    "__kernel void corrs_kernel(__global const long* scores, int n, int max_dist, __global int* output) {"
//    "    const int index = get_global_id(0);"
//    "    if (index > n) {"
//    "        return;"
//    "    }"
//    "    output[index] = (int)(scores[index]) <= max_dist ? index : -1;"
//    "}";

constexpr char program_source[] =
    "__kernel void icp_projection(__global const float4* pnts, int n,\n"
    "                             __global const float4* voxel,\n"
    "                             int sx, int sy, int sz,\n"
    "                             __global const float* proj,\n"
    "                             __global const float* guess,\n"//"                             float max_voxel_dist,\n"
    "                             float max_score_dist,\n"
    "                             __global float4* out_positions,\n"
    "                             __global int* model_indices,\n"
    "                             __global int* scene_indices,\n"
    "                             __global int* mask,\n"
    "                             int consider_mask) {\n"
    "    const uint index = get_global_id(0);\n"
    "    if (index >= n) {\n"
    "        return;\n"
    "    }\n"
    "    model_indices[index] = -1;\n"
    "    scene_indices[index] = -1;\n"
    "    out_positions[index] = (float4)(0.f, 0.f, 0.f, 1.f);\n"
    "    if (consider_mask && mask[index] == 0) {\n"
    "       return;\n"
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
    "    if (x >= 0 && x < sx && y >= 0 && y < sy && z >= 0 && z < sz) {\n"
    "       int idx = x*sy*sz + y*sz + z;\n"
    "       float dist = length(voxel[idx].xyz - aligned.xyz);\n" //"       float dist = length(voxel[idx].xyz - aligned.xyz) / max_voxel_dist;\n"
    "       if (dist < max_score_dist) {\n"
    "           model_indices[index] = idx;\n"
    "           scene_indices[index] = index;\n"
    "           out_positions[index] = (float4)(aligned.x, aligned.y, aligned.z, 1.f);\n"
    "       }\n"
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
    "    if (index >= n) {\n"
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
    "    output[index].s9 = 0.f;\n"
    "    output[index].sa = 0.f;\n"
    "    output[index].sb = 0.f;\n"
    "    output[index].sc = 0.f;\n"
    "    output[index].sd = 0.f;\n"
    "    output[index].se = 0.f;\n"
    "    output[index].sf = 0.f;\n"
    "}\n"
    ;

constexpr int threads_per_block = 512;

} // detail

template <typename PointSceneT, typename PointModelT>
struct score_functor<PointSceneT, PointModelT>::impl {
    impl(gpu_state::sptr_t gstate) : gstate_(gstate), projection_(16, gstate->context), transform_(16, gstate->context), idx_pred_("idx_pred") {
    }

    void init() {
        if (!has_scene_ || !has_model_) {
            throw std::runtime_error("Score functor needs both, a scene and a model set.");
        }

        //ready_scene_.wait();
        //ready_model_.wait();

        program_ = gpu::program::create_with_source(detail::program_source, gstate_->context);
        program_.build();

        // buffers
        uint32_t n = this->scene_->device_data().size();
        model_corrs_ = gpu::vector<int>(n, this->gstate_->context);
        scene_corrs_ = gpu::vector<int>(n, this->gstate_->context);
        best_corrs_  = gpu::vector<int>(n, this->gstate_->context);
        //std::cout << "VS: unique_indices_: " << (static_cast<double>(n * sizeof(int)) / (1024*1024)) << "MiB\n";
        //unique_indices_ = gpu::vector<int>(n, this->gstate_->context);
        positions_ = gpu::vector<gpu::float4_>(n, this->gstate_->context);
        corr_matrices_ = gpu::vector<gpu::float16_>(n, this->gstate_->context);

        // functions
        idx_pred_ = gpu::detail::make_function_impl<bool (int x)>("idx_pred", "int x", " { return nt >= 0 ? true : false; }");

        // kernels
        icp_projection_kernel_ = gpu::kernel(program_, "icp_projection");
        icp_projection_kernel_.set_arg(0, this->scene_->device_data().get_buffer());
        icp_projection_kernel_.set_arg(1, n);
        icp_projection_kernel_.set_arg(2, model_->device_data().get_buffer());
        icp_projection_kernel_.set_arg(3, static_cast<int>(model_->extents()[0]));
        icp_projection_kernel_.set_arg(4, static_cast<int>(model_->extents()[1]));
        icp_projection_kernel_.set_arg(5, static_cast<int>(model_->extents()[2]));
        std::cout << model_->extents()[0] << " " << model_->extents()[1] << " " << model_->extents()[2] << "\n";
        mat4f_t proj = model_->projection();
        gpu::eigen_copy_matrix_to_buffer(proj, projection_.begin(), gstate_->queue);
        icp_projection_kernel_.set_arg(6, projection_.get_buffer());
        // 7 -> guess (dynamic input)
        //icp_projection_kernel_.set_arg(8, model_->max_distance());
        // 8 -> max score distance threshold (dynamic input)
        icp_projection_kernel_.set_arg(9, positions_.get_buffer());
        icp_projection_kernel_.set_arg(10, model_corrs_.get_buffer());
        icp_projection_kernel_.set_arg(11, scene_corrs_.get_buffer());
        icp_projection_kernel_.set_arg(12, this->scene_->device_mask().get_buffer());

        icp_correlation_kernel_ = gpu::kernel(program_, "icp_correlation");
        icp_correlation_kernel_.set_arg(0, positions_.get_buffer());
        icp_correlation_kernel_.set_arg(1, model_->device_data().get_buffer());
        icp_correlation_kernel_.set_arg(2, scene_corrs_.get_buffer());
        icp_correlation_kernel_.set_arg(3, model_corrs_.get_buffer());
        // 4 -> correlation count
        // 5 -> scene centroid
        // 6 -> model centroid
        icp_correlation_kernel_.set_arg(7, corr_matrices_.get_buffer());

        global_threads_ = n;
        int res = n % detail::threads_per_block;
        if (res) {
            global_threads_ += detail::threads_per_block - res;
        }

        ready_ = true;
    }

    void project_(const mat4f_t& guess, float score_distance_factor, bool use_mask = false) {
        if (!ready_) {
            this->init();
        }

        icp_projection_kernel_.set_arg(13, use_mask ? 1 : 0);

        // upload transform and threshold
        gpu::eigen_copy_matrix_to_buffer(guess, this->transform_.begin(), this->gstate_->queue);
        icp_projection_kernel_.set_arg(7, this->transform_.get_buffer());
        float dist_threshold = score_distance_factor * model_->resolution();
        icp_projection_kernel_.set_arg(8, dist_threshold);

        // compute correspondence centroid
        this->gstate_->queue.enqueue_1d_range_kernel(icp_projection_kernel_, 0, global_threads_, detail::threads_per_block);
    }

    //template <typename InputIterator>
    //auto get_unique_indices_(InputIterator begin, InputIterator end) {
        //auto copied_end = gpu::copy(begin, end, unique_indices_.begin(), this->gstate_->queue);
        //gpu::sort(unique_indices_.begin(), copied_end, this->gstate_->queue);
        //return gpu::unique(unique_indices_.begin(), copied_end,
                           //this->gstate_->queue);
    //}

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

    //uint32_t correspondence_count(const mat4f_t& t, float max_ndist, bool use_mask) {
        //project_(t, max_ndist, use_mask);

        //uint32_t sum = 0;
        //BOOST_COMPUTE_FUNCTION(int, indicator, (int x),
        //{
            //return x >= 0 ? 1 : 0;
        //});
        //gpu::transform_reduce(scene_corrs_.begin(), scene_corrs_.end(), &sum, indicator, gpu::plus<int>(), this->gstate_->queue);

        //gpu::sort(model_corrs_.begin(), model_corrs_.end(), this->gstate_->queue);
        //auto unique_end = gpu::unique(model_corrs_.begin(), model_corrs_.end(), this->gstate_->queue);
        //uint32_t correspondence_count = unique_end.get_index() - 1;
        //pdebug("correspondences: {} (in model: {})", sum, correspondence_count);
        //return correspondence_count; //sum;
    //}

    uint32_t correspondence_count(const mat4f_t& t, float score_distance_factor, bool use_mask) {
        project_(t, score_distance_factor, use_mask);

        uint32_t sum = 0;
        BOOST_COMPUTE_FUNCTION(int, indicator, (int x),
        {
            return x >= 0 ? 1 : 0;
        });
        gpu::transform_reduce(scene_corrs_.begin(), scene_corrs_.end(), &sum, indicator, gpu::plus<int>(), this->gstate_->queue);

        //pdebug("score: {}", sum);
        return sum;
    }

    std::vector<int>::iterator
    correspondences_(const gpu::vector<int>& in, std::vector<int>& out) {
        out.resize(in.size());
        gpu::copy(in.begin(), in.end(), out.begin(), this->gstate_->queue);
        return std::stable_partition(out.begin(), out.end(), [&] (int x) { return x >= 0; });
    }

    std::pair<mat4f_t, std::vector<int>>
    icp(const mat4f_t& guess, float score_distance_factor, uint32_t max_iterations, bool use_mask = false, bool update_mask = true) {
        mat4f_t transform = guess;

        project_(transform, score_distance_factor, use_mask);

        //gpu::buffer_iterator<int> valid_model;
        //gpu::buffer_iterator<int> valid_scene;

        //std::vector<int> valid_scene = correspondences_(scene_corrs_);

        //valid_model = gpu::stable_partition(
            //model_corrs_.begin(), model_corrs_.end(),
            //boost::compute::lambda::_1 >= 0, this->gstate_->queue);
        //valid_scene = gpu::stable_partition(
            //scene_corrs_.begin(), scene_corrs_.end(),
            //idx_pred_, this->gstate_->queue);

        //gpu::vector<int> tmp(scene_corrs_.begin(), scene_corrs_.end(), this->gstate_->queue);
        //gpu::copy(scene_corrs_.begin(), scene_corrs_.end(), best_corrs_.begin(), this->gstate_->queue);

        //gpu::buffer_iterator<int> last_true = gpu::transform_if(best_corrs_.begin(), best_corrs_.end(), scene_corrs_.begin(), gpu::identity<int>(), idx_pred_, this->gstate_->queue);
        //gpu::buffer_iterator<int> last_true =
            //gpu::copy_if(best_corrs_.begin(),
                         //best_corrs_.end(),
                         //scene_corrs_.begin(),
                         //idx_pred_,
                         //this->gstate_->queue);
        //gpu::buffer_iterator<int> last_false =
        //::boost::compute::copy_if(tmp.begin(),
                                  //tmp.end(),
                                  //last_true,
                                  //not1(predicate),
                                  //queue);
        //uint32_t best_score = valid_scene.get_index();
        //gpu::copy(scene_corrs_.begin(), scene_corrs_.end(), best_corrs_.begin(), this->gstate_->queue);
        //mat4f_t best_transform = transform;

        //for (uint32_t i = 0; i < max_iterations; ++i) {
        //    // model indices probably are redundant - for centroid computation we
        //    // need unique indices (copied into unique_indices_)
        //    uint32_t n_model = valid_model.get_index();
        //    uint32_t n_scene = valid_scene.get_index();
        //    if (!n_model) {
        //        // no correspondences
        //        return {mat4f_t::Identity(), std::vector<int>()};
        //    }

        //    // compute centroids
        //    gpu::float4_ centroid_scene =
        //        compute_centroid_(positions_, scene_corrs_.begin(), valid_scene);
        //    gpu::float4_ centroid_model =
        //        compute_centroid_(this->model_->device_data(),
        //                          model_corrs_.begin(), valid_model);

        //    // compute correlation matrices
        //    icp_correlation_kernel_.set_arg(4, n_scene);
        //    icp_correlation_kernel_.set_arg(5, centroid_scene);
        //    icp_correlation_kernel_.set_arg(6, centroid_model);
        //    int threads = n_scene;
        //    int res = n_scene % detail::threads_per_block;
        //    if (res) {
        //        threads += detail::threads_per_block - res;
        //    }
        //    this->gstate_->queue.enqueue_1d_range_kernel(icp_correlation_kernel_, 0, threads, detail::threads_per_block);
        //    gpu::float16_ corr_16;
        //    gpu::reduce(corr_matrices_.begin(), gpu::make_buffer_iterator<gpu::float16_>(corr_matrices_.get_buffer(), n_scene), &corr_16, this->gstate_->queue);
        //    mat3f_t corr_matrix;
        //    for (int i = 0; i < 9; ++i) {
        //        corr_matrix.data()[i] = corr_16[i];
        //    }

        //    Eigen::JacobiSVD<mat3f_t> svd(corr_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
        //    mat4f_t icp_mat = mat4f_t::Identity();
        //    icp_mat.topLeftCorner<3,3>() = svd.matrixV() * svd.matrixU().transpose();
        //    vec3f_t cm(centroid_model[0], centroid_model[1], centroid_model[2]);
        //    vec3f_t cs(centroid_scene[0], centroid_scene[1], centroid_scene[2]);
        //    icp_mat.block<3,1>(0,3) = cm - icp_mat.topLeftCorner<3,3>() * cs;

        //    //pdebug("ICP transform:\n {}", icp_mat);
        //    //icp_mat = mat4f_t::Identity();

        //    // update and reproject for correct correspondences
        //    transform = icp_mat * transform;
        //    project_(transform, score_distance_factor, use_mask);
        //    valid_model = gpu::stable_partition(
        //        model_corrs_.begin(), model_corrs_.end(),
        //        boost::compute::lambda::_1 >= 0, this->gstate_->queue);
        //    valid_scene = gpu::stable_partition(
        //        scene_corrs_.begin(), scene_corrs_.end(),
        //        boost::compute::lambda::_1 >= 0, this->gstate_->queue);

        //    uint32_t score = valid_scene.get_index();
        //    if (score > best_score) {
        //        // update current best
        //        best_score = score;
        //        gpu::copy(scene_corrs_.begin(), scene_corrs_.end(), best_corrs_.begin(), this->gstate_->queue);
        //        best_transform = transform;
        //    } else {
        //        // no convergence - exit early
        //        break;
        //    }
        //}

        // copy correspondences
        //valid_scene = gpu::stable_partition(
            //best_corrs_.begin(), best_corrs_.end(),
            //boost::compute::lambda::_1 >= 0, this->gstate_->queue);

        //uint32_t n_scene = valid_scene.get_index();
        //std::vector<int> corr(n_scene);
        //gpu::copy(best_corrs_.begin(), valid_scene, corr.begin(), this->gstate_->queue);

        std::vector<int> corr;
        auto new_end = correspondences_(scene_corrs_, corr);
        corr.resize(std::distance(corr.begin(), new_end));

        //if (use_mask && update_mask) {
            //scene_->update_device_mask(gstate_, corr);
        //}

        //return {best_transform, corr};
        return {transform, corr};
    }

    void
    reset(const subset_t& valid) {
        scene_->reset_device_mask(gstate_, valid);
    }

    void
    set_scene(typename pcl::PointCloud<PointSceneT>::ConstPtr scene_cloud, const subset_t& valid) {
        this->scene_ = std::make_unique<::voxel_score::scene<PointSceneT>>();
        scene_->init(scene_cloud, gstate_, valid);
        this->valid_scene_ = valid;
        this->has_scene_ = true;
    }

    void
    set_model(typename pcl::PointCloud<PointModelT>::ConstPtr model_cloud) {
        this->model_ = std::make_unique<::voxel_score::model<PointModelT>>();
        model_->init(model_cloud, this->gstate_);
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
    gpu::vector<int> model_corrs_;
    gpu::vector<int> scene_corrs_;
    gpu::vector<int> best_corrs_;
    //gpu::vector<int> unique_indices_;
    gpu::vector<gpu::float4_> positions_;
    gpu::vector<gpu::float16_> corr_matrices_;

    gpu::function<bool (int x)> idx_pred_;

    gpu::program program_;
    gpu::kernel  icp_projection_kernel_;
    gpu::kernel  icp_correlation_kernel_;
    int global_threads_;

    typename scene<PointSceneT>::uptr_t scene_;
    typename model<PointModelT>::uptr_t model_;

    subset_t valid_scene_;

    //gpu::future<void>   ready_scene_;
    //gpu::future<void>   ready_model_;
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
score_functor<PointSceneT, PointModelT>::set_scene(typename scene_cloud_t::ConstPtr scene_cloud, const subset_t& valid) {
    impl_->set_scene(scene_cloud, valid);
}

template <typename PointSceneT, typename PointModelT>
inline void
score_functor<PointSceneT, PointModelT>::set_model(typename model_cloud_t::ConstPtr model_cloud) {
    impl_->set_model(model_cloud);
}

template <typename PointSceneT, typename PointModelT>
inline void
score_functor<PointSceneT, PointModelT>::reset(const subset_t& valid) {
    impl_->reset(valid);
}

template <typename PointSceneT, typename PointModelT>
inline uint32_t
score_functor<PointSceneT, PointModelT>::correspondence_count(const mat4f_t& transform, float score_distance_factor, bool use_mask) {
    return impl_->correspondence_count(transform, score_distance_factor, use_mask);
}

template <typename PointSceneT, typename PointModelT>
inline std::pair<mat4f_t, std::vector<int>>
score_functor<PointSceneT, PointModelT>::icp(const mat4f_t& guess, float score_distance_factor, uint32_t max_iterations, bool use_mask, bool update_mask) {
    return impl_->icp(guess, score_distance_factor, max_iterations, use_mask, update_mask);
}

template <typename PointSceneT, typename PointModelT>
inline const model<PointModelT>&
score_functor<PointSceneT, PointModelT>::get_model() const {
    return impl_->get_model();
}

} // voxel_score
