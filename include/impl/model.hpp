namespace voxel_score {

namespace detail {

template <typename PointT>
inline mat3f_t
pca(typename pcl::PointCloud<PointT>::ConstPtr cloud) {
    // compute centroid
    rvec3f_t centroid = vec3f_t::Zero();
    rvec3f_t delta = vec3f_t::Zero();
    for (uint32_t i = 0; i < cloud->size(); ++i) {
        delta = cloud->points[i].getVector3fMap().transpose() - centroid;
        centroid += delta / static_cast<float>(i + 1);
    }

    // build scatter matrix
    matf_t scatter(cloud->size(), 3);
    for (uint32_t i = 0; i < cloud->size(); ++i) {
        scatter.row(i) =
            cloud->points[i].getVector3fMap().transpose() - centroid;
    }
    Eigen::JacobiSVD svd(scatter, Eigen::ComputeThinU | Eigen::ComputeThinV);
    return svd.matrixV();
}

}  // namespace detail

template <typename PointT>
struct model<PointT>::impl {
    impl() = default;

    ~impl() = default;

    std::pair<gpu::future<void>, gpu::future<void>>
    init(typename cloud_t::ConstPtr cloud, gpu_state::sptr_t state,
         int max_dim_size) {
        // compute local base
        mat3f_t base = detail::pca<PointT>(cloud).transpose();
        // compute bbox in local base
        bbox3_t bbox;
        for (const auto& pnt : *cloud) {
            bbox.extend(base * pnt.getVector3fMap());
        }
        vec3f_t lower = bbox.min();
        vec3f_t upper = bbox.max();
        vec3f_t range = upper - lower;
        // fix for thin bboxes
        float min_size = 0.1f * range.maxCoeff();
        for (int i = 0; i < 3; ++i) {
            if (range[i] < min_size) {
                lower[i] -= 0.5f * min_size;
                upper[i] += 0.5f * min_size;
                range[i] = upper[i] - lower[i];
            }
        }

        float voxel_step = range.maxCoeff() / max_dim_size;
        extents_ = (range / voxel_step)
                       .template cast<int>()
                       .cwiseMax(vec3i_t::Constant(1));

        // map global to normalized local
        proj_ = mat4f_t::Identity();
        proj_.topLeftCorner<3, 3>() = base;
        proj_.block<3, 1>(0, 3) = -lower;
        // scale to grid size
        proj_.row(0) *= extents_[0] / range[0];
        proj_.row(1) *= extents_[1] / range[1];
        proj_.row(2) *= extents_[2] / range[2];
        // subvoxel-shift
        proj_.block<3, 1>(0, 3) -= vec3f_t::Constant(0.5f);

        mat4f_t inv = proj_.inverse();

        pcl::search::KdTree<PointT> kdtree;
        kdtree.setInputCloud(cloud);

        // TODO: get rid of these
        float max_dist = 0.f;
        for (int i = 0; i < extents_[0]; ++i) {
            for (int j = 0; j < extents_[1]; ++j) {
                for (int k = 0; k < extents_[2]; ++k) {
                    PointT query;
                    vec3f_t coords = vec3i_t(i, j, k).template cast<float>();
                    query.getVector3fMap() =
                        (inv * coords.homogeneous()).head(3);
                    std::vector<int> nns(1);
                    std::vector<float> dists(1);
                    kdtree.nearestKSearch(query, 1, nns, dists);
                    max_dist = std::max(max_dist, dists[0]);
                }
            }
        }
        max_dist = sqrtf(max_dist);

        centroid_ = vec3f_t::Zero();
        for (uint32_t i = 0; i < cloud->size(); ++i) {
            centroid_ += (cloud->points[i].getVector3fMap() - centroid_) / (i+1);
        }

        distance_data_.resize(extents_[0] * extents_[1] * extents_[2]);
        corr_data_.resize(extents_[0] * extents_[1] * extents_[2] * 3);

        for (int i = 0; i < extents_[0]; ++i) {
            int x = i * extents_[1] * extents_[2];
            int xd = i * extents_[1] * extents_[2] * 3;
            for (int j = 0; j < extents_[1]; ++j) {
                int y = j * extents_[2];
                int yd = j * extents_[2] * 3;
                for (int k = 0; k < extents_[2]; ++k) {
                    PointT query;
                    vec3f_t coords = vec3i_t(i, j, k).template cast<float>();

                    query.getVector3fMap() =
                        (inv * coords.homogeneous()).head(3);
                    std::vector<int> nns(1);
                    std::vector<float> dists(1);
                    kdtree.nearestKSearch(query, 1, nns, dists);
                    distance_data_[x + y + k] =
                        max_integer_dist * (sqrtf(dists[0]) / max_dist);
                    corr_data_[xd + yd + k*3 + 0] = cloud->points[nns[0]].x - centroid_[0];
                    corr_data_[xd + yd + k*3 + 1] = cloud->points[nns[0]].y - centroid_[1];
                    corr_data_[xd + yd + k*3 + 2] = cloud->points[nns[0]].z - centroid_[2];
                }
            }
        }

        gpu_distance_data_ =
            gpu::vector<uint8_t>(distance_data_.size(), state->context);
        gpu_corr_data_ =
            gpu::vector<float>(corr_data_.size(), state->context);
        return {gpu::copy_async(distance_data_.begin(), distance_data_.end(),
                                gpu_distance_data_.begin(), state->queue),
                gpu::copy_async(corr_data_.begin(), corr_data_.end(),
                                gpu_corr_data_.begin(), state->queue)};
    }

    vec3i_t extents_;
    mat4f_t proj_;
    std::vector<uint8_t> distance_data_;
    gpu_dist_data_t gpu_distance_data_;
    std::vector<uint32_t> corr_data_;
    gpu_corr_data_t gpu_corr_data_;
    vec3f_t centroid_;
};

template <typename PointT>
inline model<PointT>::model() : impl_(new impl()) {}

template <typename PointT>
inline model<PointT>::~model() {}

template <typename PointT>
inline const mat4f_t&
model<PointT>::projection() const {
    return impl_->proj_;
}

template <typename PointT>
inline const vec3i_t&
model<PointT>::extents() const {
    return impl_->extents_;
}

template <typename PointT>
inline std::pair<gpu::future<void>, gpu::future<void>>
model<PointT>::init(typename cloud_t::ConstPtr cloud, gpu_state::sptr_t state,
                    int max_dim_size) {
    return impl_->init(cloud, state, max_dim_size);
}

template <typename PointT>
inline typename model<PointT>::gpu_dist_data_t&
model<PointT>::device_distance_data() {
    return impl_->gpu_distance_data_;
}

template <typename PointT>
inline const typename model<PointT>::gpu_dist_data_t&
model<PointT>::device_distance_data() const {
    return impl_->gpu_distance_data_;
}

template <typename PointT>
inline const std::vector<uint8_t>&
model<PointT>::host_distance_data() const {
    return impl_->distance_data_;
}

template <typename PointT>
inline typename model<PointT>::gpu_corr_data_t&
model<PointT>::device_correspondence_data() {
    return impl_->gpu_corr_data_;
}

template <typename PointT>
inline const typename model<PointT>::gpu_corr_data_t&
model<PointT>::device_correspondence_data() const {
    return impl_->gpu_corr_data_;
}

template <typename PointT>
inline const std::vector<uint32_t>&
model<PointT>::host_correspondence_data() const {
    return impl_->corr_data_;
}

template <typename PointT>
inline const vec3f_t&
model<PointT>::centroid() const {
    return impl_->centroid_;
}

}  // namespace voxel_score
