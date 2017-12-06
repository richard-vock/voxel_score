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

    void
    init(typename cloud_t::ConstPtr cloud, gpu_state::sptr_t state) {
        pcl::search::KdTree<PointT> kdtree;
        kdtree.setInputCloud(cloud);

        // estimate resolution
        std::mt19937 gen;
        std::uniform_int_distribution<uint32_t> dis(0, cloud->size()-1);

        std::vector<int> is(2);
        std::vector<float> ds(2);
        resolution_ = 0.f;
        for (uint32_t i = 0; i < 100; ++i) {
            kdtree.nearestKSearch(cloud->points[dis(gen)], 2, is, ds);

            resolution_ += (sqrtf(ds[1])-resolution_) / (i+1);
        }

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
        vec3f_t margin = vec3f_t::Constant(4.f * resolution_);
        lower -= margin;
        upper += margin;
        range = upper-lower;

        //float voxel_step = range.maxCoeff() / max_dim_size;
        //redundancy_ = static_cast<double>(std::pow(voxel_step / resolution_, 3.f));
        //extents_ = (range / voxel_step)
                       //.template cast<int>()
                       //.cwiseMax(vec3i_t::Constant(1));

        extents_ = (range / resolution_)
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

        uint32_t voxel_count = extents_[0] * extents_[1] * extents_[2];
        //distance_data_.resize(voxel_count);
        cpu_data_.resize(voxel_count);

        max_dist_ = 0.f;
        for (int i = 0; i < extents_[0]; ++i) {
            int x = i * extents_[1] * extents_[2];
            for (int j = 0; j < extents_[1]; ++j) {
                int y = j * extents_[2];
                for (int k = 0; k < extents_[2]; ++k) {
                    PointT query;
                    vec3f_t coords = vec3i_t(i, j, k).template cast<float>();

                    query.getVector3fMap() =
                        (inv * coords.homogeneous()).head(3);
                    std::vector<int> nns(1);
                    std::vector<float> dists(1);
                    kdtree.nearestKSearch(query, 1, nns, dists);
                    //distance_data_[x + y + k] =
                        //max_integer_dist * (sqrtf(dists[0]) / max_dist);
                    vec3f_t neigh = cloud->points[nns[0]].getVector3fMap();
                    cpu_data_[x + y + k][0] = neigh[0];
                    cpu_data_[x + y + k][1] = neigh[1];
                    cpu_data_[x + y + k][2] = neigh[2];
                    cpu_data_[x + y + k][3] = 1.f;
                    max_dist_ = std::max(max_dist_, dists[0]);
                }
            }
        }
        max_dist_ = sqrtf(max_dist_);

        centroid_ = vec3f_t::Zero();
        for (uint32_t i = 0; i < cloud->size(); ++i) {
            centroid_ += (cloud->points[i].getVector3fMap() - centroid_) / (i+1);
        }

        gpu_data_ =
            gpu::vector<gpu::float4_>(cpu_data_.size(), state->context);
        std::cout << "VS model alloc: " << (static_cast<double>(4 * cpu_data_.size() * sizeof(float)) / (1024*1024)) << "MiB\n";
        gpu::copy(cpu_data_.begin(), cpu_data_.end(), gpu_data_.begin(), state->queue);
    }

    float resolution_;
    vec3i_t extents_;
    mat4f_t proj_;
    cpu_data_t cpu_data_;
    gpu_data_t gpu_data_;
    float max_dist_;
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
inline float
model<PointT>::resolution() const {
    return impl_->resolution_;
}

template <typename PointT>
inline void
model<PointT>::init(typename cloud_t::ConstPtr cloud, gpu_state::sptr_t state) {
    return impl_->init(cloud, state);
}

template <typename PointT>
inline typename model<PointT>::gpu_data_t&
model<PointT>::device_data() {
    return impl_->gpu_data_;
}

template <typename PointT>
inline const typename model<PointT>::gpu_data_t&
model<PointT>::device_data() const {
    return impl_->gpu_data_;
}

template <typename PointT>
inline const typename model<PointT>::cpu_data_t&
model<PointT>::host_data() const {
    return impl_->cpu_data_;
}

template <typename PointT>
inline const vec3f_t&
model<PointT>::centroid() const {
    return impl_->centroid_;
}

template <typename PointT>
inline float
model<PointT>::max_distance() const {
    return impl_->max_dist_;
}

}  // namespace voxel_score
