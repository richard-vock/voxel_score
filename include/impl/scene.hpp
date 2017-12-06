namespace voxel_score {

template <typename PointT>
struct scene<PointT>::impl {
    impl(){};

    ~impl() {}

    void
    init(typename cloud_t::ConstPtr cloud, gpu_state::sptr_t state, const subset_t& valid) {
        cloud_data_.resize(cloud->size() * 4);
        for (uint32_t i = 0; i < cloud->size(); ++i) {
            cloud_data_[i * 4 + 0] = cloud->points[i].x;
            cloud_data_[i * 4 + 1] = cloud->points[i].y;
            cloud_data_[i * 4 + 2] = cloud->points[i].z;
            cloud_data_[i * 4 + 3] = 1.f;
        }

        gpu_data_ = gpu_data_t(cloud->size(), state->context);
        std::cout << "VS: scene alloc: " << (static_cast<double>(4 * cloud->size() * sizeof(float)) / (1024*1024)) << "MiB\n";
        gpu::copy(
            reinterpret_cast<gpu::float4_*>(cloud_data_.data()),
            reinterpret_cast<gpu::float4_*>(cloud_data_.data()) + cloud->size(),
            gpu_data_.begin(), state->queue);

        gpu_mask_data_ = gpu::vector<int>(cloud->size(), state->context);

        reset_device_mask(state, valid);
    }

    void reset_device_mask(gpu_state::sptr_t state, const subset_t& valid) {
        cpu_mask_data_.resize(gpu_mask_data_.size(), valid.empty() ? 1 : 0);
        for (uint32_t idx : valid) {
            cpu_mask_data_[idx] = 1;
        }
        gpu::copy(cpu_mask_data_.begin(), cpu_mask_data_.end(), gpu_mask_data_.begin(), state->queue);
    }

    void update_device_mask(gpu_state::sptr_t state, const std::vector<int>& damage) {
        for (int idx : damage) {
            cpu_mask_data_[idx] = 0;
        }
        gpu::copy(cpu_mask_data_.begin(), cpu_mask_data_.end(), gpu_mask_data_.begin(), state->queue);
    }

    std::vector<float> cloud_data_;
    gpu_data_t gpu_data_;
    std::vector<int> cpu_mask_data_;
    gpu::vector<int> gpu_mask_data_;
};

template <typename PointT>
inline scene<PointT>::scene() : impl_(new impl()) {}

template <typename PointT>
inline scene<PointT>::~scene() {}

template <typename PointT>
void
inline scene<PointT>::init(typename cloud_t::ConstPtr cloud,
                                             gpu_state::sptr_t state, const subset_t& valid) {
    return impl_->init(cloud, state, valid);
}

template <typename PointT>
inline typename scene<PointT>::gpu_data_t&
scene<PointT>::device_data() {
    return impl_->gpu_data_;
}

template <typename PointT>
inline const typename scene<PointT>::gpu_data_t&
scene<PointT>::device_data() const {
    return impl_->gpu_data_;
}

template <typename PointT>
inline gpu::vector<int>&
scene<PointT>::device_mask() {
    return impl_->gpu_mask_data_;
}

template <typename PointT>
inline const gpu::vector<int>&
scene<PointT>::device_mask() const {
    return impl_->gpu_mask_data_;
}

template <typename PointT>
inline void
scene<PointT>::reset_device_mask(gpu_state::sptr_t state, const subset_t& valid) {
    impl_->reset_device_mask(state, valid);
}

template <typename PointT>
inline void
scene<PointT>::update_device_mask(gpu_state::sptr_t state, const std::vector<int>& damage) {
    impl_->update_device_mask(state, damage);
}

}  // namespace voxel_score
