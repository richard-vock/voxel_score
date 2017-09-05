namespace voxel_score {

template <typename PointT>
struct scene<PointT>::impl {
    impl(){};

    ~impl() {}

    void
    init(typename cloud_t::ConstPtr cloud, gpu_state::sptr_t state) {
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
    }

    std::vector<float> cloud_data_;
    gpu_data_t gpu_data_;
};

template <typename PointT>
inline scene<PointT>::scene() : impl_(new impl()) {}

template <typename PointT>
inline scene<PointT>::~scene() {}

template <typename PointT>
void
inline scene<PointT>::init(typename cloud_t::ConstPtr cloud,
                                             gpu_state::sptr_t state) {
    return impl_->init(cloud, state);
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

}  // namespace voxel_score
