namespace voxel_score {

inline gpu_state::gpu_state()
    : device(gpu::system::default_device()),
      context(device),
      queue(context, device) {}

}  // namespace voxel_score
