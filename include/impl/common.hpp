namespace voxel_score {

inline gpu_state::gpu_state()
    : device(gpu::system::default_device()),
      context(device),
      queue(context, device) {
    pdebug("Compute device: {}", device.name());
}

}  // namespace voxel_score
