#include <test.cuh>

#include <iostream>
#include <cstdint>

#include <cuda_runtime.h>

#include <pcl/gpu/utils/safe_call.hpp>
#include <pcl/gpu/containers/device_array.h>

namespace voxel_score {

namespace detail {

__global__ void
test_kernel(int* data, size_t count) {
    const uint32_t index = blockIdx.x;
    if (index >= count) {
        return;
    }
    data[index] += 3;
}

}

void
test_compute(int* array, int count, int* sum_value, int* min_value) {
    pcl::gpu::DeviceArray<int> dev_ar;
    dev_ar.upload(array, count);

    detail::test_kernel<<<4, 1>>>(dev_ar, dev_ar.size());
    cudaSafeCall(cudaDeviceSynchronize());

    dev_ar.download(array);
    for(int i=0; i<4; ++i) {
        std::cout << array[i] << "\n";
    }

    *sum_value = 3;
    *min_value = 2;
}

} // voxel_score
