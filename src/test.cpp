#include <common.hpp>

namespace voxel_score {

std::pair<int, int>
test_func(int* array, int count) {
    gpu_state state;

    gpu::vector<int> dev_vector(count, state.context);
    gpu::copy(array, array+count, dev_vector.begin(), state.queue);

    gpu::function<int(int)> add_func = gpu::make_function_from_source<int(int)>(
        "add_func", "int add_func(int x) { return x+1; }");
    gpu::transform(dev_vector.begin(), dev_vector.end(), dev_vector.begin(), add_func, state.queue);

    std::vector<int> host_vector(count);
    gpu::copy(dev_vector.begin(), dev_vector.end(), host_vector.begin(), state.queue);

    for (const auto& v : host_vector) {
        std::cout << v << "\n";
    }

    int sum_value = 0, min_value = 0;

    return { sum_value, min_value };
}

} // voxel_score
