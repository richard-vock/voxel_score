#include <test.hpp>
#include <test.cuh>

namespace voxel_score {

std::pair<int, int>
test_func(int* array, int count) {
    int sum_value = 0, min_value = 0;
    test_compute(array, count, &sum_value, &min_value);

    return { sum_value, min_value };
}

} // voxel_score
