#include <iostream>
#include <test.hpp>

#include <pcl/gpu/containers/initialization.h>

int main (int argc, char const* argv[]) {
    pcl::gpu::setDevice(0);

    int* test_array = new int[4];
    test_array[0] = 3;
    test_array[1] = 1;
    test_array[2] = 4;
    test_array[3] = 1;
    std::pair<int, int> bla = voxel_score::test_func(test_array, 4);
    std::cout << bla.first << "\n";
    std::cout << bla.second << "\n";
}
