#include "include/macros.cuh"
#include "include/cuda_helpers.cuh"

int main() {
    int accelerator_count = device_count();
    print_device_info();
    return 0;
}