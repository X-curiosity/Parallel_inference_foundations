#pragma once

#include <cstdint>

void multiply_matrices(
    int rows,
    int inner,
    int columns,
    const float *matrix1,
    const float *matrix2,
    float *result
);

std::uint64_t benchmark_matrices(
    int rows,
    int inner,
    int columns,
    std::uint32_t seed
);

int run(int x);
