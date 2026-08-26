/*
You can use this function to benchmark your code. You can use the parameter x,
whose value is 1 by default, to avoid the compiler optimizing your code away.
For the same reason, you should return some value which is based on the
computation.
*/

#include "freeform.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif



static std::vector<float> create_matrix(int rows, int columns, std::uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(1.0f, 1000.0f);

    std::vector<float> matrix(rows * columns);
    for (int i = 0; i < rows * columns; i++) {
        matrix[i] = dist(rng);
    }
    return matrix;
}

std::uint64_t benchmark_matrices(
    int rows,
    int inner,
    int columns,
    std::uint32_t seed
) {
    std::vector<float> matrix1 = create_matrix(rows, inner, seed);
    std::vector<float> matrix2 = create_matrix(inner, columns, seed);
    std::vector<float> result(rows * columns);

    multiply_matrices(
        rows, inner, columns,
        matrix1.data(), matrix2.data(), result.data()
    );

    std::uint64_t hash = 1469598103934665603ULL;
    for (float value : result) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void multiply_matrices(
    int rows,
    int inner,
    int columns,
    const float *matrix1,
    const float *matrix2,
    float *result
) {
    #pragma omp parallel for schedule(dynamic,4)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            float sum = 0.0f;
            for (int k = 0; k < inner; k++) {
                sum += matrix1[i * inner + k] * matrix2[k * columns + j];
            }
            result[i * columns + j] = sum;
        }
    }
}

struct Shape {
    int row1;
    int column1;
    int row2;
    int column2;
};

static Shape case_shape(int x)
{
    if (x < 0) {
        x = 0;
    }

    if (x < 100) {
        static const Shape base_cases[20] = {
            {24, 24, 24, 24},    {48, 16, 16, 64},
            {16, 64, 64, 48},    {31, 47, 47, 29},
            {64, 32, 32, 16},    {32, 96, 96, 32},
            {96, 32, 32, 96},    {45, 63, 63, 37},
            {128, 16, 16, 128},  {16, 128, 128, 16},
            {72, 48, 48, 80},    {80, 72, 72, 48},
            {53, 89, 89, 67},    {112, 40, 40, 104},
            {40, 112, 112, 104}, {96, 96, 96, 96},
            {127, 65, 65, 33},   {65, 127, 127, 97},
            {144, 48, 48, 80},   {80, 144, 144, 112}
        };
        static const int scales[5] = {1, 2, 3, 4, 6};

        Shape shape = base_cases[x % 20];
        int scale = scales[x / 20];
        shape.row1 *= scale;
        shape.column1 *= scale;
        shape.row2 *= scale;
        shape.column2 *= scale;
        return shape;
    }

    switch (x - 100) {
    case 0:
        return {1536, 1536, 1536, 1536};
    case 1:
        return {2048, 2048, 2048, 2048};
    case 2:
        return {4096, 1024, 1024, 1024};
    case 3:
        return {8192, 512, 512, 2048};
    case 4:
        return {2560, 2560, 2560, 2560};
    case 5:
        return {16384, 768, 768, 1536};
    case 6:
        return {2816, 2816, 2816, 2816};
    case 7:
        return {24576, 768, 768, 1536};
    case 8:
        return {8192, 1536, 1536, 2048};
    default:
        return {32768, 512, 512, 1536};
    }
}

int run(int x)
{
    Shape shape = case_shape(x);

    std::uint64_t hash = benchmark_matrices(
        shape.row1,
        shape.column1,
        shape.column2,
        123456789u
    );
    return static_cast<int>(hash ^ (hash >> 32) ^ static_cast<std::uint32_t>(x));
}
