/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/

/*
INSTRUCTIONS:
Parallelize your solution to CP1 with the help of vector operations so that you can perform multiple useful arithmetic operations with one instruction. Do not use any other form of parallelism yet in this exercise.
Please do all arithmetic with double-precision floating-point numbers.
For this technical exercise, we have disabled auto-vectorization.
*/

#include <cmath>
#include <vector>

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

void correlate(int ny, int nx, const float *data, float *result)
{
    constexpr int lanes = 4;
    std::vector<double> normalized(ny * nx);

    for (int y = 0; y < ny; y++) {
        double4_t sum_vector = {0.0, 0.0, 0.0, 0.0};
        int x = 0;

        for (; x + lanes <= nx; x += lanes) {
            double4_t values = {
                static_cast<double>(data[x + y * nx]),
                static_cast<double>(data[x + 1 + y * nx]),
                static_cast<double>(data[x + 2 + y * nx]),
                static_cast<double>(data[x + 3 + y * nx])
            };
            sum_vector += values;
        }

        double sum = sum_vector[0] + sum_vector[1] +
                     sum_vector[2] + sum_vector[3];

        for (; x < nx; x++) {
            sum += static_cast<double>(data[x + y * nx]);
        }

        double mean = sum / static_cast<double>(nx);
        double4_t mean_vector = {mean, mean, mean, mean};
        double4_t squared_sum_vector = {0.0, 0.0, 0.0, 0.0};
        double squared_sum = 0.0;
        x = 0;

        for (; x + lanes <= nx; x += lanes) {
            double4_t values = {
                static_cast<double>(data[x + y * nx]),
                static_cast<double>(data[x + 1 + y * nx]),
                static_cast<double>(data[x + 2 + y * nx]),
                static_cast<double>(data[x + 3 + y * nx])
            };
            double4_t centered = values - mean_vector;

            normalized[x + y * nx] = centered[0];
            normalized[x + 1 + y * nx] = centered[1];
            normalized[x + 2 + y * nx] = centered[2];
            normalized[x + 3 + y * nx] = centered[3];

            squared_sum_vector += centered * centered;
        }

        squared_sum = squared_sum_vector[0] + squared_sum_vector[1] +
                      squared_sum_vector[2] + squared_sum_vector[3];

        for (; x < nx; x++) {
            double centered = static_cast<double>(data[x + y * nx]) - mean;
            normalized[x + y * nx] = centered;
            squared_sum += centered * centered;
        }

        double scale = 1.0 / std::sqrt(squared_sum);
        double4_t scale_vector = {scale, scale, scale, scale};
        x = 0;

        for (; x + lanes <= nx; x += lanes) {
            double4_t values = {
                normalized[x + y * nx],
                normalized[x + 1 + y * nx],
                normalized[x + 2 + y * nx],
                normalized[x + 3 + y * nx]
            };
            values *= scale_vector;

            normalized[x + y * nx] = values[0];
            normalized[x + 1 + y * nx] = values[1];
            normalized[x + 2 + y * nx] = values[2];
            normalized[x + 3 + y * nx] = values[3];
        }

        for (; x < nx; x++) {
            normalized[x + y * nx] *= scale;
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double4_t correlation_vector = {0.0, 0.0, 0.0, 0.0};
            int x = 0;

            for (; x + lanes <= nx; x += lanes) {
                double4_t left = {
                    normalized[x + i * nx],
                    normalized[x + 1 + i * nx],
                    normalized[x + 2 + i * nx],
                    normalized[x + 3 + i * nx]
                };
                double4_t right = {
                    normalized[x + j * nx],
                    normalized[x + 1 + j * nx],
                    normalized[x + 2 + j * nx],
                    normalized[x + 3 + j * nx]
                };
                correlation_vector += left * right;
            }

            double correlation = correlation_vector[0] +
                                 correlation_vector[1] +
                                 correlation_vector[2] +
                                 correlation_vector[3];

            for (; x < nx; x++) {
                correlation += normalized[x + i * nx] *
                               normalized[x + j * nx];
            }

            result[i + j * ny] = static_cast<float>(correlation);
        }
    }
}
