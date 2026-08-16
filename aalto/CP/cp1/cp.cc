/*
Instructions : Implement a simple sequential baseline solution. Do not try to use any form of parallelism yet;
try to make it work correctly first. 
Please do all arithmetic with double-precision floating-point numbers.
For this initial exercise, we have disabled auto-vectorization.
*/

/*
Grading : In this task your submission will be graded using benchmarks/2: 
the input contains 4000 × 1000 pixels, and the output should contain 4000 × 4000 pixels.
The point thresholds are as follows. If you submit your solution no later than on Sunday, 
26 April 2026, at 23:59:59 (Helsinki), your score will be 
≤ 20.000 s	1
≤ 16.000 s	2
≤ 12.000 s	3
≤ 10.000 s	4
≤ 9.000 s	5
*/


/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/



#include <cmath>
#include <vector>

void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<double> normalized(ny * nx);

    for (int y = 0; y < ny; y++) {
        double sum = 0.0;

        for (int x = 0; x < nx; x++) {
            sum += static_cast<double>(data[x + y * nx]);
        }

        double mean = sum / nx;
        double squared_sum = 0.0;

        for (int x = 0; x < nx; x++) {
            double centered =
                static_cast<double>(data[x + y * nx]) - mean;

            normalized[x + y * nx] = centered;
            squared_sum += centered * centered;
        }

        double scale = 1.0 / std::sqrt(squared_sum);

        for (int x = 0; x < nx; x++) {
            normalized[x + y * nx] *= scale;
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double correlation = 0.0;

            for (int x = 0; x < nx; x++) {
                correlation +=
                    normalized[x + i * nx] *
                    normalized[x + j * nx];
            }

            result[i + j * ny] =
                static_cast<float>(correlation);
        }
    }
}








