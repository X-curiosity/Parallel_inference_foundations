/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- the correlation between rows i and j has to be stored in result[i + j*ny]
- only elements with 0 <= j <= i < ny need to be filled
*/


/*

INSTRCUTIONS :

Parallelize your solution to CP1 with the help of OpenMP and multithreading so that you are exploiting multiple CPU cores in parallel. Do not use any other form of parallelism yet in this exercise. Please do all arithmetic with double-precision floating-point numbers.
For this technical exercise, we have disabled auto-vectorization.

*/


/* GRADING : 

In this task your submission will be graded using benchmarks/2: the input contains 4000 × 1000 pixels, and the output should contain 4000 × 4000 pixels.

The point thresholds are as follows. If you submit your solution no later than on Sunday, 3 May 2026, at 23:59:59 (Helsinki), your score will be:

Running time	Points
≤ 4.000 s	1
≤ 1.000 s	2
≤ 0.700 s	3
If you submit your solution after the deadline, but before the course ends on Sunday, 31 May 2026, at 23:59:59 (Helsinki), your score will be:

Running time	Points
≤ 2.000 s	1
≤ 0.700 s	2


*/
#include <cmath>
#include <vector>
#include <omp.h>


void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<double> normalized(ny * nx);

    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        double sum = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (int x = 0; x < nx; x++) {
            sum += static_cast<double>(data[x + y * nx]);
        }

        double mean = sum / nx;
        double squared_sum = 0.0;

        std::vector<double> centered(nx,0);

        
        for (int x = 0; x < nx; x++) {
            centered[x] = static_cast<double>(data[x + y * nx]) - mean;

            normalized[x + y * nx] = centered[x];
            squared_sum += centered[x] * centered[x];
        }

        double scale = 1.0 / std::sqrt(squared_sum);

        for (int x = 0; x < nx; x++) {
            normalized[x + y * nx] *= scale;
        }
    }

    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double correlation = 0.0;

            #pragma omp parallel for reduction(+:correlation)
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








