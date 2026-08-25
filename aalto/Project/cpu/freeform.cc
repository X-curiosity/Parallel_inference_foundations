/*
You can use this function to benchmark your code. You can use the parameter x,
whose value is 1 by default, to avoid the compiler optimizing your code away.
For the same reason, you should return some value which is based on the
computation.
*/

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif



int mul_matrix(int x,const std::vector<int>& m1,
    const std::vector<int>& m2,
    int row1,
    int column1,
    int row2,
    int column2) {

        if (column1 != row2) {
        std::cerr << "Those 2 matrices can't be multiplied!\n";
        std::exit(1);
    }

    std::vector<int> result(row1 * column2, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < column2; j++) {
            int sum = 0;

            for (int k = 0; k < column1; k++) {
                sum += m1[i * column1 + k] * m2[k * column2 + j];
            }

            result[i * column2 + j] = sum;
        }
    }
    return x;
}



