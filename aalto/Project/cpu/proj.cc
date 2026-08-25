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

std::vector<int> rows1 = {
    16, 24, 32, 40, 48, 64,
    72, 80, 96, 112, 128, 144,
    160, 192, 224, 256, 288, 320,
    384, 448, 512, 576, 640, 768
};

std::vector<int> columns1 = {
    24, 32, 48, 64, 80, 96,
    128, 160, 192, 224, 256, 288,
    320, 384, 448, 512, 576, 640,
    768, 896, 1024, 1152, 1280, 1536
};

std::vector<int> rows2 = {
    24, 32, 48, 64, 80, 96,
    128, 160, 192, 224, 256, 288,
    320, 384, 448, 512, 576, 640,
    768, 896, 1024, 1152, 1280, 1536
};

std::vector<int> columns2 = {
    12, 16, 24, 32, 40, 48,
    64, 96, 128, 160, 192, 224,
    256, 288, 320, 384, 448, 512,
    576, 640, 768, 832, 896, 1024
};

int gen_rand(int start, int end)
{
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(start, end);
    return dist(rng);
}

std::vector<int> create_matrix(int rows, int columns)
{
    std::vector<int> matrix(rows * columns);

    for (int i = 0; i < rows * columns; i++) {
        matrix[i] = gen_rand(10, 1000);
    }

    return matrix;
}

std::vector<int> mul_matrix(
    const std::vector<int>& m1,
    const std::vector<int>& m2,
    int row1,
    int column1,
    int row2,
    int column2
) {
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

    return result;
}

void show_matrix(const std::vector<int>& matrix, int rows, int columns)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            std::cout << matrix[j + i * columns];

            if (j == columns - 1) {
                std::cout << '\n';
            } else {
                std::cout << ' ';
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <label>\n";
        return 1;
    }

    std::string label = argv[1];
    std::string filename = label + ".json";

    std::ofstream file(filename);

    if (!file) {
        std::cerr << "Could not open output file.\n";
        return 1;
    }

    int length = static_cast<int>(columns1.size());

    file << "[\n";

    for (int i = 0; i < length; i++) {
        int row1 = rows1[i];
        int column1 = columns1[i];
        int row2 = rows2[i];
        int column2 = columns2[i];

        std::vector<int> matrix1 = create_matrix(row1, column1);
        std::vector<int> matrix2 = create_matrix(row2, column2);

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int> final_mat =
            mul_matrix(matrix1, matrix2, row1, column1, row2, column2);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        double time_taken = elapsed.count();

        int nbr_of_ops = row1 * column2 * (column1 + 2);

        file << "  {\n";
        file << "    \"m1 size\": \"" << row1 << "x" << column1 << "\",\n";
        file << "    \"m2 size\": \"" << row2 << "x" << column2 << "\",\n";
        file << "    \"final_m size\": \"" << row1 << "x" << column2 << "\",\n";
        file << "    \"nb_operations\": " << nbr_of_ops << ",\n";
        file << "    \"inference time (in sec)\": " << time_taken << "\n";
        file << "  }";

        if (i != length - 1) {
            file << ",";
        }

        file << "\n";
    }

    file << "]\n";

    std::cout << "Results written to " << filename << "\n";

    return 0;
}