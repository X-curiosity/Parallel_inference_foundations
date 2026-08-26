#include "freeform.h"
#include "ppc.h"

#include <chrono>
#include <ctime>
#include <thread>

int main(int argc, const char **argv) {
    const char *ppc_output = std::getenv("PPC_OUTPUT");
    int ppc_output_fd = 0;
    if (ppc_output) {
        ppc_output_fd = std::stoi(ppc_output);
    }
    if (ppc_output_fd <= 0) {
        ppc_output_fd = 1;
    }
    std::unique_ptr<ppc::fdostream> stream = std::unique_ptr<ppc::fdostream>(new ppc::fdostream(ppc_output_fd));

    argc--;
    argv++;
    bool correctness_test = false;
    if (argc == 2 && std::string(argv[0]) == "--test") {
        correctness_test = true;
        argc--;
        argv++;
    }
    if (argc != 1) {
        std::cerr << "Invalid usage" << std::endl;
        return 1;
    }

    std::ifstream input_file(argv[0]);
    if (!input_file) {
        std::cerr << "Failed to open input file" << std::endl;
        return 2;
    }

    std::string timeout;
    CHECK_READ(input_file >> timeout);
    if (timeout == "timeout") {
        input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    } else {
        std::cerr << "Missing timeout" << std::endl;
        return 1;
    }

    if (correctness_test) {
        int rows;
        int inner;
        int columns;
        CHECK_READ(input_file >> rows >> inner >> columns);
        if (rows <= 0 || inner <= 0 || columns <= 0) {
            std::cerr << "Matrix dimensions must be positive" << std::endl;
            return 3;
        }

        std::vector<float> matrix1(rows * inner);
        std::vector<float> matrix2(inner * columns);
        std::vector<float> expected(rows * columns);
        std::vector<float> actual(rows * columns);
        for (float& value : matrix1) CHECK_READ(input_file >> value);
        for (float& value : matrix2) CHECK_READ(input_file >> value);
        for (float& value : expected) CHECK_READ(input_file >> value);
        CHECK_END(input_file);

        ppc::perf timer;
        timer.start();
        multiply_matrices(
            rows, inner, columns,
            matrix1.data(), matrix2.data(), actual.data()
        );
        timer.stop();
        timer.print_to(*stream);

        for (std::size_t i = 0; i < actual.size(); i++) {
            float tolerance = 1.0e-4f * std::max(1.0f, std::abs(expected[i]));
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance) {
                std::cerr << "Incorrect result at output index " << i
                          << ": expected " << expected[i]
                          << ", got " << actual[i] << std::endl;
                return 4;
            }
        }

        *stream << "result\tdone\n";
        *stream << "input\t0\n";
        *stream << "output\t1\n";
    } else {
        int rows;
        int inner;
        int columns;
        std::uint32_t seed;
        std::uint64_t expected_hash;
        CHECK_READ(input_file >> rows >> inner >> columns);
        CHECK_READ(input_file >> seed);
        CHECK_READ(input_file >> expected_hash);
        CHECK_END(input_file);

        ppc::perf timer;
        std::clock_t cpu_start = std::clock();
        auto wall_start = std::chrono::steady_clock::now();
        timer.start();
        std::uint64_t actual_hash = benchmark_matrices(
            rows, inner, columns, seed
        );
        timer.stop();
        auto wall_end = std::chrono::steady_clock::now();
        std::clock_t cpu_end = std::clock();
        timer.print_to(*stream);

        if (actual_hash != expected_hash) {
            std::cerr << "Incorrect benchmark result: expected hash "
                      << expected_hash << ", got " << actual_hash << std::endl;
            return 5;
        }

        double wall_seconds = std::chrono::duration<double>(
            wall_end - wall_start
        ).count();
        double cpu_seconds = static_cast<double>(cpu_end - cpu_start)
                           / CLOCKS_PER_SEC;
        double average_cores = wall_seconds > 0.0
                             ? cpu_seconds / wall_seconds : 0.0;
        std::uint64_t output_elements =
            static_cast<std::uint64_t>(rows) * columns;
        std::uint64_t operations = output_elements
            * (2ULL * static_cast<std::uint64_t>(inner) - 1ULL);
        double gflops = wall_seconds > 0.0
                      ? static_cast<double>(operations) / wall_seconds / 1.0e9
                      : 0.0;

        *stream << "metric_dimensions\t" << rows << 'x' << inner
                << " * " << inner << 'x' << columns << '\n';
        *stream << "metric_operations\t" << operations << '\n';
        *stream << "metric_cpu_seconds\t" << cpu_seconds << '\n';
        *stream << "metric_average_cores\t" << average_cores << '\n';
        *stream << "metric_logical_cores\t"
                << std::thread::hardware_concurrency() << '\n';
        *stream << "metric_gflops\t" << gflops << '\n';

        *stream << "result\tdone\n";
        *stream << "input\t0\n";
        *stream << "output\t1\n";
    }
    *stream << std::endl;
}
