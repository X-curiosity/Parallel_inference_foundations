#include <cmath>
#include <vector>

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/

Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    std::vector<double> prefix((ny + 1) * (nx + 1) * 3, 0.0);
    std::vector<double> prefix_sq((ny + 1) * (nx + 1) * 3, 0.0);

    auto p = [nx](int y, int x, int c) {
        return c + 3 * x + 3 * (nx + 1) * y;
    };

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int c = 0; c < 3; c++) {
                double value = data[c + 3 * x + 3 * nx * y];

                prefix[p(y + 1, x + 1, c)] =
                    value
                    + prefix[p(y, x + 1, c)]
                    + prefix[p(y + 1, x, c)]
                    - prefix[p(y, x, c)];

                prefix_sq[p(y + 1, x + 1, c)] =
                    value * value
                    + prefix_sq[p(y, x + 1, c)]
                    + prefix_sq[p(y + 1, x, c)]
                    - prefix_sq[p(y, x, c)];
            }
        }
    }

    double total[3];
    double total_sq[3];
    for (int c = 0; c < 3; c++) {
        total[c] = prefix[p(ny, nx, c)];
        total_sq[c] = prefix_sq[p(ny, nx, c)];
    }

    double best_cost = INFINITY;
    int total_area = ny * nx;

    for (int y0 = 0; y0 < ny; y0++) {
        for (int y1 = y0 + 1; y1 <= ny; y1++) {
            for (int x0 = 0; x0 < nx; x0++) {
                for (int x1 = x0 + 1; x1 <= nx; x1++) {
                    int inner_area = (x1 - x0) * (y1 - y0);
                    int outer_area = total_area - inner_area;

                    if (outer_area == 0) {
                        continue;
                    }

                    double inner[3];
                    double outer[3];
                    double cost = 0.0;

                    for (int c = 0; c < 3; c++) {
                        double inner_sum =
                            prefix[p(y1, x1, c)]
                            - prefix[p(y0, x1, c)]
                            - prefix[p(y1, x0, c)]
                            + prefix[p(y0, x0, c)];

                        double inner_sq =
                            prefix_sq[p(y1, x1, c)]
                            - prefix_sq[p(y0, x1, c)]
                            - prefix_sq[p(y1, x0, c)]
                            + prefix_sq[p(y0, x0, c)];

                        double outer_sum = total[c] - inner_sum;
                        double outer_sq = total_sq[c] - inner_sq;

                        inner[c] = inner_sum / inner_area;
                        outer[c] = outer_sum / outer_area;

                        cost += inner_sq - inner_sum * inner_sum / inner_area;
                        cost += outer_sq - outer_sum * outer_sum / outer_area;
                    }

                    if (cost < best_cost) {
                        best_cost = cost;
                        result.y0 = y0;
                        result.y1 = y1;
                        result.x0 = x0;
                        result.x1 = x1;

                        for (int c = 0; c < 3; c++) {
                            result.inner[c] = static_cast<float>(inner[c]);
                            result.outer[c] = static_cast<float>(outer[c]);
                        }
                    }
                }
            }
        }
    }

    return result;
}
