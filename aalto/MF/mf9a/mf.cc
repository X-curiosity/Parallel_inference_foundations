/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
  int max_size = (2 * hx + 1) * (2 * hy + 1);

  #pragma omp parallel for schedule(dynamic)
  for(int y = 0; y < ny; y++){
    std::vector<float> window;
    window.reserve(max_size);

    int y_start = std::max(0, y - hy);
    int y_end   = std::min(ny, y + hy + 1);

    int old_x_start = std::max(0, 0 - hx);
    int old_x_end   = std::min(nx, 0 + hx + 1);

    for (int j = y_start; j < y_end; j++){
      for (int i = old_x_start; i < old_x_end; i++){
        window.push_back(in[i + j * nx]);
      }
    }

    std::sort(window.begin(), window.end());

    for(int x = 0; x < nx; x++){
      int x_start = std::max(0, x - hx);
      int x_end   = std::min(nx, x + hx + 1);

      if (x != 0) {
        for (int i = old_x_start; i < x_start; i++) {
          for (int j = y_start; j < y_end; j++) {
            auto position = std::lower_bound(window.begin(), window.end(), in[i + j * nx]);
            window.erase(position);
          }
        }

        for (int i = old_x_end; i < x_end; i++) {
          for (int j = y_start; j < y_end; j++) {
            float value = in[i + j * nx];
            auto position = std::upper_bound(window.begin(), window.end(), value);
            window.insert(position, value);
          }
        }
      }

      int size = static_cast<int>(window.size());

      if(size % 2 == 0){
        out[x + y * nx] = (window[size / 2 - 1] + window[size / 2]) / 2;
      }else{
        out[x + y * nx] = window[size / 2];
      }

      old_x_start = x_start;
      old_x_end = x_end;
    }
  }
}
