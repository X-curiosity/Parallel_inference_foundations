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

  #pragma omp parallel for schedule(dynamic, 4)
  for(int x = 0; x < nx; x++){
    for(int y = 0; y < ny; y++){


      int x_start = std::max(0, x - hx);
      int x_end   = std::min(nx, x + hx + 1);
      int y_start = std::max(0, y - hy);
      int y_end   = std::min(ny, y + hy + 1);
      
      int size = (x_end - x_start) * (y_end - y_start);
      int count = 0;

      std::vector<float> window(size,0);
     
     
     // Filling in the sliding window
      for (int j = y_start; j < y_end; j++){
        for (int i = x_start; i < x_end; i++){
          window[count] = in[i + j * nx];
          count++;
        }
      }
      std::sort(window.begin(), window.end()); // Sorting the matrix/vector
     
  
        

      if(size % 2 == 0){
        int index1 = size/2-1;
        int index2 = size/2;
        out[x + y*nx] = (window[index1] + window[index2])/2;
      }else{
        int index = size/2 + 1 - 1;
        out[x + y*nx] =  window[index];
      }

    }
    
  }
}

