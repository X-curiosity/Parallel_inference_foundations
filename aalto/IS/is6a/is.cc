#include <cmath>
#include <vector>
#include <omp.h>

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};


Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    std::vector<float> prefix((ny + 1) * (nx + 1), 0.0);
   
    float total[3];
    float best_cost = INFINITY;
    int total_area = ny * nx;


    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++){ 
            float value = data[3 * x + 3 * nx * y];
            prefix[(x+1) + (nx + 1) * (y+1)] =
            value
            + prefix[(x+1) + (nx + 1) * y]
            + prefix[x + (nx + 1) * (y+1)]
            - prefix[x + (nx + 1) * y];  
       
            
        }
    }
    

    
    
    

        
        
    total[0] = prefix[nx + (nx + 1) * ny];
    total[1] = total[0];
    total[2] = total[0];
        
        
    #pragma omp parallel for
    for (int x0 = 0; x0 < nx; x0++) {
        for (int y0 = 0; y0 < ny; y0++) {
            for (int y1 = y0+1; y1 <= ny; y1++){
                for (int x1 = x0 + 1; x1 <= nx; x1++) {
                    int inner_area = (x1 - x0) * (y1 - y0);
                    int outer_area = total_area - inner_area;
                        
                    if (outer_area == 0) {
                        continue;
                    }
                        
                    float inner;
                    float outer;
                    float cost = 0.0;
                        
                        
                    float inner_sum =
                    prefix[x1 + (nx + 1) * y1]
                    - prefix[x1 + (nx + 1) * y0]
                    - prefix[x0 + (nx + 1) * y1]
                    + prefix[x0 + (nx + 1) * y0];


                    float outer_sum = total[0] - inner_sum;



                    inner = inner_sum / inner_area;
                    outer = outer_sum / outer_area;



                    for (int c = 0; c < 3; c++) {
                        cost += inner_sum - inner_sum * inner_sum / inner_area;
                        cost += outer_sum - outer_sum * outer_sum / outer_area;
                    }
                        
                        
                        
                    if (cost < best_cost) {
                        best_cost = cost;
                        result.y0 = y0;
                        result.y1 = y1;
                        result.x0 = x0;
                        result.x1 = x1;
                            
                        for (int c = 0; c < 3; c++) {
                            result.inner[c] = static_cast<float>(inner);
                            result.outer[c] = static_cast<float>(outer);
                        }
                    }
                }
            }
        }
        
    }

    return result;

}

    
