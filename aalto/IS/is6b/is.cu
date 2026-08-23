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


struct Candidate {
    float cost;
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};


__global__ void best_cost(float *prefix,int nx,int ny, Candidate* candidates){


  int x0 = blockIdx.x * blockDim.x + threadIdx.x;
  int y0 = blockIdx.y * blockDim.y + threadIdx.y;

  float total = prefix[nx + (nx + 1) * ny];
  int total_area = ny * nx;

  if (x0 >= nx || y0 >= ny) return;

  int id = x0 + y0 *nx;

  Candidate best;
  best.cost = INFINITY;
  
  for (int y1 = y0 + 1; y1 <= ny; ++y1){
    for (int x1 = x0 + 1; x1 <= nx; ++x1) {



      int inner_area = (x1 - x0) * (y1 - y0);
      int outer_area = total_area - inner_area;

      if(outer_area == 0) {
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


      float outer_sum = total - inner_sum;

      inner = inner_sum / inner_area;
      outer = outer_sum / outer_area;



      for (int c = 0; c < 3; c++) {
        cost += inner_sum - inner_sum * inner_sum / inner_area;
        cost += outer_sum - outer_sum * outer_sum / outer_area;
      }

      if (cost < best.cost) {

          best.cost = cost;
          best.x0 = x0;
          best.y0 = y0;
          best.x1 = x1;
          best.y1 = y1;
          
          
          best.inner[0] = inner;
          best.inner[1] = inner;
          best.inner[2] = inner;

          best.outer[0] = outer;
          best.outer[1] = outer;
          best.outer[2] = outer;
        
      }
    }
  }

  candidates[id] = best;
}



Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    std::vector<float> prefix((ny + 1) * (nx + 1), 0.0);
    std::vector<Candidate> candidates(nx * ny);


    Candidate *candidatesGPU;
    float *prefixGPU;
    
    float cost = INFINITY;


    //Prefix Computation

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

    dim3 costThreads(16,16);
    
    dim3 costBlocks(
    (nx +costThreads.x - 1) / costThreads.x,
    (ny + costThreads.y - 1) / costThreads.y
    );

    // Allocate vector for GPU 

    cudaMalloc(&prefixGPU, (ny+1)*(nx + 1)*sizeof(float));
    cudaMalloc(&candidatesGPU, nx*ny*sizeof(Candidate));
    
    
    // Copy Input Vectors : CPU -> GPU 
    cudaMemcpy(prefixGPU, prefix.data(), (ny+1)*(nx+1)*sizeof(float),cudaMemcpyHostToDevice);


    

    cudaError_t err = cudaDeviceSynchronize();
    

    // Copy Input Vectors : GPU -> CPU
    cudaMemcpy(candidates.data(),candidatesGPU,nx*ny*sizeof(Candidate),cudaMemcpyDeviceToHost);

    for (int i = 0; i < nx * ny; i++) {
        if (candidates[i].cost < cost) {
          cost = candidates[i].cost;
          result.x0 = candidates[i].x0;
          result.x1 = candidates[i].x1;
          result.y0 = candidates[i].y0;
          result.y1 = candidates[i].y1;
          result.inner[0] = candidates[i].inner[0];
          result.inner[1] = candidates[i].inner[1];
          result.inner[2] = candidates[i].inner[2];
          result.outer[0] = candidates[i].outer[0];
          result.outer[1] = candidates[i].outer[1];
          result.outer[2] = candidates[i].outer[2];
            
        }
    }


    cudaFree(prefixGPU);
    cudaFree(candidatesGPU);
                    
                        
    return result;

}