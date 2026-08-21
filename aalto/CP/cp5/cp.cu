#include <cmath>
#include <vector>
#include <iostream>

__global__ void compute_corr(float*result,double*normalized , int nx ,int ny){

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  

  if  (i < ny && j < ny && j <= i) {
    double correlation_x = 0.0;

    for (int x = 0; x < nx; ++x){
      correlation_x +=normalized[x + i * nx] *normalized[x + j * nx];
    }

    result[i + j * ny] = static_cast<float>(correlation_x);

    
    }


}


__global__ void normalize_rows(
    double* normalized,
    const float* data,
    int nx,
    int ny)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= ny) return;

    double sum = 0.0;

    for (int x = 0; x < nx; ++x) {
        sum += data[x + y * nx];
    }

    double mean = sum / nx;
    double squared_sum = 0.0;

    for (int x = 0; x < nx; ++x) {
        double centered = data[x + y * nx] - mean;
        normalized[x + y * nx] = centered;
        squared_sum += centered * centered;
    }

    double scale = 1.0 / sqrt(squared_sum);

    for (int x = 0; x < nx; ++x) {
        normalized[x + y * nx] *= scale;
    }
}

void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<double> normalized(ny * nx,0);

    float *dataGPU;
    float *resultGPU;
    double *normGPU;

    int normThreads = 256;
    int normBlocks = (ny + normThreads - 1) / normThreads;




    // Allocate vector for GPU 
    cudaMalloc(&normGPU, nx*ny*sizeof(double));
    cudaMalloc(&dataGPU, nx*ny*sizeof(float));
    cudaMalloc(&resultGPU, ny*ny*sizeof(float));
    cudaMemset(resultGPU, 0, ny * ny * sizeof(float));



    // Copy Input Vectors : CPU -> GPU 
    cudaMemcpy(dataGPU, data, nx*ny*sizeof(float), cudaMemcpyHostToDevice);

    
    


    normalize_rows<<<normBlocks, normThreads>>>(normGPU,dataGPU,nx,ny);

    cudaError_t err = cudaDeviceSynchronize();
    
    
    if (err != cudaSuccess) {
      printf("normalize_rows failed: %s\n", cudaGetErrorString(err));
    }
    
    


    //Correlation Computation 

    dim3 corrThreads(16,16);
    
    dim3 corrBlocks(
    (ny + corrThreads.x - 1) / corrThreads.x,
    (ny + corrThreads.y - 1) / corrThreads.y
    );


 


    compute_corr<<<corrBlocks,corrThreads>>>(resultGPU,normGPU,nx,ny);

    cudaError_t err1 = cudaDeviceSynchronize();
    
    
    if (err1 != cudaSuccess) {
      printf("normalize_rows failed: %s\n", cudaGetErrorString(err1));
    }
    
    


     // Copy Output Vectors : GPU -> CPU 

    cudaMemcpy(result,resultGPU,ny*ny*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(dataGPU);
    cudaFree(normGPU);
    cudaFree(resultGPU);

}
