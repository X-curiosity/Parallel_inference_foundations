#include <cub/cub.cuh>
typedef unsigned long long data_t;

void psort(int n, data_t *data) {



    data_t* dataGPU;
    data_t* outputGPU;

    cudaMalloc(&dataGPU, n * sizeof(data_t));
    cudaMalloc(&outputGPU, n * sizeof(data_t));

    cudaMemcpy(
        dataGPU,
        data,
        n * sizeof(data_t),
        cudaMemcpyHostToDevice
    );

    void* tempGPU = nullptr;
    size_t tempBytes = 0;

    cub::DeviceRadixSort::SortKeys(
        tempGPU,
        tempBytes,
        dataGPU,
        outputGPU,
        n
    );

    cudaMalloc(&tempGPU, tempBytes);


    cub::DeviceRadixSort::SortKeys(
        tempGPU,
        tempBytes,
        dataGPU,
        outputGPU,
        n
    );

    // Step 4: retrieve the sorted array.
    cudaMemcpy(
        data,
        outputGPU,
        n * sizeof(data_t),
        cudaMemcpyDeviceToHost
    );


    cudaFree(tempGPU);
    cudaFree(dataGPU);
    cudaFree(outputGPU);

}