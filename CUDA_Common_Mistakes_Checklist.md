# CUDA Mistakes to Avoid

This checklist is for turning a CPU loop-based function into a CUDA implementation.

## 1. Use CUDA kernel syntax exactly

```cpp
__global__ void kernel_name(...) {
    // GPU code
}
```

Use two underscores on each side of `global`.

Kernel launch syntax also needs three angle brackets on each side:

```cpp
kernel_name<<<blocks, threads>>>(arguments);
```

## 2. Keep CPU pointers and GPU pointers separate

```text
CPU pointer: data, result
GPU pointer: dataGPU, resultGPU
```

Allocate GPU memory with `cudaMalloc`, then copy CPU input to it:

```cpp
cudaMalloc(&dataGPU, count * sizeof(float));

cudaMemcpy(dataGPU, data, count * sizeof(float),
           cudaMemcpyHostToDevice);
```

Copy final output back to the CPU with:

```cpp
cudaMemcpy(result, resultGPU, count * sizeof(float),
           cudaMemcpyDeviceToHost);
```

Never pass a normal CPU pointer to a kernel that needs to access its data. Pass the corresponding GPU pointer.

## 3. Pass every array a kernel needs

A kernel cannot automatically see a local C++ variable. Put every GPU array it needs in the parameter list:

```cpp
__global__ void compute(
    const double* normalized,
    float* result,
    int nx,
    int ny)
```

## 4. Decide what one thread owns

Before writing a kernel, finish this sentence:

```text
One thread computes __________________.
```

For correlation:

```text
Normalization: one thread computes one row y.
Correlation:   one thread computes one pair of rows (i, j).
```

Each thread should normally write a unique output location. If multiple threads write one location, you need a reduction or an atomic operation.

## 5. Match the launch shape to the kernel

For one thread per row, use a 1D launch:

```cpp
int threads = 256;
int blocks = (ny + threads - 1) / threads;

normalize_rows<<<blocks, threads>>>(...);
```

For one thread per row pair `(i, j)`, use a 2D launch:

```cpp
dim3 threads(16, 16);
dim3 blocks(
    (ny + threads.x - 1) / threads.x,
    (ny + threads.y - 1) / threads.y
);

compute_corr<<<blocks, threads>>>(...);
```

Do not launch a 2D grid if the kernel ignores one of its dimensions: that causes duplicate work.

## 6. Calculate coordinates correctly

For a 2D kernel:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
```

Do not accidentally use the same expression for both `i` and `j`, or they will always be equal.

Always bounds-check coordinates:

```cpp
if (i >= ny || j >= ny) return;
```

For the lower triangle of a correlation matrix:

```cpp
if (j > i) return;
```

## 7. Allocate using output dimensions

Read the indexing formula.

```cpp
result[i + j * ny]
```

This is a `ny × ny` result, so allocate and copy:

```cpp
cudaMalloc(&resultGPU, ny * ny * sizeof(float));
cudaMemcpy(result, resultGPU, ny * ny * sizeof(float),
           cudaMemcpyDeviceToHost);
```

Do not assume output dimensions equal input dimensions.

## 8. Avoid unnecessary copies between kernels

If one kernel writes `normGPU` and the next kernel reads it, leave it on the GPU:

```text
dataGPU → normalize_rows → normGPU → compute_corr → resultGPU
```

Copy only inputs to the GPU and final outputs back to the CPU.

## 9. Release every GPU allocation

Every `cudaMalloc` needs a `cudaFree`:

```cpp
cudaFree(dataGPU);
cudaFree(normGPU);
cudaFree(resultGPU);
```

## 10. Check for special numerical cases

When normalizing a row, a constant row has zero variance. Avoid dividing by zero:

```cpp
double scale = squared_sum > 0.0
    ? 1.0 / sqrt(squared_sum)
    : 0.0;
```

## Short mental model

```text
CPU: prepares data and starts the GPU work.
GPU: computes using GPU pointers.
cudaMemcpy: moves data between CPU and GPU.

Independent output positions → separate threads.
Work needed to calculate one output → a loop inside that thread.
```
