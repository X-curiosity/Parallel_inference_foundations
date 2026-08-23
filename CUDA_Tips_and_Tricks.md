# CUDA Tips and Tricks

This is a practical reference for translating C++ loops into CUDA code and avoiding common mistakes.

## 1. The basic CUDA hierarchy

```text
Grid (one kernel launch)
└── Blocks
    └── Threads
        └── Warps of 32 threads execute together
```

- A **thread** normally computes one independent output element.
- A **block** is a cooperating group of threads. Its threads can use shared memory and `__syncthreads()`.
- A **warp** is a hardware execution group of 32 threads. Prefer block sizes that are multiples of 32.

Good starting block sizes:

```cpp
// 1D work
int threads = 256;

// 2D work
dim3 threads(16, 16);  // 256 threads total
```

The usual maximum is 1,024 threads per block, but that is a maximum, not an automatic performance recommendation.

## 2. Kernel syntax

```cpp
__global__ void kernel(float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        output[i] = /* calculation */;
    }
}
```

Launch a kernel with three angle brackets:

```cpp
kernel<<<blocks, threads>>>(arguments);
```

```text
__global__       GPU kernel, launched from host/CPU code
__device__       helper callable from GPU code
__host__         helper callable from CPU code
__host__ __device__  helper callable from both
```

Do not write `**global**`: that is Markdown bold formatting, not CUDA syntax.

## 3. One loop: one thread per item

CPU:

```cpp
for (int i = 0; i < n; ++i) {
    output[i] = work(input[i]);
}
```

CUDA:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < n) {
    output[i] = work(input[i]);
}
```

Launch shape:

```cpp
int threads = 256;
int blocks = (n + threads - 1) / threads;
```

`(n + threads - 1) / threads` rounds upward, ensuring that there are enough threads for all `n` elements.

## 4. Double loops: one thread per `(x, y)`

For a 2D array with `nx` columns and `ny` rows:

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x < nx && y < ny) {
    int index = x + y * nx;
    output[index] = work(input[index]);
}
```

Launch it with a 2D grid:

```cpp
dim3 threads(16, 16);
dim3 blocks(
    (nx + threads.x - 1) / threads.x,
    (ny + threads.y - 1) / threads.y
);

kernel<<<blocks, threads>>>(...);
```

## 5. More than two or three loops

The number of loops in CPU code does not have to equal the number of CUDA dimensions.

Ask for each loop index:

```text
Can separate iterations run independently?
```

- **Independent index:** map it to threads, or combine it with another index.
- **Dependent work:** keep it as a sequential loop inside each thread.

For example, matrix multiplication maps one thread to one output cell `(row, col)`, while the sum over `k` remains a loop inside the thread:

```cpp
float sum = 0.0f;

for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
}
```

For rectangle search, the input is 2D but a rectangle has four bounds `(x0, y0, x1, y1)`. A simple mapping is:

```text
one thread → one starting corner (x0, y0)
the thread loops over possible (x1, y1)
```

## 6. CPU memory versus GPU memory

```text
CPU/host pointer:  data, result
GPU/device pointer: dataGPU, resultGPU
```

Typical workflow:

```cpp
float* dataGPU;
float* resultGPU;

cudaMalloc(&dataGPU, n * sizeof(float));
cudaMalloc(&resultGPU, n * sizeof(float));

cudaMemcpy(dataGPU, data, n * sizeof(float),
           cudaMemcpyHostToDevice);

kernel<<<blocks, threads>>>(dataGPU, resultGPU, n);

cudaMemcpy(result, resultGPU, n * sizeof(float),
           cudaMemcpyDeviceToHost);

cudaFree(dataGPU);
cudaFree(resultGPU);
```

Small settings such as sizes and constants can be passed directly:

```cpp
kernel<<<blocks, threads>>>(dataGPU, resultGPU, n, scale);
```

Do not pass a normal CPU pointer to a kernel. A kernel needs a GPU/device pointer for data it accesses.

## 7. `std::vector` and pointers

On the CPU:

```cpp
std::vector<float> values(n);
```

To copy it with CUDA, use:

```cpp
values.data()
```

```cpp
cudaMemcpy(values.data(), valuesGPU, n * sizeof(float),
           cudaMemcpyDeviceToHost);
```

If a function parameter is already `float* result`, use `result` directly. It does not have a `.data()` method.

Do not use `std::vector` inside a `__global__` kernel.

## 8. Shared memory and synchronization

Shared memory is fast memory shared by threads in the **same block**:

```cpp
__shared__ float tile[256];
```

Use it when several threads reuse the same input values. In tiled matrix multiplication, threads computing different output cells reuse the same tile of `A` and `B`.

```cpp
tile[threadIdx.x] = input[index];
__syncthreads();

// All threads in this block may now use tile[] safely.
```

`__syncthreads()` is a block-wide barrier. Every thread in the block waits for every other thread in that block.

Loading a larger tile can reduce barrier overhead:

```text
load 1 value → synchronize → work   (repeated 4 times)
load 4 values → synchronize → work  (one time)
```

The tradeoff is higher shared-memory consumption.

## 9. Avoid races: one thread, one output slot

This is unsafe when many threads execute it:

```cpp
coordinates[0] = x0;
coordinates[1] = y0;
```

Every thread overwrites the same locations.

Instead, give every thread a unique result slot:

```cpp
int id = x + y * nx;
candidates[id] = local_best;
```

For a `nx × ny` grid:

```text
(x=0, y=0) → id 0
(x=1, y=0) → id 1
(x=0, y=1) → id nx
```

Do **not** use multiplication such as `x * y * nx`; it produces duplicate IDs whenever `x` or `y` is zero.

When all threads have written their own candidates, find the global best:

```text
simple approach: copy candidates to CPU and scan them
faster approach: launch a GPU reduction kernel
```

## 10. Dependent stages need separate kernels

CUDA has no simple synchronization across all blocks inside a normal kernel.

If stage B needs all of stage A to finish, use separate kernel launches:

```text
Kernel 1: produce intermediate data
Kernel 2: consume all intermediate data
```

Example: a parallel sorting algorithm typically needs separate stages for histogram counting, prefix scan, and scatter.

## 11. Prefix sums

For a 2D prefix sum:

```cpp
prefix[(x + 1) + width * (y + 1)] =
    value
  + prefix[(x + 1) + width * y]       // top
  + prefix[x + width * (y + 1)]       // left
  - prefix[x + width * y];            // upper-left
```

The formula is:

```text
value + top + left - upper-left
```

It has dependencies on the top and left entries. Do not naively assign arbitrary rows to separate threads; use a sequential implementation first, or a specialized parallel prefix/wavefront algorithm.

## 12. Debugging checklist

- Bounds-check every thread index before accessing arrays.
- Allocate output according to the output indexing formula.
  - `result[i + j * ny]` needs `ny * ny` elements.
- Initialize memory if you later copy/read entries that the kernel did not write.
  - Example: `cudaMemset(resultGPU, 0, ny * ny * sizeof(float));`
- Every `cudaMalloc` needs a matching `cudaFree`.
- Check errors after a kernel while debugging:

```cpp
kernel<<<blocks, threads>>>(...);

cudaError_t err = cudaDeviceSynchronize();

if (err != cudaSuccess) {
    printf("Kernel failed: %s\n", cudaGetErrorString(err));
}
```

- Use `cudaMemcpyHostToDevice` for CPU → GPU, and `cudaMemcpyDeviceToHost` for GPU → CPU.

## 13. Small device helper functions

Use `__device__` for small GPU-only helpers:

```cpp
__device__ float rectangle_sum(
    const float* prefix,
    int width,
    int x0, int y0,
    int x1, int y1)
{
    return prefix[x1 + width * y1]
         - prefix[x1 + width * y0]
         - prefix[x0 + width * y1]
         + prefix[x0 + width * y0];
}
```

The compiler will often inline small helpers. Keep allocation, vectors, file I/O, and CUDA launches in host/CPU code.

## 14. Loop unrolling

For a small fixed-size loop, `#pragma unroll` asks the compiler to expand it:

```cpp
#pragma unroll
for (int k = 0; k < 16; ++k) {
    sum += a[k] * b[k];
}
```

It can reduce loop overhead, but it can increase code size and register use. Use it for short compile-time-constant loops—not large loops whose bounds are only known at runtime.

## 15. CUB radix sorting

For production CUDA sorting, prefer CUB instead of writing a full radix sort from scratch:

```cpp
#include <cub/cub.cuh>
```

`cub::DeviceRadixSort::SortKeys` is called from host/CPU code, not inside a kernel. It first tells you the amount of temporary GPU memory needed; allocate that storage, then call it again to perform the sort.

Remember that CUB is template-heavy and can increase compilation time. Check whether a course assignment permits external CUDA libraries before using it.

## Core mental model

```text
Independent output locations → separate GPU threads
Work required for one output → sequential loop in that thread
Reusable block-local input → shared memory
Many threads contributing to one result → reduction or atomics
Dependent global stages → separate kernels
```
