# CUDA: Thinking About Loops and Parallelization

The central rule is: you do **not** automatically translate each CPU loop into a CUDA loop.

First decide which loop indices are independent. Make those indices GPU thread coordinates. Keep only work that must happen in order as ordinary loops inside each thread.

## One loop

CPU:

```cpp
for (int i = 0; i < N; ++i) {
    output[i] = work(input[i]);
}
```

CUDA: one thread per `i`.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) {
    output[i] = work(input[i]);
}
```

---

## Double loop

CPU:

```cpp
for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
        output[x + y * nx] = work(input[x + y * nx]);
    }
}
```

CUDA: one thread per pair `(y, x)`.

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x < nx && y < ny) {
    int index = x + y * nx;
    output[index] = work(input[index]);
}
```

---

## Triple loop

Suppose you have a 3D volume:

```cpp
for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            output[x + nx * (y + ny * z)] = work(...);
        }
    }
}
```

CUDA has `x`, `y`, and `z` grid/thread coordinates, so you can map all three:

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;

if (x < nx && y < ny && z < nz) {
    int index = x + nx * (y + ny * z);
    output[index] = work(input[index]);
}
```

In practice, CUDA blocks are often 1D or 2D because a block cannot contain too many threads. For example:

```cpp
dim3 threads(16, 16, 1);
dim3 blocks(
    (nx + 15) / 16,
    (ny + 15) / 16,
    nz
);
```

---

## Four loops—and beyond

CUDA only provides three physical coordinate dimensions:

```text
x, y, z
```

That does **not** limit you to 3D problems. You combine, or *flatten*, dimensions into one index.

For this CPU code:

```cpp
for (int a = 0; a < A; ++a) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int d = 0; d < D; ++d) {
                output[...] = work(a, b, c, d);
            }
        }
    }
}
```

launch one thread per combination:

```cpp
long long total = 1LL * A * B * C * D;

long long id =
    1LL * (blockIdx.x * blockDim.x + threadIdx.x);

if (id < total) {
    int d = id % D;
    id /= D;

    int c = id % C;
    id /= C;

    int b = id % B;
    id /= B;

    int a = id;  // 0 <= a < A

    output[/* flatten (a, b, c, d) */] = work(a, b, c, d);
}
```

The flattening formula is:

```cpp
index = (((a * B + b) * C + c) * D + d);
```

The reverse process (`%` and `/`) recovers the coordinates.

---

## The important distinction: independent vs. dependent loops

Consider matrix multiplication:

```cpp
for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
        float sum = 0;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
```

`row` and `col` are independent: each output element `C[row, col]` can be computed separately. Map them to GPU threads.

The `k` loop is different. Every iteration contributes to the same `sum`, so the simple CUDA version keeps it sequential *inside* the thread:

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

if (row < M && col < N) {
    float sum = 0;

    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}
```

This is often the best first CUDA implementation. More advanced versions parallelize the `k` loop too, using shared memory and reductions.

## General recipe

For every loop index, ask:

> Can different iterations run independently without reading or changing one another's result?

- **Yes:** map that index, or a combination of indices, to threads.
- **No:** keep it as a loop inside a thread, or use a synchronization/reduction algorithm.
- **More than three independent indices:** flatten them into one linear thread ID, then recover coordinates if needed.
- **Every thread should normally write a distinct output location.** If many threads must contribute to one value, you need a reduction, atomic operation, or staged kernel.

In short:

```text
Independent output coordinates → GPU threads
Dependent accumulation / ordered work → loop inside a thread
Too many dimensions → flatten them
```
