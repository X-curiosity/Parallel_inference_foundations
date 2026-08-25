# Matrix multiplication benchmark cases
Each `.txt` file uses this simple freeform argument shape:
```text
timeout <seconds>
<n> <m> <k>
```
This represents multiplying `A(n x m)` by `B(m x k)` to produce `C(n x k)`.
The estimated floating-point operation count is `n * k * (2*m - 1)`.

## Solid CPU throughput benchmarks

- `cpu_solid_1.txt`: `1024x1024 * 1024x1024` → 2,146,435,072 FLOPs, ideal time at 4.9 GFLOP/s ≈ 0.438s.
- `cpu_solid_2.txt`: `1344x1344 * 1344x1344` → 4,853,624,832 FLOPs, ideal time at 4.9 GFLOP/s ≈ 0.991s.
- `cpu_solid_3.txt`: `2048x1024 * 1024x1024` → 4,292,870,144 FLOPs, ideal time at 4.9 GFLOP/s ≈ 0.876s.
- `cpu_solid_4.txt`: `4096x512 * 512x1024` → 4,290,772,992 FLOPs, ideal time at 4.9 GFLOP/s ≈ 0.876s.

## Notes

- The 100 numbered files are a sweep of smaller and medium matrix shapes.
- The 4 `cpu_solid_*` files are intended to stress sustained CPU throughput more seriously.
- Your earlier example `450x1000 * 1000x10` is only `450*10*(2*1000-1) = 8,995,500` FLOPs, so it should be far below one second on a 4.9 GFLOP/s machine if overhead is ignored.
