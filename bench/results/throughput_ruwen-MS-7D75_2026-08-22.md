# LangevInMedium throughput — ruwen-MS-7D75, 2026-08-22 00:20

Julia 1.12.6, threads 1, CPU AMD Ryzen 9 9900X 12-Core Processor, GPU NVIDIA GeForce RTX 5070

Engine: 100 and 400 steps at dt = 2e-3 on a flowing Gaussian fireball, two snapshots, particles injected (no sampler). Marginal cost = (t₄₀₀ − t₁₀₀)/(300·N); fixed = per-call overhead (allocation, spline build, upload/download, cleanup).

| backend | N | momentum rows | relativistic | mode | wall [s] (steps) | wall [s] (steps) | marginal ns/particle-step | fixed per call [s] |
|---|---|---|---|---|---|---|---|---|
| CPUBackend | 10000 | 2 | true | langevin | 0.153 (100) | 0.612 (400) | 153.1 | 0.00 |
| CPUBackend | 10000 | 2 | true | rta | 0.132 (100) | 0.518 (400) | 128.7 | 0.00 |
| CPUBackend | 10000 | 2 | false | langevin | 0.145 (100) | 0.582 (400) | 145.8 | 0.00 |
| CPUBackend | 10000 | 2 | false | rta | 0.127 (100) | 0.508 (400) | 127.0 | 0.00 |
| CPUBackend | 10000 | 3 | true | langevin | 0.158 (100) | 0.633 (400) | 158.4 | 0.00 |
| CPUBackend | 10000 | 3 | true | rta | 0.132 (100) | 0.528 (400) | 131.7 | 0.00 |
| CPUBackend | 10000 | 3 | false | langevin | 0.148 (100) | 0.598 (400) | 149.8 | 0.00 |
| CPUBackend | 10000 | 3 | false | rta | 0.131 (100) | 0.516 (400) | 128.3 | 0.00 |
| CPUBackend | 100000 | 2 | true | langevin | 1.971 (100) | 7.520 (400) | 185.0 | 0.12 |
| CPUBackend | 100000 | 2 | true | rta | 1.564 (100) | 6.541 (400) | 165.9 | 0.00 |
| CPUBackend | 100000 | 2 | false | langevin | 1.890 (100) | 7.163 (400) | 175.7 | 0.13 |
| CPUBackend | 100000 | 2 | false | rta | 1.549 (100) | 6.449 (400) | 163.3 | 0.00 |
| CPUBackend | 100000 | 3 | true | langevin | 2.033 (100) | 7.890 (400) | 195.3 | 0.08 |
| CPUBackend | 100000 | 3 | true | rta | 1.839 (100) | 7.004 (400) | 172.2 | 0.12 |
| CPUBackend | 100000 | 3 | false | langevin | 2.006 (100) | 7.793 (400) | 192.9 | 0.08 |
| CPUBackend | 100000 | 3 | false | rta | 1.813 (100) | 6.984 (400) | 172.4 | 0.09 |
| CPUBackend | 1000000 | 2 | true | langevin | 5.012 (25) | 21.024 (100) | 213.5 | 0.00 |
| CPUBackend | 1000000 | 2 | false | langevin | 4.790 (25) | 19.101 (100) | 190.8 | 0.02 |
| CPUBackend | 1000000 | 3 | true | langevin | 5.074 (25) | 20.872 (100) | 210.6 | 0.00 |
| CPUBackend | 1000000 | 3 | false | langevin | 5.014 (25) | 20.477 (100) | 206.2 | 0.00 |
| GPUBackend | 10000 | 2 | true | langevin | 0.992 (100) | 1.008 (400) | 5.6 | 0.99 |
| GPUBackend | 10000 | 2 | true | rta | 0.997 (100) | 1.031 (400) | 11.3 | 0.99 |
| GPUBackend | 10000 | 2 | false | langevin | 0.988 (100) | 0.993 (400) | 1.7 | 0.99 |
| GPUBackend | 10000 | 2 | false | rta | 0.999 (100) | 1.019 (400) | 6.4 | 0.99 |
| GPUBackend | 10000 | 3 | true | langevin | 0.986 (100) | 0.997 (400) | 3.7 | 0.98 |
| GPUBackend | 10000 | 3 | true | rta | 1.000 (100) | 1.013 (400) | 4.4 | 1.00 |
| GPUBackend | 10000 | 3 | false | langevin | 0.986 (100) | 0.996 (400) | 3.1 | 0.98 |
| GPUBackend | 10000 | 3 | false | rta | 1.000 (100) | 1.016 (400) | 5.3 | 1.00 |
| GPUBackend | 100000 | 2 | true | langevin | 1.022 (100) | 1.108 (400) | 2.9 | 0.99 |
| GPUBackend | 100000 | 2 | true | rta | 1.024 (100) | 1.114 (400) | 3.0 | 0.99 |
| GPUBackend | 100000 | 2 | false | langevin | 1.044 (100) | 1.104 (400) | 2.0 | 1.02 |
| GPUBackend | 100000 | 2 | false | rta | 1.046 (100) | 1.483 (400) | 14.6 | 0.90 |
| GPUBackend | 100000 | 3 | true | langevin | 1.042 (100) | 1.080 (400) | 1.3 | 1.03 |
| GPUBackend | 100000 | 3 | true | rta | 1.033 (100) | 1.134 (400) | 3.4 | 1.00 |
| GPUBackend | 100000 | 3 | false | langevin | 1.017 (100) | 1.122 (400) | 3.5 | 0.98 |
| GPUBackend | 100000 | 3 | false | rta | 1.026 (100) | 1.125 (400) | 3.3 | 0.99 |
| GPUBackend | 1000000 | 2 | true | langevin | 1.258 (100) | 2.040 (400) | 2.6 | 1.00 |
| GPUBackend | 1000000 | 2 | true | rta | 1.211 (100) | 1.984 (400) | 2.6 | 0.95 |
| GPUBackend | 1000000 | 2 | false | langevin | 1.272 (100) | 2.048 (400) | 2.6 | 1.01 |
| GPUBackend | 1000000 | 2 | false | rta | 1.325 (100) | 1.951 (400) | 2.1 | 1.12 |
| GPUBackend | 1000000 | 3 | true | langevin | 1.506 (100) | 2.491 (400) | 3.3 | 1.18 |
| GPUBackend | 1000000 | 3 | true | rta | 1.467 (100) | 2.338 (400) | 2.9 | 1.18 |
| GPUBackend | 1000000 | 3 | false | langevin | 1.471 (100) | 2.316 (400) | 2.8 | 1.19 |
| GPUBackend | 1000000 | 3 | false | rta | 1.372 (100) | 2.161 (400) | 2.6 | 1.11 |

## Host-side phases

| phase | N | wall [s] | note |
|---|---|---|---|
| FONLL sampler, cartesian=true | 100000 | 0.079 | 0.79 µs/particle; Gaussian σ_r≈3.9 fm in r≤20 fm |
| FONLL sampler, cartesian=false | 100000 | 0.037 | 0.37 µs/particle; Gaussian σ_r≈3.9 fm in r≤20 fm |
| randn! (2 × N) per step | 100000 | 0.0002 | CPU path draws this every step |
| randn! (3 × N) per step | 100000 | 0.0003 | CPU path draws this every step |
| snapshot save (2 × N) | 100000 | 0.0000 | per saved snapshot |
| CUDA.randn 2 × N | 100000 | 0.00008 | device draw per step |
| device→host copy (2 × 10N) | 100000 | 0.0007 | whole history download |
