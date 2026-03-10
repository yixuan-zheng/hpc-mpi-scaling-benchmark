# MPI-Based HPC Workload Scaling Evaluator

## Overview

This project implements a **distributed matrix multiplication benchmark using MPI** and evaluates its scaling behavior on the **Georgia Tech PACE HPC cluster**. The goal is to analyze how computation and communication costs interact as the number of processes increases.

The project also provides a lightweight **benchmarking pipeline** that automates:

* repeated MPI experiments
* performance data collection
* speedup and efficiency computation
* visualization of scaling behavior

This serves as a simple prototype of an **HPC workload evaluation tool**, useful for understanding whether a workload is compute-bound or communication-bound.

# Project Goals

The objectives of this project are:

1. Implement a distributed matrix multiplication algorithm using MPI.
2. Evaluate **strong scaling performance** across multiple nodes.
3. Measure the relationship between:

$$
[\text{runtime},\; \text{speedup},\; \text{efficiency},\; \text{communication overhead}]
$$

4. Demonstrate key HPC principles:

* strong scaling
* communication overhead
* workload size impact on parallel performance

# Method

## Parallel Decomposition

We use a **row-wise distributed matrix multiplication** approach.

Given:

$$
C = A \times B
$$

$$
\text{where } (A, B, C \in \mathbb{R}^{N \times N})
$$

### Data distribution

1. Matrix (A) is **row-partitioned** across MPI ranks.
2. Matrix (B) is **broadcast** to all ranks.
3. Each rank computes its local block of (C).
4. Partial results are gathered to reconstruct (C).

## Algorithm

High-level workflow:

```
Rank 0:
    initialize matrices A and B

Scatter rows of A across ranks
Broadcast matrix B to all ranks

Each rank:
    compute local rows of C

Gather rows of C back to root
```

This design is intentionally simple and highlights the impact of **communication costs** in distributed environments.

# Performance Metrics

The benchmark records several HPC metrics.

### Runtime

Parallel runtime using \(P\) processes: $T_P$

Sequential runtime: $T_1$

### Speedup

$$
S(P) = \frac{T_1}{T_P}
$$

Ideal speedup:

$$
S(P) = P
$$

### Parallel Efficiency

$$
E(P) = \frac{S(P)}{P}
$$

Efficiency indicates how effectively additional processors are utilized.

### Communication Ratio

We measure the fraction of time spent on MPI communication:

$$
R_{comm} = \frac{T_{comm}}{T_{total}}
$$

Higher values indicate the workload is becoming **communication-bound**.

# Experiment Setup

Experiments were performed on the **Georgia Tech PACE HPC cluster**.

### Matrix sizes tested

$$
N = 1024,\ 2048,\ 4096
$$

### Process counts

$$
P = 1,\ 2,\ 4
$$

### Trials

Each configuration was executed **3 times**, and the minimum runtime was used to compute speedup.

# Experimental Results

## Speedup vs Processes

*(Insert plot here)*

```
![Plot 1] (results/benchmark_pace_main_speedup.png)
```

---

## Efficiency vs Processes

*(Insert plot here)*

```
![Plot 2] (results/benchmark_pace_main_efficiency.png)
```

---

## Communication Ratio vs Processes

*(Insert plot here)*

```
![Plot 3] (results/benchmark_pace_main_comm_ratio.png)
```

# Analysis

## Strong Scaling Behavior

The algorithm demonstrates near-linear scaling for all tested matrix sizes.

Example (N = 4096):

| Processes | Runtime (s) | Speedup | Efficiency |
| --------- | ----------- | ------- | ---------- |
| 1         | 717         | 1.00    | 1.00       |
| 2         | 350         | 2.05    | 1.02       |
| 4         | 185         | 3.87    | 0.97       |

Speedup remains close to the ideal line:

$$
S(P) \approx P
$$

indicating strong parallel performance.

## Superlinear Speedup

Some configurations show:

$$
E(P) > 1
$$

This phenomenon is known as **superlinear speedup**, typically caused by improved **cache locality** and **memory bandwidth utilization** when the workload is distributed across multiple processors.

## Communication Overhead

Communication costs increase with the number of processes.

For example (N = 4096):

| Processes | Communication Ratio |
| --------- | ------------------- |
| 1         | ~0                  |
| 2         | ~0.03               |
| 4         | ~0.36               |

This occurs because the algorithm requires broadcasting matrix (B) and gathering results.

## Problem Size Effect

Compute complexity: $O(N^3)$

Communication complexity: $O(N^2)$

As (N) increases, computation grows faster than communication.

This explains why **larger matrices scale more efficiently**.

# Conclusion

This project demonstrates how a simple MPI workload can be used to analyze HPC scaling behavior.

Key findings:

* Matrix multiplication exhibits strong parallel scalability.
* Communication overhead increases with processor count.
* Larger workloads amortize communication costs.
* Near-linear speedup is achievable on distributed systems.

The benchmarking pipeline provides a reproducible framework for evaluating **compute vs communication tradeoffs** in parallel workloads.

# How to Reproduce

## 1. Clone repository

```bash
git clone <repo-url>
cd hpc_benchmark
```

## 2. Compile

Requires an MPI implementation such as OpenMPI or MPICH.

```bash
make
```

This builds:

```
matmul_mpi
```

## 3. Run a simple MPI test

Example:

```bash
mpirun -np 4 ./matmul_mpi 1024
```

## 4. Run benchmark experiments

Example benchmark sweep:

```bash
python3 run_benchmarks.py \
  --sizes 1024 2048 4096 \
  --procs 1 2 4 \
  --trials 3 \
  --exe ./matmul_mpi \
  --launcher srun \
  --tag pace_main
```

Results are saved to:

```
results/
```

## 5. Generate plots

```bash
python3 plot_benchmarks.py \
  --input results/benchmark_pace_main_*.json \
  --output-dir results
```

# Installation Guide

### Requirements

* C++17 compiler
* MPI implementation (OpenMPI / MPICH)
* Python 3.8+
* Python packages:

```
matplotlib
numpy
```

Install Python dependencies:

```bash
pip install matplotlib numpy
```

# Repository Structure

```
.
├── matmul_mpi.cpp
├── matmul_mpi
├── run_benchmarks.py
├── plot_benchmarks.py
├── Makefile
└── results/
```

# Acknowledgements

Experiments were conducted on the **Georgia Tech PACE HPC cluster**.
