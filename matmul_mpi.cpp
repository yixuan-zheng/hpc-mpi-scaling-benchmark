#include <mpi.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Utility: index helper for flattened row-major matrices
inline int idx(int row, int col, int ncols) {
    return row * ncols + col;
}

// Initialize a matrix with deterministic values
// This makes debugging easier than random initialization.
void init_matrix(std::vector<double>& M, int rows, int cols, int seed_offset = 0) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            M[idx(i, j, cols)] = static_cast<double>((i + j + seed_offset) % 10 + 1);
        }
    }
}

void zero_matrix(std::vector<double>& M) {
    std::fill(M.begin(), M.end(), 0.0);
}

// ------------------------------------------------------------
// Serial matrix multiply for correctness checking on root
// C = A x B
// A: rowsA x N
// B: N x N
// C: rowsA x N
// ------------------------------------------------------------
void serial_matmul(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C,
                   int rowsA,
                   int N) {
    zero_matrix(C);

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[idx(i, k, N)] * B[idx(k, j, N)];
            }
            C[idx(i, j, N)] = sum;
        }
    }
}

// ------------------------------------------------------------
// Local matrix multiply on each MPI rank
// A_local: local_rows x N
// B:       N x N
// C_local: local_rows x N
// ------------------------------------------------------------
void local_matmul(const std::vector<double>& A_local,
                  const std::vector<double>& B,
                  std::vector<double>& C_local,
                  int local_rows,
                  int N) {
    zero_matrix(C_local);

    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A_local[idx(i, k, N)] * B[idx(k, j, N)];
            }
            C_local[idx(i, j, N)] = sum;
        }
    }
}

// Compare two matrices with tolerance
bool matrices_close(const std::vector<double>& X,
                    const std::vector<double>& Y,
                    double tol = 1e-9) {
    if (X.size() != Y.size()) return false;

    for (size_t i = 0; i < X.size(); ++i) {
        double diff = std::abs(X[i] - Y[i]);
        if (diff > tol) return false;
    }
    return true;
}

// Parse matrix size N from command line
// Usage: mpirun -np <p> ./matmul_mpi <N>
int parse_N(int argc, char* argv[], int rank) {
    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Usage: ./matmul_mpi <N>\n";
        }
        return -1;
    }

    int N = std::atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0) {
            std::cerr << "Error: N must be a positive integer.\n";
        }
        return -1;
    }

    return N;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = parse_N(argc, argv, rank);
    if (N < 0) {
        MPI_Finalize();
        return 1;
    }

    if (N % size != 0) {
        if (rank == 0) {
            std::cerr << "Error: N must be divisible by number of processes.\n";
            std::cerr << "Received N = " << N << ", P = " << size << "\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int local_rows = N / size;

    // Root owns full A, full B, full C
    std::vector<double> A;
    std::vector<double> C;

    // All ranks need full B, local A, local C
    std::vector<double> B(N * N);
    std::vector<double> A_local(local_rows * N);
    std::vector<double> C_local(local_rows * N);

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);

        init_matrix(A, N, N, 0);
        init_matrix(B, N, N, 100);
    }

    // Non-root ranks still need storage for B before broadcast
    if (rank != 0) {
        zero_matrix(B);
    }

    // Timing variables
    double comm_time = 0.0;
    double compute_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();

    // Scatter rows of A
    double t0 = MPI_Wtime();
    MPI_Scatter(
        rank == 0 ? A.data() : nullptr,
        local_rows * N,
        MPI_DOUBLE,
        A_local.data(),
        local_rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );
    comm_time += MPI_Wtime() - t0;

    // Broadcast full B
    t0 = MPI_Wtime();
    MPI_Bcast(
        B.data(),
        N * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );
    comm_time += MPI_Wtime() - t0;

    // Local compute
    t0 = MPI_Wtime();
    local_matmul(A_local, B, C_local, local_rows, N);
    compute_time += MPI_Wtime() - t0;

    // Gather local C back to root
    t0 = MPI_Wtime();
    MPI_Gather(
        C_local.data(),
        local_rows * N,
        MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr,
        local_rows * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );
    comm_time += MPI_Wtime() - t0;

    double total_time = MPI_Wtime() - total_start;

    // Reduce timings to root using MAX across ranks
    double total_time_max = 0.0;
    double comm_time_max = 0.0;
    double compute_time_max = 0.0;

    MPI_Reduce(&total_time, &total_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &comm_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &compute_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        bool correct = true;

        if (N <= 256) {
            std::vector<double> C_ref(N * N);
            serial_matmul(A, B, C_ref, N, N);
            correct = matrices_close(C, C_ref);
        }

        double comm_ratio = (total_time_max > 0.0) ? (comm_time_max / total_time_max) : 0.0;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "RESULT "
                  << "N=" << N
                  << " P=" << size
                  << " TOTAL=" << total_time_max
                  << " COMM=" << comm_time_max
                  << " COMPUTE=" << compute_time_max
                  << " COMM_RATIO=" << comm_ratio
                  << " CORRECT=" << (correct ? 1 : 0)
                  << "\n";
    }

    MPI_Finalize();
    return 0;
}