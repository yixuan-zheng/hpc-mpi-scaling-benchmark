CXX = mpicxx
CXXFLAGS = -O2 -std=c++17 -Wall
TARGET = matmul_mpi
SRC = matmul_mpi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run:
	mpirun -np 2 ./$(TARGET) 512

smoke:
	python3 run_benchmarks.py --sizes 512 --procs 1 2 --trials 2 --exe ./$(TARGET) --launcher mpirun --tag smoke

plot-smoke:
	python3 plot_benchmarks.py --input $$(ls -t results/benchmark_smoke_*.json | head -n 1) --output-dir results