# pycsp3-solvers-extra Examples

This folder contains example problems that demonstrate the use of pycsp3-solvers-extra
with different constraint solvers. Each example compares the performance of:

- **ortools**: Google OR-Tools CP-SAT solver
- **cpo**: IBM DOcplex CP Optimizer (requires CPLEX installation)
- **ace**: ACE solver (native pycsp3 solver)
- **choco**: Choco solver (native pycsp3 solver)

## Examples

### 1. N-Queens (`queens.py`)
The classic N-Queens problem: place n queens on an n×n chessboard so that
no two queens attack each other.

```bash
python examples/queens.py -n 8
python examples/queens.py -n 12 --solvers ortools ace
```

### 2. Golomb Ruler (`golomb_ruler.py`)
Find a ruler with n marks such that all pairwise distances are unique,
minimizing the ruler length.

```bash
python examples/golomb_ruler.py -n 7
python examples/golomb_ruler.py -n 8 -t 120  # 2 minute timeout
```

### 3. Magic Sequence (`magic_sequence.py`)
Find a sequence where x[i] equals the count of value i in the sequence.

```bash
python examples/magic_sequence.py -n 10
python examples/magic_sequence.py -n 100
```

### 4. SEND + MORE = MONEY (`send_more_money.py`)
Classic cryptarithmetic puzzle where each letter represents a unique digit.

```bash
python examples/send_more_money.py
```

### 5. Graph Coloring (`graph_coloring.py`)
Color vertices of a graph with minimum colors such that adjacent vertices
have different colors.

```bash
python examples/graph_coloring.py -g petersen
python examples/graph_coloring.py -g wheel6 --solvers ortools choco
```

## Running All Examples

```bash
python examples/run_all.py
```

## Command Line Options

Most examples support these common options:

- `-v`, `--verbose`: Verbosity level (0=quiet, 1=normal, 2=detailed)
- `--solvers`: List of solvers to compare (default: ortools ace choco)
- `-t`, `--time-limit`: Time limit in seconds for optimization problems

## Adding CPO Solver

To include the CPO solver in comparisons, you need:

1. Install IBM CPLEX Optimization Studio
2. Add `cpo` to the solvers list:
   ```bash
   python examples/queens.py -n 8 --solvers ortools cpo ace choco
   ```

## Problem Sources

These examples are inspired by the [pycsp3-models](https://github.com/xcsp3team/pycsp3-models)
repository, which contains over 400 constraint programming models.
