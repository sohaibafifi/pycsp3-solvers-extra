# pycsp3-solvers-extra

Extension library that adds some extra solver backends to [pycsp3](https://github.com/xcsp3team/pycsp3).

## Supported Solvers

- **OR-Tools CP-SAT** - Google's constraint programming solver
- **IBM DOcplex CP Optimizer** - IBM's CP solver (requires CPLEX Studio)

## Installation

```bash
pip install pycsp3-solvers-extra
```

### Dependencies

- `pycsp3` - base constraint modeling library
- `ortools` - for OR-Tools backend
- `docplex` - for CPO backend (requires IBM CPLEX Studio installed separately)

## Usage

```python
from pycsp3 import *
from pycsp3_solvers_extra import solve

# Define your model
x = VarArray(size=3, dom=range(1, 10))
satisfy(AllDifferent(x))
minimize(Sum(x))

# Solve with OR-Tools
status = solve(solver="ortools")

# Or solve with CPO
status = solve(solver="cpo")

# Of course, you still can use the native supported pycsp3 solvers as well ('ace', 'choco')
status = solve(solver="ace")

# Access solution values
print([v.value for v in x])
```

### Loading XCSP3 instances

```python
from pycsp3 import clear, solution
from pycsp3_solvers_extra import load, solve

clear()
load("path/to/instance.xml.lzma")  # or .xml
status = solve(solver="ortools", time_limit=10)
print(status)
print(solution())
```

## API

```python
solve(
    solver="ortools",  # "ortools" or "cpo"
    time_limit=None,   # seconds
    sols=None,         # number of solutions to find
    verbose=0,         # verbosity level (0-2)
    options=""         # solver-specific options
)
```

Returns `TypeStatus.SAT`, `TypeStatus.OPTIMUM`, `TypeStatus.UNSAT`, or `TypeStatus.UNKNOWN`.

## Examples

See the `examples/` directory:

```bash
python examples/send_more_money.py --solvers ortools cpo
python examples/solve_xcsp.py path/to/instance.xml.lzma --solver ortools --time-limit 10
```

## Running Tests

```bash
pytest tests/ -v
```

## License
This project is licensed under the MIT License.
