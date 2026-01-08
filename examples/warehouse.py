# Warehouse Problem, Imported from pycsp3-models

from pycsp3 import *
from pycsp3_solvers_extra import solve, supported_solvers

fixed_cost = 30                 # cost of opening a warehouse
capacities = [1, 4, 2, 1, 3]    # capacities of the warehouses
costs = [                       # costs of supplying stores
   [100, 24, 11, 25, 30],
    [28, 27, 82, 83, 74],
    [74, 97, 71, 96, 70],
    [ 2, 55, 73, 69, 61],
    [46, 96, 59, 83,  4],
    [42, 22, 29, 67, 59],
    [ 1,  5, 73, 59, 56],
    [10, 73, 13, 43, 96],
    [93, 35, 63, 85, 46],
    [47, 65, 55, 71, 95]
]
nWarehouses, nStores = len(capacities), len(costs)

for solver in supported_solvers():
    print("\n--- Using solver:", solver, "---\n")
    costs = cp_array(costs)


    # w[i] is the warehouse supplying the ith store
    w = VarArray(size=nStores, dom=range(nWarehouses))

    satisfy(
        # capacities of warehouses must not be exceeded
        Count(w, value=j) <= capacities[j] for j in range(nWarehouses)
    )

    minimize(
        # minimizing the overall cost
        Sum(costs[i][w[i]] for i in range(nStores)) + NValues(w) * fixed_cost
    )

    if solve(solver=solver) in [SAT, OPTIMUM]:
        print(values(w))
        for i in range(nStores):
            print(f"Cost of supplying the store {i} is {costs[i][value(w[i])]}")
        print("Total supplying cost: ", bound())
    clear()