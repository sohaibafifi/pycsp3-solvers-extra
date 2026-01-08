# Sport Scheduling Problem, Imported from pycsp3-models

from pycsp3 import *
from pycsp3_solvers_extra import solve

nTeams = data or 8
nWeeks, nPeriods, nMatches = nTeams - 1, nTeams // 2, (nTeams - 1) * nTeams // 2


def match_number(t1, t2):
    return nMatches - ((nTeams - t1) * (nTeams - t1 - 1)) // 2 + (t2 - t1 - 1)


T = {(t1, t2, match_number(t1, t2)) for t1, t2 in combinations(range(nTeams), 2)}

# m[w][p] is the number of the match at week w and period p
m = VarArray(size=[nWeeks, nPeriods], dom=range(nMatches))

# x[w][p] is the first team for the match at week w and period p
x = VarArray(size=[nWeeks, nPeriods], dom=range(nTeams))

# y[w][p] is the second team for the match at week w and period p
y = VarArray(size=[nWeeks, nPeriods], dom=range(nTeams))

satisfy(
    # all matches are different (no team can play twice against another team)
    AllDifferent(m),

    # linking variables through ternary table constraints
    [(x[w][p], y[w][p], m[w][p]) in T for w in range(nWeeks) for p in range(nPeriods)],

    # each week, all teams are different (each team plays each week)
    [AllDifferent(x[w] + y[w]) for w in range(nWeeks)],

    # each team plays at most two times in each period
    [Cardinality(x[:, p] + y[:, p], occurrences={t: range(1, 3) for t in range(nTeams)}) for p in range(nPeriods)]
)

if solve(solver='cpo') in [SAT, OPTIMUM]:
    print("Schedule of matches (teams are numbered from 0 to", nTeams - 1, "):")
    for w in range(nWeeks):
        print("Week", w + 1, ":", end=" ")
        for p in range(nPeriods):
            print("(", value(x[w][p]), "-", value(y[w][p]), ")", end=" ")
        print()
