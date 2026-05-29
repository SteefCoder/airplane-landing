from enum import Enum, auto

import pulp as lp


class PlaneType(Enum):
    SMALL = auto()
    LARGE = auto()


def construct_lp_problem(
    types: list[PlaneType],
    arrivals: list[int],  # the arrival times
    turnarounds: dict[PlaneType, int],  # turnaround times for each plane type
    airtime_weights: dict[PlaneType, int],  # cost of airtime per plane type
    airtime_multiplier: int,  # additional weight to reduce airtimes
    separation_time: int,  # minimum separation time between two planes
    gate_capacity: int,  # maximal capacity for planes waiting on the ground
):
    
    n = len(types)
    assert len(arrivals) == n

    M = 10_000

    problem = lp.LpProblem("Priority-landing", lp.LpMinimize)

    landings = lp.LpVariable.dict("l", (range(n),), 0, cat=lp.LpInteger)
    takeoffs = lp.LpVariable.dict("t", (range(n),), 0, cat=lp.LpInteger)

    x = lp.LpVariable.dict("x", (range(n), range(n)), cat=lp.LpBinary)
    y = lp.LpVariable.dict("y", (range(n), range(n)), cat=lp.LpBinary)
    z = lp.LpVariable.dict("z", (range(n), range(n)), cat=lp.LpBinary)

    airtime_sum = airtime_multiplier * lp.lpSum(
        airtime_weights[types[i]] * (landings[i] - arrivals[i])
        for i in range(n)
    )
    ground_sum = lp.lpSum(
        takeoffs[i] - landings[i] - turnarounds[types[i]]
        for i in range(n)
    )

    problem += airtime_sum + ground_sum

    for i in range(n):
        problem += arrivals[i] <= landings[i]
        problem += landings[i] + turnarounds[types[i]] <= takeoffs[i]

        problem += lp.lpSum(
            z[i, j] - y[i, j]
            for j in range(n) if i != j
        ) <= gate_capacity - 1

        for j in range(n):
            if i == j: continue

            problem += (1 - x[i, j]) * M + landings[i] - landings[j] >= separation_time
            problem += -x[i, j] * M + landings[i] - landings[j] <= -separation_time

            problem += (1 - y[i, j]) * M + takeoffs[i] - takeoffs[j] >= separation_time
            problem += -y[i, j] * M + takeoffs[i] - takeoffs[j] <= -separation_time

            problem += (1 - z[i, j]) * M + takeoffs[i] - landings[j] >= separation_time
            problem += -z[i, j] * M + takeoffs[i] - landings[j] <= -separation_time
    print(problem)
    return problem, landings, takeoffs


def solve_lp_problem(problem: lp.LpProblem):
    solver = lp.GUROBI()
    status = problem.solve(solver)
    return lp.LpStatus[status]


def main():
    turnarounds = {
        PlaneType.SMALL: 7,  # in minutes
        PlaneType.LARGE: 12,
    }
    airtime_weights = {
        PlaneType.SMALL: 10,  # for now, the number of passengers / 10
        PlaneType.LARGE: 25,
    }
    arrivals = [10, 12, 15, 20]
    problem, landings, takeoffs = construct_lp_problem(
        [PlaneType.SMALL, PlaneType.LARGE, PlaneType.SMALL, PlaneType.SMALL],
        arrivals,
        turnarounds,
        airtime_weights,
        airtime_multiplier=3,
        separation_time=4,
        gate_capacity=1
    )

    status = solve_lp_problem(problem)
    print("Status   : ", status)
    print("Objective: ", lp.value(problem.objective))
    print("Arrivals : ", arrivals)
    print("Landings : ", [lp.value(x) for x in landings.values()])
    print("Takeoffs : ", [lp.value(x) for x in takeoffs.values()])


if __name__ == '__main__':
    main()