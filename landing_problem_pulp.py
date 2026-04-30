from collections.abc import Iterable

import pandas as pd
import numpy as np
from numpy.typing import NDArray
import pulp as lp

IntArray = NDArray[np.int64]

small_dataset_file = "/mnt/c/Users/stefa/Downloads/Data assignment aircraft landing 2 Small.xlsx"
large_dataset_file = "/mnt/c/Users/stefa/Downloads/Data assignment aircraft landing 2 Large.xlsx"


def read_data_xlsx(file: str) -> tuple[IntArray, int, int]:
    # df containing the earliest, target and latest landing times
    df = pd.read_excel(file, "Times per aircraft")
    planes, seperation_time = pd.read_excel(file, "General information", header=None).iloc[:, 1]

    e = df["Earliest landing time"]
    t = df["Target landing time"]
    l = df["Latest landing time"]

    return np.array([e, t, l], dtype=np.int64), seperation_time, planes


def add_constraints(
    problem: lp.LpProblem,
    vars: Iterable[lp.LpAffineExpression],
    lower: IntArray | None = None,
    upper: IntArray | None = None,
) -> None:
    for i, expr in enumerate(vars):
        if lower is not None:
            expr = expr >= int(lower[i])
        if upper is not None:
            expr = expr <= int(upper[i])
        problem += expr


def add_variable_int_array(problem: lp.LpProblem, name: str, n: int, lower: int | None = None, upper: int | None = None):
    a = problem.add_variable_matrix(name, range(n), lowBound=lower, upBound=upper, cat=lp.LpInteger)
    return np.array(a)


def construct_problem(landing_times: IntArray, seperation_time: int, planes: int):
    indices = np.argsort(landing_times[1])
    e, t, l = landing_times[:, indices]
    n = planes
    s = seperation_time

    problem = lp.LpProblem("Landing times problem")
    x = add_variable_int_array(problem, "x", n)
    u = add_variable_int_array(problem, "u", n, lower=0)

    problem += lp.lpSum_vars(u)
    add_constraints(problem, x, lower=e, upper=l)
    add_constraints(problem, x + u, lower=t)
    add_constraints(problem, x - u, upper=t)

    separation = np.subtract.outer(x, x)[np.tril_indices(n, -1)]
    add_constraints(problem, separation, lower=np.full_like(separation, s))
    return problem, x[np.argsort(indices)]


def main():
    landing_times, seperation_time, planes = read_data_xlsx(large_dataset_file)
    problem, schedule = construct_problem(landing_times, seperation_time, planes)
    problem.solve(lp.HiGHS())

    get_value = lambda v: int(lp.value(v) or 0)
    solution = [get_value(x) for x in schedule]
    objective = get_value(problem.objective)

    for i, (x, t) in enumerate(zip(solution, landing_times[1])):
        print(f"Aircraft {i+1}: target {t}, scheduled at {x} (diff {abs(t - x)})")
    print("Objective (minimal):", objective)

if __name__ == '__main__':
    main()
