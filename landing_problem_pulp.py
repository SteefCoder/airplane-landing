import pandas as pd
import numpy as np
import pulp as lp

small_dataset_file = "..."
large_dataset_file = "..."


def read_data_xlsx(file: str) -> tuple[np.ndarray, int, int]:
    # df containing the earliest, target and latest landing times
    df = pd.read_excel(file, "Times per aircraft")
    planes, separation_time = pd.read_excel(file, "General information", header=None).iloc[:, 1]

    e = df["Earliest landing time"]
    t = df["Target landing time"]
    l = df["Latest landing time"]

    return np.array([e, t, l], dtype=np.float64), separation_time, planes


def construct_problem(landing_times: np.ndarray, separation_time: int, planes: int):
    indices = np.argsort(landing_times[1])
    e, t, l = landing_times[:, indices]
    n = planes
    s = separation_time

    problem = lp.LpProblem("Landing_times_problem")
    # x is the scheduled times
    x = problem.add_variable_matrix("x", range(n), cat=lp.LpInteger)
    # we will constrict these so that u = |x - t|
    u = problem.add_variable_matrix("u", range(n), cat=lp.LpInteger, lowBound=0)

    # objective function f = sum |x - t|
    problem += lp.lpSum_vars(u)

    for ei, li, ti, xi, ui in zip(e, l, t, x, u):
        problem += ei <= xi
        problem += xi <= li
        problem += xi + ui >= ti
        problem += xi - ui <= ti
    
    for i in range(n):
        for j in range(i + 1, n):
            problem += x[j] - x[i] >= s

    sorted_x = [x[i] for i in np.argsort(indices)]
    return problem, sorted_x


def print_solution(problem: lp.LpProblem, target: list[int], solution: list[lp.LpVariable]) -> None:
    get_value = lambda v: int(lp.value(v) or 0)
    solution_values = [get_value(x) for x in solution]
    objective = get_value(problem.objective)

    for i, (x, t) in enumerate(zip(solution_values, target)):
        print(f"Aircraft {i+1}: target {t}, scheduled at {x} (diff {abs(t - x)})")
    print("Objective (minimal):", objective)


def main():
    landing_times, separation_time, planes = read_data_xlsx(large_dataset_file)
    problem, schedule = construct_problem(landing_times, separation_time, planes)
    problem.solve(lp.HiGHS(msg=False))
    print_solution(problem, landing_times[1].astype(np.int64), schedule)


if __name__ == '__main__':
    main()
