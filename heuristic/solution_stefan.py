import pandas as pd
import numpy as np

small_dataset_file = "airplane-landing/Data assignment aircraft landing 2 Small.xlsx"
large_dataset_file = "airplane-landing/Data assignment aircraft landing 2 Large.xlsx"


def read_data_xlsx(file: str) -> tuple[np.ndarray, int, int]:
    # df containing the earliest, target and latest landing times
    df = pd.read_excel(file, "Times per aircraft")
    planes, separation_time = pd.read_excel(file, "General information", header=None).iloc[:, 1]

    e = df["Earliest landing time"]
    t = df["Target landing time"]
    l = df["Latest landing time"]

    return np.array([e, t, l], dtype=np.int64), separation_time, planes


def push_down(solution: list[int], separation_time: int, index: int, by: int) -> list[int]:
    solution[index] -= by
    for i in range(index - 1, -1, -1):
        solution[i] = min(solution[i], solution[i + 1] - separation_time)
    return solution


def calculate_error(solution: list[int], target: list[int]) -> int:
    return sum(abs(s - t) for s, t in zip(solution, target))


def push_down_error_change(solution: list[int], target: list[int], separation_time: int, index: int, by: int) -> int:
    error_change = 0
    current = solution[index] - by
    for i in range(index, -1, -1):
        if solution[i] < current: break

        error_change += abs(target[i] - current) - abs(target[i] - solution[i])
        current -= separation_time
    return error_change


def solve_heuristic(landing_times: np.ndarray, separation_time: int, planes: int) -> tuple[list[int] | None, int]:
    indices = np.argsort(landing_times[1])
    e, t, l = landing_times[:, indices]

    solution = [t[0]]
    for i in range(1, planes):
        solution.append(max(t[i], solution[-1] + separation_time))

        best_push = 0
        while True:
            error_change = push_down_error_change(solution, t, separation_time, i, best_push + 1)
            if error_change > 0: break
            best_push += 1

        if best_push > 0:
            push_down(solution, separation_time, i, best_push)
    
    if not (np.all(e <= solution) and np.all(solution <= l)):
        return None, -1
    
    sorted_solution = []
    for i in np.argsort(indices):
        sorted_solution.append(solution[i])

    return sorted_solution, calculate_error(solution, t)


def main():
    landing_times, separation_time, planes = read_data_xlsx(large_dataset_file)
    solution, error = solve_heuristic(landing_times, separation_time, planes)
    print(solution)
    print(error)


if __name__ == '__main__':
    main()