import pandas as pd
import numpy as np
import pulp as lp

# Data sets
small_dataset_file = r"C:\Users\flori\OneDrive\Desktop\Downloads\Data assignment aircraft landing 2 Small (2).xlsx"
large_dataset_file = r"C:\Users\flori\OneDrive\Desktop\Downloads\Data assignment aircraft landing 2 Large (1).xlsx"

def read_data_xlsx(file):
    """
    Read aircraft landing data from Excel file.
    """
    df = pd.read_excel(file, sheet_name="Times per aircraft")
    general = pd.read_excel(file, sheet_name="General information", header=None)
    
    planes = general.iloc[0, 1]
    separation_time = general.iloc[1, 1]
    
    e = df["Earliest landing time"].to_numpy(dtype=int)
    t = df["Target landing time"].to_numpy(dtype=int)
    l = df["Latest landing time"].to_numpy(dtype=int)
    
    return e, t, l, separation_time, planes

def add_variable_int_array(name: str, n: int):
    
    a = []
    
    for i in range(n):
        variable = lp.LpVariable(f"{name}_{i}", cat=lp.LpInteger)
        a.append(variable)
    
    return a
    
def construct_problem(e, t, l, separation_time, planes):
    """
    Build the optimization problem.
    """
    indices = np.argsort(t)
    e, t, l = e[indices], t[indices], l[indices]
    n = planes
    s = separation_time
    
    problem = lp.LpProblem("Aircraft_Landing_Problem", lp.LpMinimize)
    
    x = add_variable_int_array("x", n)    
    u = add_variable_int_array("u", n)
    
    problem += lp.lpSum(u)
    
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


def main():
    """
    The main function to solve the ILP
    """
    e, t, l, separation_time, planes = (read_data_xlsx(small_dataset_file))
    problem, schedule = construct_problem(e, t, l, separation_time, planes)
    problem.solve(lp.PULP_CBC_CMD(msg=False))

    print("Status:", lp.LpStatus[problem.status])

    solution = []

    for variable in schedule:
        value = int(lp.value(variable))
        solution.append(value)

    objective = int(lp.value(problem.objective))

    for i in range(planes):
        scheduled_time = solution[i]
        target_time = int(t[i])
        difference = abs(target_time - scheduled_time)
        print(f"Aircraft {i + 1}: " f"target = {target_time}, " f"scheduled = {scheduled_time}, " f"difference = {difference}")

    print("Total deviation:", objective)


if __name__ == "__main__":
    main()