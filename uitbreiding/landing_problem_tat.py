import random

import pulp as lp
import airport_timeline as at


def add_variable_array(
    problem: lp.LpProblem,
    name: str,
    size: int | tuple[int, int],
    cat: str = lp.LpInteger
):
    if isinstance(size, int):
        return problem.add_variable_matrix(name, range(size), lowBound=0, cat=cat)
    
    return [add_variable_array(problem, name + str(i), size[1], cat) for i in range(size[0])]


def construct_lp_problem(
    a: list[int],
    T: int,
    s: int,
    G: int,
    wA: int,
    wG: int,
    n: int,
):
    problem = lp.LpProblem("Landing-times-TAT")

    M = 10_000

    l = add_variable_array(problem, "l", n)
    t = add_variable_array(problem, "t", n)

    # x_ij = 1 if li > lj. 
    x = add_variable_array(problem, "x", (n, n), cat=lp.LpBinary)
    # y_ij = 1 if ti > tj
    y = add_variable_array(problem, "y", (n, n), cat=lp.LpBinary)
    # z_ij = 1 if ti > lj
    z = add_variable_array(problem, "z", (n, n), cat=lp.LpBinary)

    problem += lp.lpSum(li + ti - 2*ai for ai, li, ti in zip(a, l, t)) - n * T
    
    for ai, li, ti in zip(a, l, t):
        problem += ai <= li
        problem += li <= ai + wA
        problem += li + T <= ti
        problem += ti <= li + T + wG

    for i in range(n):
        for j in range(n):
            if i == j: continue

            problem += (1 - x[i][j]) * M + l[i] - l[j] >= s
            problem += -x[i][j] * M + l[i] - l[j] <= -s

            problem += (1 - y[i][j]) * M + t[i] - t[j] >= s
            problem += -y[i][j] * M + t[i] - t[j] <= -s
            
            problem += (1 - z[i][j]) * M + t[i] - l[j] >= s
            problem += -z[i][j] * M + t[i] - l[j] <= -s
    
    for i in range(n):
        problem += lp.lpSum(z[i][j] - y[i][j] for j in range(n) if i != j) + 1 <= G

    return problem, (l, t)


def main():
    a = []
    for _ in range(18):
        a.append(random.randint(0, 3 * 60))

    s = 3
    wA, wG = 60, 180
    T = 55
    G = 6

    n = len(a)
    problem, (l, t) = construct_lp_problem(a, T, s, G, wA, wG, n)
    problem.solve(lp.GUROBI())
    
    get_value = lambda v: int(lp.value(v) or 0)
    get_arr_value = lambda v: [get_value(x) for x in v]
    objective = get_value(problem.objective)

    start_time = 8 * 60

    data = []
    for i, (ai, li, ti) in enumerate(zip(a, l, t)):
        data.append({
            "id": "FLIGHT" + str(i + 1),
            "arrM": ai + start_time,
            "landM": get_value(li) + start_time,
            "readyM": get_value(li) + T + start_time,
            "takeoffM": get_value(ti) + start_time,
        })

    data.sort(key=lambda x: x["arrM"])

    print("Target:   ", a, [x + T for x in get_arr_value(l)])
    print("Scheduled:", get_arr_value(l), get_arr_value(t))
    print("Objective (minimal):", objective)

    at.render_timeline(data)

if __name__ == '__main__':
    main()
