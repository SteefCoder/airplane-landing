import pandas as pd

small_dataset_file = "/{use your own path for the data, uploaded both on github!}/Data assignment aircraft landing 2 Small.xlsx"
large_dataset_file = "/{use your own path for the data, uploaded both on github!}/Data assignment aircraft landing 2 Large.xlsx"


def load_dataset(filepath):
    info = pd.read_excel(filepath, sheet_name="General information", header=None)
    times = pd.read_excel(filepath, sheet_name="Times per aircraft")
    sep = info.iloc[1, 1]
    E = list(times["Earliest landing time"])
    T = list(times["Target landing time"])
    L = list(times["Latest landing time"])
    planes = list(zip(E, T, L))
    return sep, planes

def solve(sep, planes):
    order = sorted(range(len(planes)), key=lambda i: planes[i][1])
    last_x = 0
    results = []
    for i in order:
        E, T, L = planes[i]
        floor = max(E, last_x + sep)
        x = max(floor, min(T, L))
        deviation = abs(x - T)
        results.append((i + 1, x, deviation))
        last_x = x
    return results

def print_results(name, results):
    print(f"\n{name}")
    total = 0
    for plane, x, dev in results:
        print(f"plane {plane}, landed at {x}, deviation {dev}")
        total += dev
    print(f"total deviaton: {total}")

sep_s, planes_s = load_dataset(small_dataset_file)
sep_l, planes_l = load_dataset(large_dataset_file)

print_results("Small dataset", solve(sep_s, planes_s))
print_results("Large dataset", solve(sep_l, planes_l))