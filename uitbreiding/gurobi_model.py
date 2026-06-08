from enum import auto, Enum
from dataclasses import dataclass

import gurobipy as gp


M = 300

class PlaneType(Enum):
    SMALL = auto()
    LARGE = auto()


def add_sep_constraint(model: gp.Model, expr: gp.LinExpr, sep: int) -> gp.Var:
    x = model.addVar(vtype=gp.GRB.BINARY)
    model.addGenConstrIndicator(x, True, expr, gp.GRB.GREATER_EQUAL, sep)
    model.addGenConstrIndicator(x, False, expr, gp.GRB.LESS_EQUAL, -sep)
    # model.addConstr(expr >= sep - M * (1 - x))
    # model.addConstr(expr <= -sep + M * x)
    return x


@dataclass
class LandingProblem:
    types: list[PlaneType]
    arrivals: list[int]
    turnarounds: dict[PlaneType, int]
    airtime_weights: dict[PlaneType, int]
    airtime_multiplier: int
    separation_time: int
    gate_capacity: int

    def construct_model(
        self,
        landing_start: list[int] | None = None,
        takeoff_start: list[int] | None = None,
    ) -> gp.Model:
        n = len(self.types)
        turnarounds = [self.turnarounds[t] for t in self.types]
        weights = [self.airtime_weights[t] for t in self.types]

        model = gp.Model("landing-model")
        
        landings = model.addVars(n, lb=0, ub=M, vtype=gp.GRB.INTEGER, name="l")
        takeoffs = model.addVars(n, lb=0, ub=M, vtype=gp.GRB.INTEGER, name="t")
        for i in range(n):
            if landing_start:
                landings[i].Start = landing_start[i]
            if takeoff_start:
                takeoffs[i].Start = takeoff_start[i]

        airtime_sum = self.airtime_multiplier * gp.quicksum(
            weights[i] * (landings[i] - self.arrivals[i])
            for i in range(n)
        )
        ground_sum = gp.quicksum(
            takeoffs[i] - landings[i] - turnarounds[i]
            for i in range(n)
        )
        model.setObjective(airtime_sum + ground_sum)

        all_y = {}
        for i in range(n):
            for j in range(i):
                add_sep_constraint(
                    model,
                    landings[i] - landings[j],
                    self.separation_time,
                )
                y = add_sep_constraint(
                    model,
                    takeoffs[i] - takeoffs[j],
                    self.separation_time,
                )
                all_y[i, j] = y
                all_y[j, i] = 1 - y
                
        for i in range(n):
            model.addConstr(landings[i] - self.arrivals[i] >= 0)
            model.addConstr(takeoffs[i] - landings[i] - turnarounds[i] >= 0)

            plane_count = gp.LinExpr()
            for j in range(n):
                if i == j: continue

                z = add_sep_constraint(
                    model,
                    takeoffs[i] - landings[j],
                    self.separation_time,
                )
                plane_count += z - all_y[i, j]

            model.addConstr(plane_count + 1 <= self.gate_capacity)
        
        self.model = model
        self.landings = list(landings.values())
        self.takeoffs = list(takeoffs.values())
        return model
    
    def solve(self) -> tuple[list[int], list[int]]:
        if not hasattr(self, 'model'):
            self.construct_model()

        self.model.Params.MIPFocus = 3
        # self.model.Params.Presolve = 2
        # self.model.Params.PreSparsify = 2
        # self.model.Params.Heuristics = 0.3
        self.model.Params.Cuts = 1
        # self.model.Params.OBBT = 3
        self.model.Params.NoRelHeurTime = 10

        self.model.optimize()
        return [round(l.X) for l in self.landings], [round(t.X) for t in self.takeoffs]

    def verify_solution(self, landings: list[int], takeoffs: list[int]) -> int:
        objective = 0
        for landing, takeoff, arrival, type in zip(landings, takeoffs, self.arrivals, self.types):
            assert takeoff - landing >= self.turnarounds[type]
            assert landing >= arrival
            objective += self.airtime_multiplier * self.airtime_weights[type] * (landing - arrival)
            objective += takeoff - landing - self.turnarounds[type]

        for i, time1 in enumerate(landings + takeoffs):
            for j, time2 in enumerate(landings + takeoffs):
                if i == j:
                    continue
                assert abs(time1 - time2) >= self.separation_time, \
                    f"Planes {i} and {j} are not separated enough ({time1} and {time2})."

        gate = 0
        for time in range(max(landings + takeoffs)):
            if time in landings:
                gate += 1
            if time in takeoffs:
                gate -= 1
            assert gate <= self.gate_capacity, \
                f"Gate overloaded by plane {landings.index(time)} at {time}."

        return objective


def main():
    turnarounds = {
        PlaneType.SMALL: 45,  # in minutes
        PlaneType.LARGE: 90,
    }
    airtime_weights = {
        PlaneType.SMALL: 10,  # for now, the number of passengers / 10
        PlaneType.LARGE: 25,
    }
    S, L = PlaneType.SMALL, PlaneType.LARGE
    arrivals = [6, 7, 12, 14, 25, 27, 30, 42, 49, 51, 66, 67, 78, 81, 97, 102, 104, 113, 127, 132]
    types = [S, L, S, S, L, S, L, S, S, L, S, L, S, S, L, L, S, S, S, S]
    problem = LandingProblem(
        types,
        arrivals,
        turnarounds,
        airtime_weights,
        airtime_multiplier=3,
        separation_time=4,
        gate_capacity=5
    )

    problem.construct_model(
    #    [6, 10, 18, 14, 25, 119, 55, 157, 206, 63, 165, 71, 243, 214, 104, 149, 222, 198, 251, 173],
    #    [51, 100, 67, 59, 115, 169, 145, 202, 255, 153, 210, 161, 288, 259, 194, 239, 267, 247, 296, 218],
    )

    landings, takeoffs = problem.solve()

    # outfile = open("model.txt", "w")
    print("Arrivals : ", arrivals)#, file=outfile)
    print("Landings : ", landings)#, file=outfile)
    print("Takeoffs : ", takeoffs)#, file=outfile)

    checked_objective = problem.verify_solution(landings, takeoffs)
    print("Objective: ", checked_objective)#, file=outfile)


if __name__ == '__main__':
    main()
