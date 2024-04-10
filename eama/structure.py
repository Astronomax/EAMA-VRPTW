import math


class Customer:
    def __init__(self, number, x, y, demand, e, l, s):
        self.number = number
        self.x = x
        self.y = y
        self.demand = demand
        self.e = e
        self.l = l
        self.s = s
        self.is_serviced = False

    def __repr__(self):
        return f"C_{self.number}: x={self.x}, y={self.y}, demand = {self.demand}, from_date={self.e}, due_date={self.l}, service_time={self.s}"

    def depot(self):
        return self.number == 0

    def c(self, target):
        return math.sqrt(math.pow(self.x - target.x, 2) + math.pow(self.y - target.y, 2))


class Problem:
    def __init__(self, name, customers: list, vehicle_number, vehicle_capacity):
        self.name = name
        self.customers = customers
        self.vehicle_number = vehicle_number
        self.vehicle_capacity = vehicle_capacity
        self.depot: Customer = list(filter(lambda x: x.number == 0, customers))[0]
        self.depot.is_serviced = True

    def __repr__(self):
        customers_repr = "\n".join(list(map(lambda x: str(x), self.customers)))
        return f"Instance: {self.name}\n" \
               f"Vehicle number: {self.vehicle_number}\n" \
               f"Vehicle capacity: {self.vehicle_capacity}\n" \
               f"Customers:\n{customers_repr}"

    def obj_func(self, routes):
        return sum(map(lambda x: x.total_distance, routes))

    def print_canonical(self, routes):
        return "\n".join(list(map(lambda x: x.canonical_view, routes)))


class Route:
    def __init__(self, problem: Problem, customers: list):
        self.problem: Problem = problem
        self._customers: list = [self.problem.depot, *customers, self.problem.depot]

    def __repr__(self):
        time = 0
        result = [[0, 0.0]]
        for source, target in zip(self._customers, self._customers[1:]):
            start_time = max([target.e, time + source.c(target)])
            time = start_time + target.s
            result.append([target.number, start_time])
        temp = "\n".join(f'Node: {x[0]}, time: {x[1]}' for x in result);
        return f'[{temp}]'

    def customers(self):
        return self._customers[1:-1]

    def total_distance(self):
        time = 0
        for source, target in zip(self._customers, self._customers[1:]):
            start_time = max([target.e, time + source.c(target)])
            time = start_time + target.s
        return time

    def edges(self):
        return list(zip(self._customers, self._customers[1:]))

    def sum_demand(self):
        return sum([c.demand for c in self._customers])

    def is_feasible(self):
        time = 0
        capacity = self.problem.vehicle_capacity
        is_feasible = True
        for source, target in zip(self._customers, self._customers[1:]):
            start_service_time = max([target.from_date, time + source.distance(target)])
            if start_service_time >= target.due_date:
                is_feasible = False
            time = start_service_time + target.service_time
            capacity -= target.demand
        if time >= self.problem.depot.due_date or capacity < 0:
            is_feasible = False
        return is_feasible

    @property
    def canonical_view(self):
        time = 0
        result = [0, 0.0]
        for source, target in zip(self._customers, self._customers[1:]):
            start_time = max([target.e, time + source.c(target)])
            time = start_time + target.s
            result.append(target.number)
            result.append(start_time)
        return " ".join(str(x) for x in result)


class Solution:
    def __init__(self, routes: list):
        self.routes: list = routes

    def __repr__(self):
        return "\n".join([f'route {i+1}:\n{str(x)}' for i, x in enumerate(self.routes)])
