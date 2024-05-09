import argparse
import os

from eama import *


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Solutions validator for VRPTW problem')
    parser.add_argument('problem', type=str, help='Problem file (in Solomon format)')
    parser.add_argument('solution', type=str, default=False, help="Solution file")
    return parser.parse_args()


def check_route(problem, line):
    route_customers, timestamps = line.split()[::2], line.split()[1::2]
    assert route_customers[0] == route_customers[-1] == '0', "Each route should begins and ends at the DEPOT"
    customers = []
    for c, t in zip(route_customers, timestamps):
        customer = list(filter(lambda x: x.number == int(c), problem.customers))[0]
        customers.append(customer)
        customer.is_serviced = True
        if not (customer.e <= float(t) < customer.l):
            print(customer.e, float(t), customer.l)
        assert customer.e <= float(t) < customer.l, f"Time violation for customer {c}"
    assert sum([x.demand for x in customers]) <= problem.vehicle_capacity, f"Capacity violation for line {line}"
    distance = 0
    for a, b in zip(customers[:-1], customers[1:]):
        distance += a.c(b)
    c_tilde = sum([max(0, -c.demand) for c in customers])
    pfsum = c_tilde
    for c in customers:
        pfsum += c.demand
        assert(pfsum <= problem.vehicle_capacity)
    return distance


def check_solution(problem: Problem, solution_file: str):
    with open(solution_file, 'r') as f:
        routes = f.readlines()
    assert len(routes) <= problem.vehicle_number, "Vehicle number violation"

    vehicles = len(routes)
    distance = 0
    for route in routes:
        distance += check_route(problem, route)
    count = 0
    for cust in problem.customers:
        if cust.is_serviced:
            count += 1
    assert all(map(lambda x: x.is_serviced, problem.customers)), "Each customer must be served"
    return vehicles, distance


if __name__ == '__main__':
    args = arguments()
    assert os.path.exists(args.problem), "Incorrect problem file"
    assert os.path.exists(args.solution), "Incorrect solution file"
    problem = SolomonFormatParser(os.path.abspath(args.problem)).get_problem()
    vehicles, distance = check_solution(problem, os.path.abspath(args.solution))
    print(vehicles, distance)