from eama.meta_wrapper import MetaWrapper
from eama.structure import Problem, Customer
from random import randint


def generate_random_customer(number):
    x = randint(0, 100)
    y = randint(0, 100)
    demand = randint(0, 10)
    e = randint(0, 50)
    l = randint(e, 100)
    s = randint(0, 20)
    return Customer(number, x, y, demand, e, l, s)

def generate_random_problem():
    customers_num = randint(2, 100)
    customers = [generate_random_customer(i) for i in range(customers_num)]
    customers[0].s = 0
    customers[0].demand = 0
    return Problem('random', customers, 100, randint(0, 100))
    
def generate_random_solution(problem: Problem):
    routes = []
    customers = problem.customers[1:].copy()
    while len(customers) > 0:
        route_size = randint(1, len(customers))
        route = customers[:route_size]
        routes.append(route)
        customers = customers[route_size:]
    return MetaWrapper(problem, routes)