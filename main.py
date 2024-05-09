from eama import *

import argparse
import sys
import os
import time
import cProfile
from random import seed
from random import randint, choice
from itertools import combinations
from copy import copy
from math import inf

seed(239)

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EAMA VRPTW solver')
    parser.add_argument('problem', type=str, help='Problem file (in Solomon format)')
    return parser.parse_args()


def generate_random_customer(number):
    x = randint(0, 100)
    y = randint(0, 100)
    demand = randint(0, 10) - 5
    e = randint(0, 50)
    l = randint(e, 100)
    s = randint(0, 20)
    return Customer(number, x, y, demand, e, l, s)

def generate_random_problem(max_n: int = 100):
    customers_num = randint(2, max_n)
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
    solution = MetaWrapper(problem=problem, routes=routes)
    for nearest in solution.nearest.values():
        for v in nearest:
            assert isinstance(v, CustomerWrapper)
    return solution

def subsets_lexicographic_order(nums, k):
    nums.sort()
    n = len(nums)
    for r in range(1, k + 1):
        for combo in combinations(nums, r):
            yield list(combo)

if __name__ == '__main__':
    args = arguments()
    assert os.path.exists(args.problem), "Incorrect problem file"
    #problem_path = "instances/my_instance.TXT"
    problem = SolomonFormatParser(args.problem).get_problem()

    start_time = time.time()
    rmh_settings = RMHSettings()
    rmh_settings.i_rand = 1000
    rmh_settings.t_max = 5 * 60
    rmh_settings.lower_bound = 0
    gip_settings = GIPSettings()
    eama_settings = EAMASettings()
    eama = EAMA(problem, rmh_settings=rmh_settings, gip_settings=gip_settings, eama_settings=eama_settings, debug=True)
    #result = eama.powerful_route_minimization_heuristic(rmh_settings)

    def wrapper_function():
        result = eama.powerful_route_minimization_heuristic(rmh_settings)
        s = result.get_solution()
        print(f'elapsed time: {time.time() - start_time}')
        print(f'routes: {len(s)}')
        with open(f"""solutions/{args.problem.split(os.sep)[-1].split(".")[0]}.sol""", 'w') as f:
            f.write(problem.print_canonical(s))

    #cProfile.run('wrapper_function()')
    wrapper_function()


    '''
    problem = generate_random_problem(20)
    solution = generate_random_solution(problem)
    ind, route = choice(list(enumerate(solution._routes)))
    p = [randint(0, 5) for _ in range(len(problem.customers))]
    n = len(route)
    all_ejections = sorted(subsets_lexicographic_order(list(range(n)), 5))
    ejections = [(copy(e), p_sum) for e, p_sum in feasible_ejections(route, p, 5)]
    #print([([v.number for v in e[0]._ejection], e[1]) for e in ejections])
    
    print([node.value._customer for node in route._route.head.iter()])

    j = 0
    p_best = inf
    for ejection_list in all_ejections:
        solution_copy = solution.inherit()
        route_copy = solution_copy._routes[ind]
        ejection = [route_copy[i] for i in ejection_list]
        Ejection(solution_copy, ejection, 0, 0, 0).apply()
        p_sum = sum([p[v.number] for v in ejection])
        if route_copy.feasible():
            if p_sum < p_best:
                assert j < len(ejections)
                if j == len(ejections):
                    print([c._customer for c in ejection])
                    sys.exit(0)
                if not all([x == y for x, y in zip(ejections[j][0]._ejection, ejection)]):
                    print([node.value.number for node in route._route.head.iter()])
                    for node in route._route.head.iter():
                        print(node.value._customer)
                    assert False
                    #sys.exit(0)
                #assert(ejections[j][0]._ejection, ejection)
                #assert(ejections[j][1], p_sum)
                p_best = p_sum
                j += 1
        solution.activate()
    assert j == len(ejections)
    '''

