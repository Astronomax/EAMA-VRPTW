from eama.meta_wrapper import MetaWrapper, RouteWrapper, CustomerWrapper
from eama.ejection import Ejection, feasible_ejections
from eama.structure import Customer, Problem
from eama.routelist import RouteList
from eama.insertion import insertions

from math import inf
from itertools import combinations
from copy import copy

customers = [
    Customer(0,     92,     44,     0,      0,      72,   0),
    Customer(16,    75,    37,    2,     50,     63,    3),
    Customer(17,    4,    49,    2,     34,    48,    7),
    Customer(18,     66,      94,    4,     46,    97,    7),
]
problem = Problem("temp", customers, 1000, 200)
solution = MetaWrapper(problem=problem, routes=[customers[1:]])
'''
route = solution._routes[0]
v = route[0]
route.eject(v, update=True)
print([insertion._index.number for insertion in list(insertions(v, meta_wrapper=solution))])
'''
#'''
route = solution._routes[0]
p = list(range(25))
ejections = [(copy(e), p_sum) for e, p_sum in feasible_ejections(route, p, 5)]
print([([v.number for v in e[0]._ejection], e[1]) for e in ejections])




def subsets_lexicographic_order(nums, k):
    nums.sort()
    n = len(nums)
    for r in range(1, k + 1):
        for combo in combinations(nums, r):
            yield list(combo)

n = len(route)
all_ejections = sorted(subsets_lexicographic_order(list(range(n)), 5))

print(all_ejections)

j = 0
p_best_tmp = inf
for ejection_list in all_ejections:
    solution_copy = solution.inherit()
    route_copy = solution_copy._routes[0]
    ejection = [route_copy[i] for i in ejection_list]
    Ejection(solution_copy, ejection, 0, 0, 0).apply()
    p_sum = sum([p[v.number] for v in ejection])
    if route_copy.feasible():
        if p_sum < p_best_tmp:
            print(([v.number for v in ejection], p_sum))
            #if j == len(ejections):
            #    for node in route._route.head.iter():
            #        print(node.value._customer)
            #    print(ejection_list)
            #assert j < len(ejections)
            #assert all([x == y for x, y in zip(ejections[j][0]._ejection, ejection)])
            #assert ejections[j][1] == p_sum
            p_best_tmp = p_sum
            j += 1
    solution.activate()
#'''