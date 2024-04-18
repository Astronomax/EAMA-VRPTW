from eama.meta_wrapper import MetaWrapper
from eama.ejection import Ejection, feasible_ejections
from generators import generate_random_problem, generate_random_solution
from random import choice

import unittest


def random_ejection_test_factory(num_tests=100):
    def test_method(self):
        for _ in range(num_tests):
            problem = generate_random_problem()
            solution = generate_random_solution(problem)
            ind, route = choice(list(enumerate(solution._routes)))
            p = list(range(len(problem.customers)))
            for e, p_sum in feasible_ejections(route, p, 5):
                c_delta = e.penalty_delta(1, 0)
                tw_delta = e.penalty_delta(0, 1)
                dist_delta = e.distance_delta()
                routes = [[v._customer for v in route] for route in solution._routes]
                solution_copy = MetaWrapper(problem=problem, routes=routes)
                route_copy = solution_copy._routes[ind]
                c_before_ejection = route_copy._pc.get_penalty(1, 0)
                tw_before_ejection = route_copy._pc.get_penalty(0, 1)
                dist_before_ejection = route_copy._dc.get_distance()
                ejection = [v.number for v in e._ejection]
                p_sum_target = sum([p[num] for num in ejection])
                ejection = list(filter(lambda v: v.number in ejection, route_copy))
                Ejection(solution_copy, ejection, 0, 0, 0).apply()
                c_target = route_copy._pc.get_penalty(1, 0) - c_before_ejection
                tw_target = route_copy._pc.get_penalty(0, 1) - tw_before_ejection
                dist_target = route_copy._dc.get_distance() - dist_before_ejection
                self.assertAlmostEqual(c_delta, c_target, delta=1e-4)
                self.assertAlmostEqual(tw_delta, tw_target, delta=1e-4)
                self.assertAlmostEqual(dist_delta, dist_target, delta=1e-4)
                self.assertAlmostEqual(p_sum, p_sum_target, delta=1e-4)
                self.assertTrue(route_copy.feasible())
    return test_method


class TestEjection(unittest.TestCase):
    test_random_ejections = random_ejection_test_factory(100)
