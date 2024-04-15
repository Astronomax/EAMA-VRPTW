from eama.routelist import RouteList
from eama.meta_wrapper import RouteWrapper, CustomerWrapper
from eama.exchange import ExchangeFast, ExchangeType
from eama.insertion import Insertion
from eama.distance_calculator import DistanceCalculator
from eama.structure import Problem, Customer
from random import choice
from generators import generate_random_problem, generate_random_solution

import unittest


def random_exchanges_test_factory(num_tests=100):
    def test_method(self):
        for _ in range(num_tests):
            problem = generate_random_problem()
            solution = generate_random_solution(problem)
            v = choice(list(choice(solution._routes)))
            w = choice(list(choice(solution._routes)))
            type = choice(list(ExchangeType))
            e = ExchangeFast(v, w, type)
            if e.appliable():
                v_route = v.route()
                w_route = w.route()
                dist_before_exchange = v_route._dc.get_distance() + w_route._dc.get_distance()
                dist_delta = e.distance_delta()
                e.apply()
                dist_target = (v_route._dc.get_distance() + w_route._dc.get_distance()) - dist_before_exchange
                if v_route is w_route:
                    dist_target /= 2
                self.assertAlmostEqual(dist_delta, dist_target, delta=1e-4)

    return test_method

def random_insertions_test_factory(num_tests=100):
    def test_method(self):
        for _ in range(num_tests):
            problem = generate_random_problem()
            solution = generate_random_solution(problem)
            v = choice(list(choice(solution._routes)))
            w = choice(list(choice(solution._routes)))
            e = Insertion(v, w)
            v_route = v.route()
            w_route = w.route()
            if v_route is w_route:
                continue
            w_route.eject(w)
            assert w.ejected()
            if e.appliable():
                dist_before_exchange = v.dc().get_distance()
                dist_delta = e.distance_delta()
                e.apply()
                dist_target = v.dc().get_distance() - dist_before_exchange
                self.assertAlmostEqual(dist_delta, dist_target, delta=1e-4)

    return test_method

def random_ejections_test_factory(num_tests=100):
    def test_method(self):
        for _ in range(num_tests):
            problem = generate_random_problem()
            solution = generate_random_solution(problem)
            v = choice(list(choice(solution._routes)))
            dist_before_exchange = v.dc().get_distance()
            dist_delta = DistanceCalculator.get_eject_delta(v)
            route = v.route()
            route.eject(v)
            assert v.ejected()
            route._dc.update()
            dist_target = route._dc.get_distance() - dist_before_exchange
            self.assertAlmostEqual(dist_delta, dist_target, delta=1e-4)

    return test_method


class TestDistanceCalculator(unittest.TestCase):
    test_random_exchanges = random_exchanges_test_factory(100)
    test_random_insertions = random_insertions_test_factory(100)
    test_random_ejections = random_ejections_test_factory(100)
