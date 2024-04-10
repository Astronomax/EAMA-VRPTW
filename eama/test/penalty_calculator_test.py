from eama.routelist import RouteList
from eama.meta_wrapper import RouteWrapper, CustomerWrapper
from eama.exchange import Exchange, ExchangeType
from eama.insertion import Insertion
from eama.penalty_calculator import PenaltyCalculator
from eama.structure import Problem, Customer
from random import choice
from generators import generate_random_problem, generate_random_solution

import unittest


def random_exchanges_test_factory(num_tests=100):
    def test_method(self):
        for _ in range(num_tests):
            problem = generate_random_problem()
            solution = generate_random_solution(problem)
            v = choice(list(choice(solution._routes)._route.head.iter())).value
            w = choice(list(choice(solution._routes)._route.head.iter())).value
            type = choice(list(ExchangeType))
            e = Exchange(v, w, type)
            if e.appliable():
                v_route = v.route()
                w_route = w.route()
                c_before_exchange = v_route._pc.get_penalty(1, 0) + w_route._pc.get_penalty(1, 0)
                tw_before_exchange = v_route._pc.get_penalty(0, 1) + w_route._pc.get_penalty(0, 1)
                c_delta = e.penalty_delta(1, 0)
                tw_delta = e.penalty_delta(0, 1)
                e.apply()

                self.assertTrue([node.value.route() is route for route in solution._routes for node in route._route.head.iter()])
                self.assertTrue([v.node().value is v for route in solution._routes for v in route])

                c_target = (v_route._pc.get_penalty(1, 0) + w_route._pc.get_penalty(1, 0)) - c_before_exchange
                tw_target = (v_route._pc.get_penalty(0, 1) + w_route._pc.get_penalty(0, 1)) - tw_before_exchange
                if v_route is w_route:
                    c_target /= 2
                    tw_target /= 2
                self.assertAlmostEqual(c_delta, c_target, delta=1e-4)
                self.assertAlmostEqual(tw_delta, tw_target, delta=1e-4)

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
                c_before_exchange = v.pc().get_penalty(1, 0)
                tw_before_exchange = v.pc().get_penalty(0, 1)
                c_delta = e.penalty_delta(1, 0)
                tw_delta = e.penalty_delta(0, 1)
                e.apply()
                c_target = v.pc().get_penalty(1, 0) - c_before_exchange
                tw_target = v.pc().get_penalty(0, 1) - tw_before_exchange
                self.assertAlmostEqual(c_delta, c_target, delta=1e-4)
                self.assertAlmostEqual(tw_delta, tw_target, delta=1e-4)

    return test_method

def random_ejections_test_factory(num_tests=100):
    def test_method(self):
        for _ in range(num_tests):
            problem = generate_random_problem()
            solution = generate_random_solution(problem)
            v = choice(list(choice(solution._routes)))

            c_before_exchange = v.pc().get_penalty(1, 0)
            tw_before_exchange = v.pc().get_penalty(0, 1)
            c_delta = PenaltyCalculator.get_eject_delta(v, 1, 0)
            tw_delta = PenaltyCalculator.get_eject_delta(v, 0, 1)
            route = v.route()
            route.eject(v)
            assert v.ejected()
            route._pc.update()
            c_target = route._pc.get_penalty(1, 0) - c_before_exchange
            tw_target = route._pc.get_penalty(0, 1) - tw_before_exchange
            self.assertAlmostEqual(c_delta, c_target, delta=1e-4)
            self.assertAlmostEqual(tw_delta, tw_target, delta=1e-4)

    return test_method


class TestPenaltyCalculator(unittest.TestCase):
    customers = [
        Customer(0, 0, 0, 0, 0, 100, 0),
        Customer(1, 0, 1, 1, 0, 2, 1),
        Customer(2, 0, 2, 1, 2, 4, 1),
        Customer(3, 0, 3, 1, 0, 2, 1),
        Customer(4, 0, 4, 1, 2, 4, 1),
    ]
    problem = Problem('test', customers, 100, 100)
    route = RouteWrapper(problem, RouteList(problem, list(map(CustomerWrapper, customers[1:]))))

    def test_simple(self):
        pc = TestPenaltyCalculator.route._pc
        route = [node.value for node in TestPenaltyCalculator.route._route.head.iter()]
        n = len(route)
        self.assertEqual(pc.get_penalty(0, 1), 3.0)
        self.assertEqual(pc.get_penalty(1, 0), 0.0)
        depot = TestPenaltyCalculator.problem.depot
        a = a_quote = depot.e
        self.assertEqual(pc.a[0], a)
        self.assertEqual(pc.a_quote[0], a_quote)
        tw_pf = 0
        demand_pf = depot.demand
        self.assertEqual(pc.tw_pf[0], tw_pf)
        self.assertEqual(pc.demand_pf[0], demand_pf)
        for i in range(1, n):
            next = route[i]
            prev = next.prev()
            a_quote = a + prev.s + prev.c(next)
            a = min(max(a_quote, next.e), next.l)
            self.assertEqual(pc.a[i], a)
            self.assertEqual(pc.a_quote[i], a_quote)
            tw_pf += max(a_quote - next.l, 0)
            demand_pf += next.demand
            self.assertEqual(pc.tw_pf[i], tw_pf)
            self.assertEqual(pc.demand_pf[i], demand_pf)
        z = z_quote = depot.l
        self.assertEqual(pc.z[n - 1], z)
        self.assertEqual(pc.z_quote[n - 1], z_quote)
        tw_sf = 0
        demand_sf = depot.demand
        self.assertEqual(pc.tw_sf[n - 1], tw_sf)
        self.assertEqual(pc.demand_sf[n - 1], demand_sf)
        for i in reversed(range(n - 1)):
            prev = route[i]
            next = prev.next()
            z_quote = z - prev.s - prev.c(next)
            z = min(max(z_quote, prev.e), prev.l)
            self.assertEqual(pc.z[i], z)
            self.assertEqual(pc.z_quote[i], z_quote)
            tw_sf += max(prev.e - z_quote, 0)
            demand_sf += prev.demand
            self.assertEqual(pc.tw_sf[i], tw_sf)
            self.assertEqual(pc.demand_sf[i], demand_sf)

    test_random_exchanges = random_exchanges_test_factory(100)
    test_random_insertions = random_insertions_test_factory(100)
    test_random_ejections = random_ejections_test_factory(100)
