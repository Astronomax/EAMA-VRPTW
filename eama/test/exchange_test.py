from eama.exchange import Exchange, ExchangeType
from eama.meta_wrapper import MetaWrapper, RouteWrapper, CustomerWrapper
from eama.structure import Problem, Customer
from eama.routelist import RouteList

import unittest


class TestExchange(unittest.TestCase):
    customers = [Customer(i, 0, 0, 0, 0, 100, 0) for i in range(7)]
    problem = Problem('test', customers, 100, 100)
    solution = MetaWrapper(problem=problem)
    solution._routes = [
        RouteWrapper(problem, RouteList(problem, list(map(CustomerWrapper, customers[1:4])))),
        RouteWrapper(problem, RouteList(problem, list(map(CustomerWrapper, customers[4:7])))),
    ]       

    def test_two_opt(self):
        customers = TestExchange.customers
        solution = TestExchange.solution.inherit()
        e = Exchange(solution._routes[0][0], solution._routes[1][0], ExchangeType.TwoOpt)
        self.assertTrue(e.appliable())
        self.assertTrue(e.apply())
        self.assertListEqual([v.number for v in solution._routes[0].customers()],\
                             [v.number for v in [customers[1], customers[5], customers[6]]])
        self.assertListEqual([v.number for v in solution._routes[1].customers()],\
                             [v.number for v in [customers[4], customers[2], customers[3]]])     
        self.assertTrue([v.route() is route for route in solution._routes for v in route])
        self.assertTrue([v.node().value is v for route in solution._routes for v in route])

    def test_two_opt_failed(self):
        solution = TestExchange.solution.inherit()
        e = Exchange(solution._routes[0][0].prev(), solution._routes[1][2], ExchangeType.TwoOpt)
        self.assertFalse(e.appliable())
        with self.assertRaises(AssertionError):
            self.assertTrue(e.apply())

    def test_out_relocate(self):
        customers = TestExchange.customers
        solution = TestExchange.solution.inherit()
        e = Exchange(solution._routes[0][1], solution._routes[1][1], ExchangeType.OutRelocate)
        self.assertTrue(e.appliable())
        self.assertTrue(e.apply())
        self.assertListEqual([v.number for v in solution._routes[0].customers()],\
                             [v.number for v in [customers[1], customers[3]]])
        self.assertListEqual([v.number for v in solution._routes[1].customers()],\
                             [v.number for v in [customers[4], customers[2], customers[5], customers[6]]])     
        self.assertTrue([v.route() is route for route in solution._routes for v in route])
        self.assertTrue([v.node().value is v for route in solution._routes for v in route])

    def test_exchange(self):
        customers = TestExchange.customers
        solution = TestExchange.solution.inherit()
        e = Exchange(solution._routes[0][1], solution._routes[1][1], ExchangeType.Exchange)
        self.assertTrue(e.appliable())
        self.assertTrue(e.apply())
        self.assertListEqual([v.number for v in solution._routes[0].customers()],\
                             [v.number for v in [customers[1], customers[5], customers[3]]])
        self.assertListEqual([v.number for v in solution._routes[1].customers()],\
                             [v.number for v in [customers[4], customers[2], customers[6]]])     
        self.assertTrue([node.value.route() is route for route in solution._routes for node in route._route.head.iter()])
        self.assertTrue([v.node().value is v for route in solution._routes for v in route])

if __name__ == '__main__':
    unittest.main()