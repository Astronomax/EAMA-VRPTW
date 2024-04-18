from eama.routelist import RouteList
from eama.meta_wrapper import MetaWrapper, RouteWrapper, CustomerWrapper
from eama.structure import Problem, Customer

import unittest


class TestRouteList(unittest.TestCase):
    customers = [Customer(i, 0, 0, 0, 0, 100, 0) for i in range(7)]
    problem = Problem('test', customers, 100, 100)
    solution = MetaWrapper(problem=problem)
    solution._routes = [
        RouteWrapper(problem, RouteList(problem, list(map(CustomerWrapper, customers[1:4])))),
        RouteWrapper(problem, RouteList(problem, list(map(CustomerWrapper, customers[4:7])))),
    ]

    def test_construct_from_list(self):
        customers = TestRouteList.customers
        subset = [customers[1], customers[2], customers[3]]
        mylist = RouteList(TestRouteList.problem, list(map(CustomerWrapper, subset)))
        self.assertListEqual([cw._customer for cw in mylist], subset)
    
    def test_insert_simple(self):
        customers = TestRouteList.customers
        mylist = RouteList(TestRouteList.problem)
        mylist.insert(0, CustomerWrapper(customers[1]))
        self.assertListEqual([cw._customer for cw in mylist], [customers[1]])

    def test_insert_failed(self):
        customers = TestRouteList.customers
        mylist = RouteList(TestRouteList.problem)
        with self.assertRaises(AssertionError):
            mylist.insert(1, CustomerWrapper(customers[1]))
        with self.assertRaises(AssertionError):
            mylist.insert(-1, CustomerWrapper(customers[1]))

    def test_remove_simple(self):
        customers = TestRouteList.customers
        subset = [customers[1], customers[2], customers[3]]
        mylist = RouteList(TestRouteList.problem, list(map(CustomerWrapper, subset)))
        node = mylist.remove(mylist.get_node(2))
        self.assertListEqual([cw._customer for cw in mylist], [customers[1], customers[3]])
        self.assertEqual(node.value._customer, customers[2])

    def test_remove_failed(self):
        mylist = RouteList(TestRouteList.problem)
        with self.assertRaises(AssertionError):
            mylist.remove(0)
        with self.assertRaises(AssertionError):
            customers = TestRouteList.customers
            subset = [customers[1], customers[2], customers[3]]
            mylist = RouteList(TestRouteList.problem, list(map(CustomerWrapper, subset)))
            mylist.remove(3)

    def test_set(self):
        customers = TestRouteList.customers
        subset = [customers[1], customers[2], customers[3]]
        mylist = RouteList(TestRouteList.problem, list(map(CustomerWrapper, subset)))
        mylist[1] = CustomerWrapper(customers[4])
        self.assertListEqual([cw._customer for cw in mylist], [customers[1], customers[4], customers[3]])

    def test_get(self):
        customers = TestRouteList.customers
        subset = [customers[1], customers[2], customers[3]]
        mylist = RouteList(TestRouteList.problem, list(map(CustomerWrapper, subset)))
        self.assertEqual(mylist[1]._customer, customers[2])

if __name__ == '__main__':
    unittest.main()