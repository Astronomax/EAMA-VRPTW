from eama.structure import Customer, Problem, Route
from eama.routelist import ListNode, RouteList
from eama.penalty_calculator import PenaltyCalculator
from eama.distance_calculator import DistanceCalculator
from eama.exchange import exchanges
from copy import copy


class RouteWrapper:
    def __init__(self, problem: Problem, route: RouteList, meta_wrapper: 'MetaWrapper'=None):
        self._problem = problem
        self._route = route
        self._meta_wrapper = meta_wrapper
        for node in self._route.head.iter():
            node.value._index = CustomerIndex(self, node)
        self._pc = PenaltyCalculator(self)
        self._dc = DistanceCalculator(self)

    def __len__(self):
        return len(self._route)

    def __getitem__(self, index):
        return self._route[index]

    def __setitem__(self, index, value: 'CustomerWrapper'):
        assert value.ejected()
        node = self._route.get_node(index + 1)
        node.value._index = None
        node.value = value
        value._index = CustomerIndex(self, node)

    def __iter__(self):
        for v in self._route:
            yield v

    def insert(self, index, value: 'CustomerWrapper'):
        assert value.ejected()
        i = index.node() if isinstance(index, CustomerWrapper) else index
        value._index = CustomerIndex(self, self._route.insert(i, value))

    def eject(self, value: 'CustomerWrapper'):
        assert value.route() == self
        self._route.remove(value._index._node)
        value._index = None

    def customers(self):
        for v in self._route:
            yield v._customer

    def __copy__(self):
        return RouteWrapper(self._problem, RouteList(self._problem, [CustomerWrapper(v) for v in self._route]))
    
    def get_penalty(self, alpha, beta):
        return self._pc.get_penalty(alpha, beta)
    
    def get_distance(self):
        return self._dc.get_distance()

    def feasible(self):
        return (self.get_penalty(1, 1) == 0)


class CustomerIndex:
    def __init__(self, route: RouteWrapper = None, node: ListNode = None):
        self._route = route
        self._node = node


class CustomerWrapper:
    def __init__(self, customer: Customer, index: CustomerIndex = None):
        self._customer: Customer = customer
        self._index: CustomerIndex = index

    def next(self):
        assert self._index._node.tail() != None
        return self._index._node.next.value
    
    def prev(self):
        assert self._index._node.head() != None
        return self._index._node.prev.value
    
    def ejected(self):
        return self._index is None
    
    def route(self):
        return self._index._route
    
    def node(self):
        return self._index._node
    
    def pc(self):
        return self._index._route._pc
    
    def dc(self):
        return self._index._route._dc
    
    def __getattr__(self, attr):
        if hasattr(self.__dict__['_customer'], attr):
            return getattr(self.__dict__['_customer'], attr)
        else:
            raise AttributeError(f"'CustomerWrapper' object has no attribute '{attr}'")


class MetaWrapper:
    def __init__(self, problem: Problem = None, routes: list[Customer] = None):
        self.problem = problem
        self._ejection_pool = []
        if problem:
            if routes:
                customers = set([v.number for v in sum(routes, [])])
                assert self.problem.depot.number not in customers
                assert len(set(customers)) == len(customers)
                customers.add(problem.depot.number)
                #assert set(customers) == set([v.number for v in problem.customers])
                self._routes = [RouteWrapper(self.problem, RouteList(self.problem, [CustomerWrapper(v) for v in route]), self) for route in routes]
                customers = set(customers)
                self._ejection_pool = list(map(CustomerWrapper, (filter(lambda x: x.number not in customers, problem.customers))))
                assert self.valid()
                customers = sum(routes, [])
            else:
                customers = list(map(CustomerWrapper, filter(lambda x: x.number != problem.depot.number, problem.customers)))
                self._routes = [RouteWrapper(self.problem, RouteList(self.problem, [customer]), self) for customer in customers]
            # precalc nearest-neighbourhood of each customer
            # it will be useful for some heuristic optimizations
            self.nearest = {v: sorted(customers.copy(), key=lambda u: v.c(u)) for v in customers}

    def get_penalty(self, alpha, beta):
        return sum([route.get_penalty(alpha, beta) for route in self._routes])
    
    def get_distance(self):
        return sum([route.get_distance() for route in self._routes])
    
    def activate(self):
        for route in self._routes:
            for node in route._route.head.iter():
                node.value._index = CustomerIndex(route, node)
        for v in self._ejection_pool:
            v._index = None

    def inherit(self):
        result = MetaWrapper()
        result.problem = self.problem
        result._ejection_pool = self._ejection_pool.copy()
        result.nearest = self.nearest # const
        result._routes = [RouteWrapper(self.problem, copy(route._route), result) for route in self._routes]

        for route in result._routes:
            for node in route._route.head.iter():
                assert node.value.route() is route
                assert node.value in node.value.pc()._index
                assert node.value in route._pc._index

        return result

    def __copy__(self):
        pass

    #def not_ejected(self):
    #    for route in self._routes:
    #        for v in route:
    #            yield v

    # \mathcal{N}(v, \sigma)
    #def N(self, v: CustomerWrapper = None):
    #    for v in self.not_ejected():
    #        for w in self.not_ejected():
    #            for e in exchanges(v, w):
    #                yield e

    # \mathcal{N}_near(v, \sigma)
    def N_near(self, n_near: int, v: CustomerWrapper = None, route: RouteWrapper = None):
        if route is not None:
            assert v is None
            customers = (v.value for v in route._route.head().iter())
        elif v is not None:
            customers = [v]
        else:
            customers = (v for route in self._routes for v in route)
        for v in customers:
            for w in self.nearest[v][:n_near]:
                if not w.ejected():
                    for e in exchanges(v, w):
                        if e.appliable():
                            yield e

    def feasible(self):
        return all([route.feasible() for route in self._routes])
    
    def completed(self):
        return self.feasible() and len(self._ejection_pool) == 0
    
    def valid(self, currently_ejected_node = None):
        customers = [v.number for route in self._routes for v in route] + [v.number for v in self._ejection_pool]
        if currently_ejected_node is not None:
            customers.extend([currently_ejected_node.number])
        if len(customers) != len(set(customers)):
            return False
        if len(set(customers)) != len(self.problem.customers) - 1:
            return False
        return True
    
    def get_solution(self):
        return [Route(self.problem, [v._customer for v in route]) for route in self._routes]