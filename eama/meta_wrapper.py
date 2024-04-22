from eama.structure import Customer, Problem, Route
from eama.routelist import ListNode, RouteList
from eama.penalty_calculator import PenaltyCalculator
from eama.distance_calculator import DistanceCalculator
from eama.exchange import exchanges, ExchangeSlow, ExchangeFast, ExchangeType

from random import shuffle, randint
from copy import copy


class ProblemWrapper:
    def __init__(self, problem: Problem):
        self._problem = problem
        # precalc nearest-neighbourhood of each customer
        # and distance between each pair of customers
        # it will be useful for some heuristic optimizations
        customers = problem.customers
        self.c = {v.number: {w.number: v.c(w) for w in customers} for v in customers}
        customers = [v.number for v in problem.customers if v.number != problem.depot.number]
        self.nearest = {v: sorted(customers, key=lambda w: self.c[v][w]) for v in customers}

    def __getattr__(self, attr):
        if hasattr(self.__dict__['_problem'], attr):
            return getattr(self.__dict__['_problem'], attr)
        else:
            raise AttributeError(f"'ProblemWrapper' object has no attribute '{attr}'")

class RouteWrapper:
    def __init__(self, problem: ProblemWrapper, route: RouteList=None, meta_wrapper: 'MetaWrapper'=None):
        self._problem = problem
        self._route = route
        self._meta_wrapper = meta_wrapper
        if route:
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

    def insert(self, index, value: 'CustomerWrapper', update=False):
        assert value.ejected()
        i = index.node() if isinstance(index, CustomerWrapper) else index
        value._index = CustomerIndex(self, self._route.insert(i, value))
        if update:
            self._pc.update()
            self._dc.update()

    def eject(self, value: 'CustomerWrapper', update=False):
        assert value.route() == self
        self._route.remove(value._index._node)
        value._index = None
        if update:
            self._pc.update()
            self._dc.update()

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
    def __init__(self, **kwargs):
        if not kwargs:
            return
        if 'problem_wrapper' in kwargs:
            self.problem = kwargs['problem_wrapper']
        elif 'problem' in kwargs:
            self.problem = ProblemWrapper(kwargs['problem'])
        elif 'problem_wrapper' in kwargs:
            self.problem = kwargs['problem_wrapper']
        if self.problem:
            routes = kwargs.get('routes', None)
            if routes:
                not_ejected = set([v for v in sum(routes, [])])
                assert self.problem.depot not in not_ejected
                not_ejected.add(self.problem.depot)
                self._routes = []
                self._ejection_pool = [CustomerWrapper(v) for v in self.problem.customers if v not in not_ejected]

                for route in routes:
                    route_list = RouteList(self.problem, [CustomerWrapper(v) for v in route])
                    route_wrapper = RouteWrapper(self.problem, route_list, meta_wrapper=self)
                    self._routes.append(route_wrapper)
                assert self.valid()
            else:
                self._routes = []
                self._ejection_pool = []
                for v in self.problem.customers:
                    if v.number != self.problem.depot.number:
                        route_list = RouteList(self.problem, [CustomerWrapper(v)])
                        route_wrapper = RouteWrapper(self.problem, route_list, self)
                        self._routes.append(route_wrapper)
            # precalc nearest-neighbourhood of each customer
            # and distance between each pair of customers
            # it will be useful for some heuristic optimizations
            self._index = {v.number: v for route in self._routes for v in route}
            self._index.update({v.number: v for v in self._ejection_pool})
            self.nearest = {self._index[v]: [self._index[w] for w in nearest] for v, nearest in self.problem.nearest.items()}
        elif 'routes' in kwargs:
            raise ValueError("'problem' or 'problem_wrapper' must be passed")

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
        #result.c = self.c # const
        routes_copy = []
        for route in self._routes:
            route_copy = RouteWrapper(self.problem)
            route_copy._route = copy(route._route)
            route_copy._meta_wrapper = result
            route_copy._pc = copy(route._pc)
            route_copy._dc = copy(route._dc)
            route_copy._pc._route = route_copy
            route_copy._dc._route = route_copy
            for node in route_copy._route.head.iter():
                node.value._index = CustomerIndex(route_copy, node)
            routes_copy.append(route_copy)
        result._routes = routes_copy
        for v in result._ejection_pool:
            v._index = None
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
            customers = (v for v in route)
        elif v is not None:
            customers = [v]
        else:
            customers = (v for route in self._routes for v in route)

        delta_min = 10*10

        for v in customers:
            v_route = v.route()

            assert v_route is not None

            v_route_copy = copy(v_route)
            cust_dict = {v.number: v for v in v_route_copy}
            v_copy = cust_dict[v.number]

            v_eject_c_delta = v_route._pc.get_eject_delta(v, 1, 0)
            v_eject_tw_delta = v_route._pc.get_eject_delta(v, 0, 1)
            v_eject_dist_delta = v_route._dc.get_eject_delta(v)   
            v_route_copy.eject(v_copy, True)

            for w in self.nearest[v][:n_near]:
                if v is w or w.ejected():
                    continue
                if v.route() is not w.route(): # inter-route exchage
                    for e in exchanges(v, w):
                        if e.appliable():
                            delta_min = min(delta_min, e.penalty_delta(1, 1))
                            yield e
                else: # intra-route exchange
                    #if v is not w:
                    w_copy = cust_dict[w.number]
                    assert not w_copy.ejected()
                    # only Out-Relocate supported
                    c_delta = v_eject_c_delta + v_route_copy._pc.get_insert_delta(w_copy, v_copy, 1, 0)
                    tw_delta = v_eject_tw_delta + v_route_copy._pc.get_insert_delta(w_copy, v_copy, 0, 1)
                    dist_delta = v_eject_dist_delta + v_route_copy._dc.get_insert_delta(w_copy, v_copy)
                    e = ExchangeSlow(v, w, ExchangeType.OutRelocate, c_delta, tw_delta, dist_delta)
                    if e.appliable():
                        delta_min = min(delta_min, e.penalty_delta(1, 1))
                        yield e

                    c_delta = v_eject_c_delta + v_route_copy._pc.get_insert_delta(w_copy.next(), v_copy, 1, 0)
                    tw_delta = v_eject_tw_delta + v_route_copy._pc.get_insert_delta(w_copy.next(), v_copy, 0, 1)
                    dist_delta = v_eject_dist_delta + v_route_copy._dc.get_insert_delta(w_copy.next(), v_copy)
                    e = ExchangeSlow(v, w.next(), ExchangeType.OutRelocate, c_delta, tw_delta, dist_delta)
                    if e.appliable():
                        delta_min = min(delta_min, e.penalty_delta(1, 1))
                        yield e

                    e = ExchangeFast(v, w.prev(), ExchangeType.Exchange)
                    if e.appliable():
                        lower_bound, accurate = e.penalty_delta_lower_bound(0, 1)
                        if lower_bound < delta_min:
                            if accurate:
                                #print("success")
                                assert abs(lower_bound - e.penalty_delta(0, 1)) < 1e-4
                                yield ExchangeSlow(v, w.prev(), ExchangeType.Exchange, 0, lower_bound, 0)
                            else:
                                delta_min = min(delta_min, e.penalty_delta(1, 1))
                                yield ExchangeFast(v, w.prev(), ExchangeType.Exchange)

                    e = ExchangeFast(v, w.next(), ExchangeType.Exchange)
                    if e.appliable():
                        lower_bound, accurate = e.penalty_delta_lower_bound(0, 1)
                        if lower_bound < delta_min:
                            if accurate:
                                #print("success")
                                assert abs(lower_bound - e.penalty_delta(0, 1)) < 1e-4
                                yield ExchangeSlow(v, w.next(), ExchangeType.Exchange, 0, lower_bound, 0)
                            else:
                                delta_min = min(delta_min, e.penalty_delta(1, 1))
                                yield e

    # \mathcal{N}_near(v, \sigma)
    def N_random(self, v: CustomerWrapper = None, route: RouteWrapper = None):
        if route is not None:
            assert v is None
            customers = [v.value for v in route]
        elif v is not None:
            customers = [v]
        else:
            customers = [v for route in self._routes for v in route]
        shuffle(customers)

        #all_customers = [v for route in self._routes for v in route]

        for v in customers:
            v_route = v.route()

            assert v_route is not None

            v_route_copy = copy(v_route)
            cust_dict = {v.number: v for v in v_route_copy}
            v_copy = cust_dict[v.number]

            v_eject_c_delta = v_route._pc.get_eject_delta(v, 1, 0)
            v_eject_tw_delta = v_route._pc.get_eject_delta(v, 0, 1)
            v_eject_dist_delta = v_route._dc.get_eject_delta(v)   
            v_route_copy.eject(v_copy, True)

            #shuffle(all_customers)

            for w in self.nearest[v]:
                if v is w or w.ejected():
                    continue
                if v.route() is not w.route(): # inter-route exchage
                    exchanges_list = list(exchanges(v, w))
                    shuffle(exchanges_list)
                    for e in exchanges_list:
                        if e.appliable() and e.feasible():
                            return e
                else: # intra-route exchange
                    w_copy = cust_dict[w.number]
                    assert not w_copy.ejected()
                    
                    # only Out-Relocate supported
                    c_delta = v_eject_c_delta + v_route_copy._pc.get_insert_delta(w_copy, v_copy, 1, 0)
                    tw_delta = v_eject_tw_delta + v_route_copy._pc.get_insert_delta(w_copy, v_copy, 0, 1)
                    dist_delta = v_eject_dist_delta + v_route_copy._dc.get_insert_delta(w_copy, v_copy)
                    e = ExchangeSlow(v, w, ExchangeType.OutRelocate, c_delta, tw_delta, dist_delta)             
                    if e.appliable() and e.feasible():
                        return e

                    c_delta = v_eject_c_delta + v_route_copy._pc.get_insert_delta(w_copy.next(), v_copy, 1, 0)
                    tw_delta = v_eject_tw_delta + v_route_copy._pc.get_insert_delta(w_copy.next(), v_copy, 0, 1)
                    dist_delta = v_eject_dist_delta + v_route_copy._dc.get_insert_delta(w_copy.next(), v_copy)
                    e = ExchangeSlow(v, w.next(), ExchangeType.OutRelocate, c_delta, tw_delta, dist_delta)                
                    if e.appliable() and e.feasible():
                        return e
        return None


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