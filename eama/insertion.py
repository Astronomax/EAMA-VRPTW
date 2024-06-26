from eama.meta_wrapper import MetaWrapper, RouteWrapper, CustomerWrapper
from eama.modification import Modification


class Insertion(Modification):
    def __init__(self, index: 'CustomerWrapper', customer: 'CustomerWrapper'):
        self._index = index
        self._customer = customer

    def appliable(self):
        if self._index.ejected():
            return False
        if not self._customer.ejected():
            return False
        if self._index.node().head():
            return False
        if self._index is self._customer:
            return False
        return True  

    def penalty(self, alpha, beta):
        pass

    def penalty_delta(self, alpha, beta):
        from eama.penalty_calculator import PenaltyCalculator
        return PenaltyCalculator.get_insert_delta(self._index, self._customer, alpha, beta)
    
    def distance(self):
        pass

    def distance_delta(self):
        from eama.distance_calculator import DistanceCalculator
        return DistanceCalculator.get_insert_delta(self._index, self._customer)

    def apply(self):
        self._index.route().insert(self._index, self._customer, True)

    def feasible(self):
        return (self.penalty_delta(1, 1) == 0)
    
    def __str__(self):
        return f'index: {self._index.number}, v: {self._customer.number}, penalty_delta = {self.penalty_delta(1, 1)}'

def insertions(v: CustomerWrapper, meta_wrapper: MetaWrapper = None, route: RouteWrapper = None, feasible_only=False):
    if route is not None:
        assert meta_wrapper is None
        if feasible_only and route._pc.demand_pf[-1] + v.demand > route._problem.vehicle_capacity:
            return
        indexes = (w.value for w in route._route.head.iter())
    else:
        assert meta_wrapper is not None
        indexes = (w.value for route in meta_wrapper._routes for w in route._route.head.iter())
    for index in indexes:
        e = Insertion(index, v)
        if e.appliable() and ((not feasible_only) or e.feasible()):
            yield e

def nearest_insertions(v: CustomerWrapper, meta_wrapper: MetaWrapper = None, route: RouteWrapper = None, feasible_only=False):
    #'''
    if route is not None:
        assert meta_wrapper is None
        #if feasible_only and route._pc.demand_pf[-1] + v.demand > route._problem.vehicle_capacity:
        #    return
        for w in route._route.head.iter():
            e = Insertion(w.value, v)
            if e.appliable() and (not feasible_only or e.feasible()):
                yield e
    else:
        assert meta_wrapper is not None
        for w in meta_wrapper.nearest[v]:
            #if not w.ejected():
            #    print(w.number, [u.number for u in w.route()])
            e = Insertion(w, v)
            if e.appliable() and (not feasible_only or e.feasible()):
                yield e
        for route in meta_wrapper._routes:
            assert route._route.tail.value.route() is route
            #print(route._route.tail.value.number, [u.number for u in route])
            e = Insertion(route._route.tail.value, v)
            #print(e.penalty_delta(0, 1), e.penalty_delta(1, 0))
            assert route._route.tail.value.number == 0
            if e.appliable() and (not feasible_only or e.feasible()):
                yield e
    #'''
    '''
    if route is not None:
        assert meta_wrapper is None
        routes = [route]
    else:
        assert meta_wrapper is not None
        routes = meta_wrapper._routes
    for route in routes:
        if feasible_only and route._pc.demand_pf[-1] + v.demand > route._problem.vehicle_capacity:
            continue
        for w in route._route.head.iter():
            e = Insertion(w.value, v)
            if e.appliable() and (not feasible_only or e.feasible()):
                yield e
    '''
def feasible_insertions(v: CustomerWrapper, meta_wrapper: MetaWrapper = None, route: RouteWrapper = None):
    return nearest_insertions(v, meta_wrapper, route, True)
