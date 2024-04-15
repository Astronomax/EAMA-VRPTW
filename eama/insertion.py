from eama.meta_wrapper import MetaWrapper, RouteWrapper, CustomerWrapper
from eama.modification import Modification


class Insertion(Modification):
    def __init__(self, index: 'CustomerWrapper', customer: 'CustomerWrapper'):
        self._index = index
        self._customer = customer

    def appliable(self):
        try:
            if self._index.node().head():
                raise Exception()
            if self._index is self._customer:
                raise Exception()
            if not self._customer.ejected():
                raise Exception()
        except Exception as _:
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
        routes = [route]
    else:
        assert meta_wrapper is not None
        routes = meta_wrapper._routes
    for route in routes:
        for w in route._route.head.iter():
            e = Insertion(w.value, v)
            if e.appliable() and (not feasible_only or e.feasible()):
                yield e

def feasible_insertions(v: CustomerWrapper, meta_wrapper: MetaWrapper = None, route: RouteWrapper = None):
    return insertions(v, meta_wrapper, route, True)
