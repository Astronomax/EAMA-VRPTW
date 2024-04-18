from eama.exchange import *
from eama.structure import *

from itertools import accumulate


class DistanceCalculator:
    def __init__(self, route: 'RouteWrapper'=None):
        self._route = route
        
        # we should be able to get the node index in O(1) time
        if route is not None:
            self._problem = route._problem
            self.update()

    def update(self):
        self._index = {v.value: i for i, v in enumerate(self._route._route.head.iter())} # hash(v) = v.number
        route = [node.value for node in self._route._route.head.iter()]
        n = len(route)
        distances = [u.c(v) for u, v in zip(route[1:], route[:-1])]
        self._dist_pf = list(accumulate(distances, initial=0))
        self._dist_sf = list(accumulate(distances[::-1], initial=0))[::-1]

    def __copy__(self):
        result = DistanceCalculator()
        result._problem = self._problem
        result._dist_pf = self._dist_pf.copy()
        result._dist_sf = self._dist_sf.copy()
        return result

    def get_distance(self):
        return self._dist_pf[-1]

    # get distance after insertion
    @staticmethod
    def get_insert_distance(index: 'CustomerWrapper', w: 'CustomerWrapper'):
        dc = index.dc()
        pos = dc._index[index]
        return dc._dist_pf[pos - 1] + index.prev().c(w) + w.c(index) + dc._dist_sf[pos]

    # get distance after insertion
    @staticmethod
    def get_eject_distance(index: 'CustomerWrapper'):
        dc = index.dc()
        pos = dc._index[index]
        return dc._dist_pf[pos - 1] + index.prev().c(index.next()) + dc._dist_sf[pos + 1]
    
    # get distance after replacement
    @staticmethod
    def get_replace_distance(index: 'CustomerWrapper', w: 'CustomerWrapper'):
        dc = index.dc()
        pos = dc._index[index]
        return dc._dist_pf[pos - 1] + index.prev().c(w) + w.c(index.next()) + dc._dist_sf[pos + 1]
    
    @staticmethod
    def get_insert_delta(index: 'CustomerWrapper', v: 'CustomerWrapper'):
        return index.prev().c(v) + v.c(index) - index.prev().c(index)

    @staticmethod
    def get_eject_delta(v: 'CustomerWrapper'):
        return v.prev().c(v.next()) - (v.prev().c(v) + v.c(v.next()))
    
    @staticmethod
    def get_replace_delta(index: 'CustomerWrapper', w: 'CustomerWrapper'):
        return (index.prev().c(w) + w.c(index.next())) - (index.prev().c(index) + index.c(index.next()))
    
    @staticmethod
    def one_opt_distance(v: 'CustomerWrapper', w: 'CustomerWrapper'):
        v_pos = v.dc()._index[v]
        w_pos = w.dc()._index[w]
        return v.dc()._dist_pf[v_pos] + v.c(w.next()) + w.dc()._dist_sf[w_pos + 1]
    
    # remove (v, v^+) and (w, w^+), and add (v, w^+) and (w, v^+)
    @staticmethod
    def two_opt_distance_delta(v: 'CustomerWrapper', w: 'CustomerWrapper'):
        assert v.route() is not w.route()
        return  (DistanceCalculator.one_opt_distance(v, w) - v.dc().get_distance()) + \
                (DistanceCalculator.one_opt_distance(w, v) - w.dc().get_distance())

    # insert v between w^- and w, and link v^- and v^+
    @staticmethod
    def out_relocate_distance_delta(v: 'CustomerWrapper', w: 'CustomerWrapper'):
        assert v is not w
        if v.next() is w:
            return 0
        v_delta = DistanceCalculator.get_eject_delta(v)
        w_delta = DistanceCalculator.get_insert_delta(w, v)
        return v_delta + w_delta

    # insert v between w^- and w^+, and insert w between v^- and v^+
    @staticmethod
    def exchange_distance_delta(v: 'CustomerWrapper', w: 'CustomerWrapper'):
        if v.next() is w:
            return v.prev().c(w) + v.c(w.next()) - (v.prev().c(v) + w.c(w.next()))
        elif w.next() is v:
            return w.prev().c(v) + w.c(v.next()) - (w.prev().c(w) + v.c(v.next()))
        v_delta = DistanceCalculator.get_replace_delta(v, w)
        w_delta = DistanceCalculator.get_replace_delta(w, v)
        return v_delta + w_delta
