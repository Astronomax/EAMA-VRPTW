from eama.exchange import *
from eama.structure import *

from copy import copy
from itertools import accumulate


class PenaltyCalculator:
    def __init__(self, route: 'RouteWrapper' = None):
        self._route = route
        # we should be able to get the node index in O(1) time
        if route is not None:
            self._problem = route._problem
            #self._meta_wrapper = route.meta_wrapper
            self.update()

    def update(self):
        self._index = {v.value: i for i, v in enumerate(self._route._route.head.iter())} # hash(v) = v.number
        self.route = [node.value for node in self._route._route.head.iter()]
        # time windows penalty precalc
        n = len(self.route)
        # calculate a, a_quote, tw_pf
        self.tw_pf = [0] * n
        self.a = [0] * n
        self.a[0] = self._problem.depot.e
        for i in range(1, n):
            prev = self.route[i - 1]
            next = self.route[i]
            a_quote = self.a[i - 1] + prev.s + prev.c(next)#self._problem.c[prev.number][next.number]#prev.c(next)
            self.a[i] = min(max(a_quote, next.e), next.l)
            self.tw_pf[i] = max(a_quote - next.l, 0)
        self.tw_pf = list(accumulate(self.tw_pf))
        # calculate z, z_quote, tw_sf
        self.tw_sf = [0] * n
        self.z = [0] * n
        self.z[n - 1] = self._problem.depot.l
        for i in reversed(range(n - 1)):
            prev = self.route[i]
            next = self.route[i + 1]
            z_quote = self.z[i + 1] - prev.s - prev.c(next)#self._problem.c[prev.number][next.number]#prev.c(next)
            self.z[i] = min(max(z_quote, prev.e), prev.l)
            self.tw_sf[i] = max(prev.e - z_quote, 0)
        self.tw_sf = list(accumulate(self.tw_sf[::-1]))[::-1]
        # capacity penalty precalc
        self.demand_pf = list(accumulate(self.route, lambda pf, c: pf + c.demand, initial=0))[1:]
        self.demand_sf = list(accumulate(self.route[::-1], lambda pf, c: pf + c.demand, initial=0))[::-1][:-1]
        
        assert abs(self.tw_pf[-1] - self.tw_sf[0]) < 1e-4

    def __copy__(self):
        result = PenaltyCalculator()
        result._problem = self._problem
        result._index = self._index.copy()
        result.route = self.route.copy()
        result.a = self.a.copy()
        result.tw_pf = self.tw_pf.copy()
        result.z = self.z.copy()
        result.tw_sf = self.tw_sf.copy()
        result.demand_pf = self.demand_pf.copy()
        result.demand_sf = self.demand_sf.copy()
        return result

    def get_penalty(self, alpha, beta):
        p_c = max(self.demand_pf[-1] - self._problem.vehicle_capacity, 0)
        p_tw = self.tw_pf[-1]
        return alpha * p_c + beta * p_tw

    def is_feasible(self):
        return self.get_penalty(1, 1) == 0

    # get penalty after insertion
    @staticmethod
    def get_insert_penalty(index: 'CustomerWrapper', w: 'CustomerWrapper', alpha, beta):
        pc = index.pc()
        pos = pc._index[index] # hash(index) = index.number
        p_c = max(pc.demand_pf[-1] + w.demand - pc._problem.vehicle_capacity, 0)
        p_tw = pc.tw_pf[pos - 1] + pc.tw_sf[pos]
        x = index.prev()
        y = index
        a_quote_v = pc.a[pos - 1] + x.s + x.c(w)#pc._problem.c[x.number][w.number]#x.c(w)
        z_quote_v = pc.z[pos] - w.s - w.c(y)#pc._problem.c[w.number][y.number]#w.c(y)
        p_tw += max(a_quote_v - w.l, 0)
        p_tw += max(w.e - z_quote_v, 0)
        a_v = min(max(a_quote_v, w.e), w.l)
        z_v = min(max(z_quote_v, w.e), w.l)
        p_tw += max(a_v - z_v, 0)
        return alpha * p_c + beta * p_tw

    # get penalty after replacement
    @staticmethod
    def get_replace_penalty(index: 'CustomerWrapper', w: 'CustomerWrapper', alpha, beta):
        pc = index.pc()
        pos = pc._index[index] # hash(index) = index.number
        p_c = max(0, pc.demand_pf[-1] - index.demand + w.demand - pc._problem.vehicle_capacity)
        p_tw = pc.tw_pf[pos - 1] + pc.tw_sf[pos + 1]
        x = index.prev()
        y = index.next()
        a_quote_v = pc.a[pos - 1] + x.s + x.c(w)#pc._problem.c[x.number][w.number]#x.c(w)
        z_quote_v = pc.z[pos + 1] - w.s - w.c(y)#pc._problem.c[w.number][y.number]#w.c(y)
        p_tw += max(a_quote_v - w.l, 0)
        p_tw += max(w.e - z_quote_v, 0)
        a_v = min(max(a_quote_v, w.e), w.l)
        z_v = min(max(z_quote_v, w.e), w.l)
        p_tw += max(a_v - z_v, 0)
        return alpha * p_c + beta * p_tw

    # get penalty after insertion
    @staticmethod
    def get_eject_penalty(index: 'CustomerWrapper', alpha, beta):
        pc = index.pc()
        pos = pc._index[index] # hash(index) = index.number
        p_c = max(0, pc.demand_pf[-1] - index.demand - pc._problem.vehicle_capacity)
        p_tw = pc.tw_pf[pos - 1] + pc.tw_sf[pos + 1]
        x = index.prev()
        v = index.next()
        a_quote_v = pc.a[pos - 1] + x.s + x.c(v)#pc._problem.c[x.number][v.number]#x.c(v)
        a_v = min(max(a_quote_v, v.e), v.l)
        p_tw += max(a_quote_v - v.l, 0)
        p_tw += max(a_v - pc.z[pos + 1], 0)
        return alpha * p_c + beta * p_tw

    @staticmethod
    def get_insert_delta(index: 'CustomerWrapper', v: 'CustomerWrapper', alpha, beta):
        return PenaltyCalculator.get_insert_penalty(index, v, alpha, beta) - index.pc().get_penalty(alpha, beta)

    @staticmethod
    def get_eject_delta(v: 'CustomerWrapper', alpha, beta):
        return PenaltyCalculator.get_eject_penalty(v, alpha, beta) - v.pc().get_penalty(alpha, beta)

    @staticmethod
    def one_opt_penalty(v: 'CustomerWrapper', w: 'CustomerWrapper', alpha, beta):
        assert v.route() is not w.route()
        v_pos = v.pc()._index[v]
        w_pos = w.pc()._index[w]
        p_c = max(0, v.pc().demand_pf[v_pos] + w.pc().demand_sf[w_pos + 1] - v.pc()._problem.vehicle_capacity) 
        p_tw = v.pc().tw_pf[v_pos] + w.pc().tw_sf[w_pos + 1]
        w_next = w.next()
        a_quote_v = v.pc().a[v_pos] + v._customer.s + v.c(w_next)#v.pc()._problem.c[v.number][w_next.number]#v.c(w_next)
        p_tw += max(a_quote_v - w.pc().z[w_pos + 1], 0)
        return alpha * p_c + beta * p_tw

    @staticmethod
    def apply_self_penalty_delta(e: 'Exchange', alpha, beta):
        v_pos = e._v.pc()._index[e._v]
        w_pos = e._w.pc()._index[e._w]
        v_route = copy(e._v.route())
        assert e._v.number == v_route._route.get_node(v_pos).value.number
        assert e._w.number == v_route._route.get_node(w_pos).value.number
        penalty_before_exchange = v_route._pc.get_penalty(alpha, beta)
        Exchange(v_route._route.get_node(v_pos).value, v_route._route.get_node(w_pos).value, e._type).apply()
        return v_route._pc.get_penalty(alpha, beta) - penalty_before_exchange
        
    # remove (v, v^+) and (w, w^+), and add (v, w^+) and (w, v^+)
    @staticmethod
    def two_opt_penalty_delta(v: 'CustomerWrapper', w: 'CustomerWrapper', alpha, beta):
        assert v.route() is not w.route()
        return  (PenaltyCalculator.one_opt_penalty(v, w, alpha, beta) - v.pc().get_penalty(alpha, beta)) + \
                (PenaltyCalculator.one_opt_penalty(w, v, alpha, beta) - w.pc().get_penalty(alpha, beta))

    # insert v between w^- and w, and link v^- and v^+
    @staticmethod
    def out_relocate_penalty_delta(v: 'CustomerWrapper', w: 'CustomerWrapper', alpha, beta):
        if v.route() is w.route():
            return PenaltyCalculator.apply_self_penalty_delta(Exchange(v, w, ExchangeType.OutRelocate), alpha, beta)
        v_delta = PenaltyCalculator.get_eject_delta(v, alpha, beta)
        w_delta = PenaltyCalculator.get_insert_delta(w, v, alpha, beta)
        return v_delta + w_delta

    # insert v between w^- and w^+, and insert w between v^- and v^+
    @staticmethod
    def exchange_penalty_delta(v: 'CustomerWrapper', w: 'CustomerWrapper', alpha, beta):
        if v.route() is w.route():
            return PenaltyCalculator.apply_self_penalty_delta(Exchange(v, w, ExchangeType.Exchange), alpha, beta)
        v_delta = PenaltyCalculator.get_replace_penalty(v, w, alpha, beta) - v.pc().get_penalty(alpha, beta)
        w_delta = PenaltyCalculator.get_replace_penalty(w, v, alpha, beta) - w.pc().get_penalty(alpha, beta)
        return v_delta + w_delta
