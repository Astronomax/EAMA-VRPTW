from eama.exchange import *
from eama.structure import *

from copy import deepcopy
from itertools import accumulate
from operator import add, sub

class PenaltyCalculator:
    def recalc(self, route):
        # time windows penalty precalc
        self.route = route
        n = len(self.route._customers)
        # calculate a, a_quote, tw_pf
        self.tw_pf = [0] * n
        self.a = [0] * n
        self.a_quote = [0] * n
        self.a[0] = self.a_quote[0] = route.problem.depot.e
        for i in range(1, n):
            prev = route._customers[i - 1]
            next = route._customers[i]
            self.a_quote[i] = self.a[i - 1] + prev.s + prev.c(next)
            self.a[i] = min(max(self.a_quote[i], next.e), next.l)
            self.tw_pf[i] = max(self.a_quote[i] - next.l, 0)
        self.tw_pf = list(accumulate(self.tw_pf))
        # calculate z, z_quote, tw_sf
        self.tw_sf = [0] * n
        self.z = [0] * n
        self.z_quote = [0] * n
        self.z[n - 1] = self.z_quote[n - 1] = route.problem.depot.l
        for i in reversed(range(1, n)):
            prev = route._customers[i - 1]
            next = route._customers[i]
            self.z_quote[i - 1] = self.z[i] - prev.s - prev.c(next)
            self.z[i - 1] = min(max(self.z_quote[i - 1], prev.e), prev.l)
            self.tw_sf[i - 1] = max(prev.e - self.z_quote[i - 1], 0)
        self.tw_sf = list(accumulate(self.tw_sf[::-1]))[::-1]
        # capacity penalty precalc
        self.demand_pf = list(accumulate(route._customers, \
                              lambda pf, c: pf + c.demand, initial=0))[1:]
        self.demand_sf = list(accumulate(route._customers[::-1], \
                              lambda pf, c: pf + c.demand, initial=0))[::-1][:-1]

    def get_penalty(self, alpha, beta):
        p_c = max(self.demand_pf[-1] - self.route.problem.vehicle_capacity, 0)
        p_tw = self.tw_pf[-1]
        return alpha * p_c + beta * p_tw

    # get penalty after insertion
    def get_insert_penalty(self, pos, v: Customer, alpha, beta):
        p_c = max(self.demand_pf[-1] + v.demand - self.route.problem.vehicle_capacity, 0)
        x_pos = pos - 1
        y_pos = pos
        p_tw = self.tw_pf[x_pos] + self.tw_sf[y_pos]
        x = self.route._customers[x_pos]
        y = self.route._customers[y_pos]
        a_quote_v = self.a[x_pos] + x.s + x.c(v)
        z_quote_v = self.z[y_pos] - v.s - v.c(y)
        p_tw += max(a_quote_v - v.l, 0)
        p_tw += max(v.e - z_quote_v, 0)
        a_v = min(max(a_quote_v, v.e), v.l)
        z_v = min(max(z_quote_v, v.e), v.l)
        p_tw += max(a_v - z_v, 0)
        return alpha * p_c + beta * p_tw

    # get penalty after replacement
    def get_replace_penalty(self, pos, v: Customer, alpha, beta):
        p_c = max(self.demand_pf[-1] - self.route._customers[pos].demand \
                  + v.demand - self.route.problem.vehicle_capacity, 0)
        x_pos = pos - 1
        y_pos = pos + 1
        p_tw = self.tw_pf[x_pos] + self.tw_sf[y_pos]
        x = self.route._customers[x_pos]
        y = self.route._customers[y_pos]
        a_quote_v = self.a[x_pos] + x.s + x.c(v)
        z_quote_v = self.z[y_pos] - v.s - v.c(y)
        p_tw += max(a_quote_v - v.l, 0)
        p_tw += max(v.e - z_quote_v, 0)
        a_v = min(max(a_quote_v, v.e), v.l)
        z_v = min(max(z_quote_v, v.e), v.l)
        p_tw += max(a_v - z_v, 0)
        return alpha * p_c + beta * p_tw

    # get penalty after insertion
    def get_extract_penalty(self, pos, alpha, beta):
        p_c = max(self.demand_pf[-1] - self.route._customers[pos].demand \
                  - self.route.problem.vehicle_capacity, 0)
        x_pos = pos - 1
        v_pos = pos + 1
        p_tw = self.tw_pf[x_pos] + self.tw_sf[v_pos]
        x = self.route._customers[x_pos]
        v = self.route._customers[v_pos]
        a_quote_v = self.a[x_pos] + x.s + x.c(v)
        a_v = min(max(a_quote_v, v.e), v.l)
        #p_tw += max(a_quote_v - self.z[v_pos], 0)
        p_tw += max(a_quote_v - v.l, 0)
        p_tw += max(a_v - self.z[v_pos], 0)
        return alpha * p_c + beta * p_tw

    def get_insert_delta(self, pos, v: Customer, alpha, beta):
        return self.get_insert_penalty(pos, v, alpha, beta) - self.get_penalty(alpha, beta)

    def get_extract_delta(self, pos, alpha, beta):
        return self.get_extract_penalty(pos, alpha, beta) - self.get_penalty(alpha, beta)

    def one_opt_penalty(self, v_pos, w_route, w_pos, alpha, beta):
        if self is w_route:
            return math.inf, math.inf
        p_c = max(self.demand_pf[v_pos] + w_route.demand_sf[w_pos + 1] \
                  - self.route.problem.vehicle_capacity, 0) 
        p_tw = self.tw_pf[v_pos] + w_route.tw_sf[w_pos + 1]
        v = self.route._customers[v_pos]
        w_next = w_route.route._customers[w_pos + 1]
        a_quote_v = self.a[v_pos] + v.s + v.c(w_next)
        p_tw += max(a_quote_v - w_route.z[w_pos + 1], 0)
        return alpha * p_c + beta * p_tw

    # remove (v, v^+) and (w, w^+), and add (v, w^+) and (w, v^+)
    def two_opt_penalty_delta(self, v_pos, w_route, w_pos, alpha, beta):
        if self is w_route:
            return math.inf
        return (self.one_opt_penalty(v_pos, w_route, w_pos, alpha, beta) - self.get_penalty(alpha, beta)) + \
            (w_route.one_opt_penalty(w_pos, self, v_pos, alpha, beta) - w_route.get_penalty(alpha, beta))

    def apply_self_penalty_delta(self, v_pos, w_pos, e: ExchangeType, alpha, beta):   
        self_copy = deepcopy(self)
        apply_exchange(Exchange(self_copy, v_pos, self_copy, w_pos, e))
        return self_copy.get_penalty(alpha, beta) + self.get_penalty(alpha, beta)

    # insert v between w^- and w, and link v^- and v^+
    def out_relocate_penalty_delta(self, v_pos, w_route, w_pos, alpha, beta):
        if self is w_route:
            return self.apply_self_penalty_delta(v_pos, w_pos, ExchangeType.OutRelocate, alpha, beta)
        v = self.route._customers[v_pos]
        v_delta = self.get_extract_delta(v_pos, alpha, beta)
        w_delta = w_route.get_insert_delta(w_pos, v, alpha, beta)
        return v_delta + w_delta

    # insert v between w^- and w^+,
    # and insert w between v^- and v^+
    def exchange_penalty_delta(self, v_pos, w_route, w_pos, alpha, beta):
        if self is w_route:
            return self.apply_self_penalty_delta(v_pos, w_pos, ExchangeType.Exchange, alpha, beta)
        v = self.route._customers[v_pos]
        w = w_route.route._customers[w_pos]
        v_delta = self.get_replace_penalty(v_pos, w, alpha, beta) - self.get_penalty(alpha, beta)
        w_delta = w_route.get_replace_penalty(w_pos, v, alpha, beta) - w_route.get_penalty(alpha, beta)
        return v_delta + w_delta

    def is_feasible(self):
        return self.get_penalty(1, 1) == 0