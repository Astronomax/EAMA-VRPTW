import random
from itertools import tee, accumulate
from enum import Enum
from operator import add, sub
from eama.structure import Customer, Route, Problem, Solution
import math
import copy


class ExchangeType(Enum):
    TwoOpt = 1,
    OutRelocate = 2,
    Exchange = 3


class Exchange:
    def __init__(self, v_route, v_pos, w_route, w_pos, operator):
      self.v_route = v_route
      self.v_pos = v_pos
      self.w_route = w_route
      self.w_pos = w_pos
      self.operator = operator


def apply_exchange(e: Exchange):
    if e.operator == ExchangeType.TwoOpt:
        if e.v_route is e.w_route:
            return
        v_pf = e.v_route.route._customers[:e.v_pos + 1]
        v_sf = e.v_route.route._customers[e.v_pos + 1:]
        w_pf = e.w_route.route._customers[:e.w_pos + 1]
        w_sf = e.w_route.route._customers[e.w_pos + 1:]
        e.v_route.route._customers = v_pf + w_sf
        e.w_route.route._customers = w_pf + v_sf
    elif e.operator == ExchangeType.OutRelocate:
        w_pos = e.w_pos
        if e.v_route is e.w_route and e.v_pos < e.w_pos:
            w_pos = e.w_pos - 1
        v = e.v_route.route._customers.pop(e.v_pos)
        e.w_route.route._customers.insert(w_pos, v)
    elif e.operator == ExchangeType.Exchange:
        e.v_route.route._customers[e.v_pos], e.w_route.route._customers[e.w_pos] = \
        e.w_route.route._customers[e.w_pos], e.v_route.route._customers[e.v_pos]
    e.v_route.recalc(e.v_route.route)
    e.w_route.recalc(e.w_route.route)


def exchange_appliable(e: Exchange):
    n_v = len(e.v_route.route._customers)
    n_w = len(e.w_route.route._customers)
    if e.v_pos < 0 or e.w_pos < 0:
        return False
    if e.operator == ExchangeType.TwoOpt:
        if e.v_route is e.w_route:
            return False
        elif e.v_pos >= n_v - 1 or e.w_pos >= n_w - 1:
            return False
        elif e.v_pos == n_v - 2 and e.w_pos == 0:
            return False
        elif e.w_pos == n_w - 2 and e.v_pos == 0:
            return False
    elif e.operator == ExchangeType.OutRelocate:
        if e.v_pos <= 0 or e.v_pos >= n_v - 1:
            return False
        elif e.v_route is e.w_route:
            w_pos = e.w_pos
            if e.v_pos < e.w_pos:
                w_pos = e.w_pos - 1
            if w_pos < 1 or w_pos >= n_v - 1:
                return False
        else:
            if e.w_pos < 1 or e.w_pos >= n_w:
                return False
    elif e.operator == ExchangeType.Exchange:
        if e.v_pos <= 0 or e.v_pos >= n_v - 1:
            return False
        elif e.w_pos <= 0 or e.w_pos >= n_w - 1:
            return False
    return True


def exchange_penalty_delta(e: Exchange):
    if e.operator == ExchangeType.TwoOpt:
        return e.v_route.two_opt_penalty_delta(e.v_pos, e.w_route, e.w_pos)
    elif e.operator == ExchangeType.OutRelocate:
        return e.v_route.out_relocate_penalty_delta(e.v_pos, e.w_route, e.w_pos)
    elif e.operator == ExchangeType.Exchange:
        return e.v_route.exchange_penalty_delta(e.v_pos, e.w_route, e.w_pos)


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

    def get_penalty(self):
        p_c = max(self.demand_pf[-1] - self.route.problem.vehicle_capacity, 0)
        p_tw = self.tw_pf[-1]
        return p_c, p_tw

    # get penalty after insertion
    def get_insert_penalty(self, pos, v: Customer):
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
        return p_c, p_tw

    # get penalty after replacement
    def get_replace_penalty(self, pos, v: Customer):
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
        return p_c, p_tw

    # get penalty after insertion
    def get_extract_penalty(self, pos):
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
        return p_c, p_tw

    def get_insert_delta(self, pos, v: Customer):
        return map(sub, self.get_insert_penalty(pos, v), self.get_penalty())

    def get_extract_delta(self, pos):
        return map(sub, self.get_extract_penalty(pos), self.get_penalty())

    def one_opt_penalty(self, v_pos, w_route, w_pos):
        if self is w_route:
            return math.inf, math.inf
        p_c = max(self.demand_pf[v_pos] + w_route.demand_sf[w_pos + 1] \
                  - self.route.problem.vehicle_capacity, 0) 
        p_tw = self.tw_pf[v_pos] + w_route.tw_sf[w_pos + 1]
        v = self.route._customers[v_pos]
        w_next = w_route.route._customers[w_pos + 1]
        a_quote_v = self.a[v_pos] + v.s + v.c(w_next)
        p_tw += max(a_quote_v - w_route.z[w_pos + 1], 0)
        return p_c, p_tw

    # remove (v, v^+) and (w, w^+), and add (v, w^+) and (w, v^+)
    def two_opt_penalty_delta(self, v_pos, w_route, w_pos):
        if self is w_route:
            return math.inf, math.inf
        return map(sub, map(add, self.one_opt_penalty(v_pos, w_route, w_pos), \
            w_route.one_opt_penalty(w_pos, self, v_pos)), \
            map(add, self.get_penalty(), w_route.get_penalty()))

    def apply_self_penalty_delta(self, v_pos, w_pos, e: ExchangeType):   
        self_copy = copy.deepcopy(self)
        apply_exchange(Exchange(self_copy, v_pos, self_copy, w_pos, e))
        return map(sub, self_copy.get_penalty(), self.get_penalty())

    # insert v between w^- and w, and link v^- and v^+
    def out_relocate_penalty_delta(self, v_pos, w_route, w_pos):
        if self is w_route:
            return self.apply_self_penalty_delta(v_pos, w_pos, ExchangeType.OutRelocate)
        v = self.route._customers[v_pos]
        v_delta = self.get_extract_delta(v_pos)
        w_delta = w_route.get_insert_delta(w_pos, v)
        return map(add, v_delta, w_delta)

    # insert v between w^- and w^+,
    # and insert w between v^- and v^+
    def exchange_penalty_delta(self, v_pos, w_route, w_pos):
        if self is w_route:
            return self.apply_self_penalty_delta(v_pos, w_pos, ExchangeType.Exchange)
        v = self.route._customers[v_pos]
        w = w_route.route._customers[w_pos]
        v_delta = map(sub, self.get_replace_penalty(v_pos, w), self.get_penalty())
        w_delta = map(sub, w_route.get_replace_penalty(w_pos, v), w_route.get_penalty())
        return map(add, v_delta, w_delta)

    def is_feasible(self):
        return sum(self.get_penalty()) == 0


class EAMA:
    exchanges = [
        # remove (v^-, v) and (w, w^+), and add (w, v) and (v^-, w^+)
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos - 1, w_route, w_pos, ExchangeType.TwoOpt),
            #v_route.two_opt_penalty_delta(v_pos - 1, w_route, w_pos)),
        # remove (v, v^+) and (w^-, w), and add (v, w) and (w^-, v^+)
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos, w_route, w_pos - 1, ExchangeType.TwoOpt),
            #v_route.two_opt_penalty_delta(v_pos, w_route, w_pos - 1)),
        # insert v between w^- and w, and link v^- and v^+
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos, w_route, w_pos, ExchangeType.OutRelocate),
            #v_route.out_relocate_penalty_delta(v_pos, w_route, w_pos)),
        # insert v between w and w^+, and link v^- and v^+
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos, w_route, w_pos + 1, ExchangeType.OutRelocate),
            #v_route.out_relocate_penalty_delta(v_pos, w_route, w_pos + 1)),
        # insert v between (w^-)^- and w, and insert w^- between v^- and v^+
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos, w_route, w_pos - 1, ExchangeType.Exchange),
            #v_route.exchange_penalty_delta(v_pos, w_route, w_pos - 1)),
        # insert v between w and (w^+)^+, and insert w^+ between v^- and v^+
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos, w_route, w_pos + 1, ExchangeType.Exchange),
            #v_route.exchange_penalty_delta(v_pos, w_route, w_pos + 1)),
    ]

    def __init__(self, problem: Problem, obj_func=None, n_near=100, debug=False):
        self.problem = problem
        if not obj_func:
            obj_func = self.problem.obj_func
        self.obj_func = obj_func
        self.n_near = n_near
        self.debug = debug

    # determine the minimum possible number of routes
    def powerful_route_minimization_heuristic(self):
        customers = []
        '''
        distance_tresholds = []
        '''
        # prepare some useful data
        def prepare():
            nonlocal customers
            '''
            nonlocal distance_tresholds
            '''
            # list of all customers
            customers = self.problem.customers
            is_not_depo = lambda x: x.number != self.problem.depot.number
            customers = list(filter(is_not_depo, customers))
            '''
            # list of distances to n_near-nearest customers
            distance_tresholds = [0] * len(customers)
            # we will have to copy because "sort" will affect
            # the for loop iterating order
            customers_copy = customers.copy()
            for from_customer, i in enumerate(customers):
                distance = lambda to_customer: from_customer.c(to_customer)
                sort(customers_copy, key=distance)
                nth_nearest = customers_copy[self.n_near]
                distance_tresholds[i] = distance(nth_nearest)
            '''

        def delete_route(routes):
            m = len(routes)
            ejection_pool = []

            def eliminate_random_route():
                nonlocal m
                i = random.randint(0, m - 1)
                ejection_pool.extend(routes.pop(i).route.customers())
                m = m - 1

            eliminate_random_route()
            # trying to empty ejection_pool
            while len(ejection_pool) > 0:
                # remove v from EP with the LIFO strategy
                v = ejection_pool.pop(len(ejection_pool) - 1)

                def try_to_insert_somewhere():
                    nonlocal routes
                    random.shuffle(routes)
                    for route in routes:
                        positions = list(range(1, len(route.route._customers)))
                        random.shuffle(positions)
                        # search for feasible insertion position
                        for position in positions:
                            # TODO optimize (reduce recalc)
                            route.route._customers.insert(position, v)
                            route.recalc(route.route)
                            if route.is_feasible():
                                return True
                            route.route._customers.pop(position)
                            route.recalc(route.route)
                    return False

                def squeeze():
                    nonlocal routes
                    # insert such that penalty is minimum
                    insertions =                                            \
                        [(route, pos)                                       \
                        for route in routes                                 \
                        for pos in range(1, len(route.route._customers))]
                    route, pos = max(insertions, key=lambda p: sum(p[0].get_insert_delta(p[1], v))) # alpha = beta = 1
                    route.route._customers.insert(pos, v)
                    route.recalc(route.route)

                    # try to reduce penalty
                    infeasible = []
                    feasible = []

                    def split_by_feasibility():
                        nonlocal infeasible
                        nonlocal feasible
                        infeasible, feasible = tee(routes)
                        # routes from "routes" copied by reference into
                        # "infeasible" and "feasible". we use this for convenience
                        infeasible = list(filter(lambda r: not r.is_feasible(), infeasible))
                        feasible = list(filter(lambda r: r.is_feasible(), feasible))

                    split_by_feasibility()
                    while len(infeasible) > 0:
                        v_route = infeasible.pop(random.randint(0, len(infeasible) - 1))
                        opt_exchange = None
                        opt_exchange_delta = math.inf
                        for v_pos in range(0, len(v_route.route._customers)):
                            for i, w_route in enumerate(routes):
                                for w_pos in range(0, len(w_route.route._customers)):
                                    for j, exchange_gen_f in enumerate(self.exchanges):
                                        if self.debug:
                                                print(
                                                    (f'ejection_pool: {len(ejection_pool)},\n'
                                                    f'infeasible: {len(infeasible)},\n'
                                                    f'v_pos: {v_pos}/{len(v_route.route._customers)},\n'
                                                    f'w_route: {i}/{len(routes)},\n'
                                                    f'w_pos: {w_pos}/{len(w_route.route._customers)},\n'
                                                    f'exchange: {j}/{len(self.exchanges)}\n')
                                                )
                                        e = exchange_gen_f(v_route, v_pos, w_route, w_pos)
                                        if not exchange_appliable(e):
                                            continue
                                        p_c_delta, p_tw_delta = exchange_penalty_delta(e)
                                        if p_c_delta + p_tw_delta < opt_exchange_delta: # alpha = beta = 1
                                            opt_exchange_delta = p_c_delta + p_tw_delta # alpha = beta = 1
                                            opt_exchange = e
                        if opt_exchange_delta >= 0:
                            return False
                        else:
                            # apply optimal exchange
                            assert len(ejection_pool) + sum([len(route.route._customers) - 2 for route in routes]) \
                                == len(self.problem.customers) - 1
                            target_c_delta, target_pw_delta = exchange_penalty_delta(opt_exchange)
                            prev = map(add, opt_exchange.v_route.get_penalty(), opt_exchange.w_route.get_penalty())
                            apply_exchange(opt_exchange)
                            cur = map(add, opt_exchange.v_route.get_penalty(), opt_exchange.w_route.get_penalty())
                            c_delta, pw_delta = map(sub, cur, prev)
                            if opt_exchange.v_route is opt_exchange.w_route:
                                pw_delta /= 2
                                c_delta /= 2
                            assert abs(pw_delta - target_pw_delta) < 1e-3
                            assert abs(c_delta - target_c_delta) < 1e-3
                            split_by_feasibility()
                            assert len(ejection_pool) + sum([len(route.route._customers) - 2 for route in routes]) \
                                == len(self.problem.customers) - 1
                    return True

                if try_to_insert_somewhere():
                    continue
                elif squeeze():
                    continue
                else:
                    # NOT IMPLEMENTED
                    return False
            return True

        prepare()
        m = len(customers)
        routes = [PenaltyCalculator() for _ in range(m)]
        for i, route in enumerate(routes):
            route.recalc(Route(self.problem, [customers[i]]))
        
        solution = None
        # try to reduce number of routes
        while m > 1:
            if not delete_route(routes):
                return m, [route.route for route in solution]
            solution = copy.deepcopy(routes)
            m = m - 1
            if self.debug:
                print(m)
        return 1, [route.route for route in solution]

    def generate_initial_population(self):
        m = self.eval_min_routes_number()
        pass

    # edge assembly crossover operator
    def eax(self):
        pass

    # restore feasibility
    def repair(self, solution: Solution):
        pass

    # improve efficiency
    def local_search(self):
        pass

    def execute(self):
        pass
