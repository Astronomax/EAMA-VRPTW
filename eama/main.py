from eama.exchange import *
from eama.insertion_ejection import *
from eama.penalty_calculator import *
from eama.structure import *

from collections.abc import Iterable
from copy import deepcopy
from itertools import tee
from math import ceil, inf
from operator import add, sub
from random import randint, sample, shuffle

import os
import time


class EAMA:
    exchanges = [
        # remove (v^-, v) and (w, w^+), and add (w, v) and (v^-, w^+)
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos - 1, w_route, w_pos, ExchangeType.TwoOpt),
        # remove (v, v^+) and (w^-, w), and add (v, w) and (w^-, v^+)
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos, w_route, w_pos - 1, ExchangeType.TwoOpt),
        # insert v between w^- and w, and link v^- and v^+
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos, w_route, w_pos, ExchangeType.OutRelocate),
        # insert v between w and w^+, and link v^- and v^+
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos, w_route, w_pos + 1, ExchangeType.OutRelocate),
        # insert v between (w^-)^- and w, and insert w^- between v^- and v^+
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos, w_route, w_pos - 1, ExchangeType.Exchange),
        # insert v between w and (w^+)^+, and insert w^+ between v^- and v^+
        lambda v_route, v_pos, w_route, w_pos:                                      \
            Exchange(v_route, v_pos, w_route, w_pos + 1, ExchangeType.Exchange),
    ]

    def __init__(self, problem: Problem, obj_func=None, n_near=100, debug=False, k_max=5, t_max=600, i_rand=1000):
        self.problem = problem
        if not obj_func:
            obj_func = self.problem.obj_func
        self.obj_func = obj_func
        self.n_near = n_near
        self.debug = debug
        self.k_max = k_max
        self.p = [0] * (len(self.problem.customers) + 1)
        self.t_max = t_max
        self.i_rand = i_rand
        self.n_pop = 100

    def assert_zero_penalty(self, routes):
        assert sum([sum(route.get_penalty()) for route in routes]) == 0

    # determine the minimum possible number of routes
    def powerful_route_minimization_heuristic(self, lower_bound=None):
        start_time = time.time()
        
        m = 0
        routes = []
        customers = []
        nearest = []

        # prepare some useful data
        def prepare():
            nonlocal m
            nonlocal routes
            nonlocal customers
            nonlocal nearest

            # list of all customers
            customers = self.problem.customers
            is_not_depo = lambda x: x.number != self.problem.depot.number
            customers = list(filter(is_not_depo, customers))

            m = len(customers)
            routes = [PenaltyCalculator() for _ in range(m)]
            for i, route in enumerate(routes):
                route.recalc(Route(self.problem, [customers[i]]))

            # list of distances to n_near-nearest customers
            nearest = [[]] * (len(customers) + 1)
            # we will have to copy because "sort" will affect
            # the for loop iterating order
            customers_sorted = customers.copy()
            for customer in customers:
                distance = lambda to_customer: customer.c(to_customer)
                customers_sorted = sorted(customers_sorted, key=distance)
                nearest[customer.number] = customers_sorted[:self.n_near]

        def delete_route():
            nonlocal routes
            nonlocal start_time
            nonlocal nearest

            m = len(routes)
            ejection_pool = []

            def eliminate_random_route():
                nonlocal m
                i = randint(0, m - 1)
                ejection_pool.extend(routes.pop(i).route.customers())
                m = m - 1

            eliminate_random_route()
            # trying to empty ejection_pool
            while len(ejection_pool) > 0:

                def check_no_losses_without_v():
                    return len(set([cust.number for cust in ejection_pool] + [customer.number for route in routes for customer in route.route._customers])) \
                        == len(self.problem.customers)
                    
                assert check_no_losses_without_v()
                
                if self.debug:
                    print(f'ejection_pool: {len(ejection_pool)}')
                # remove v from EP with the LIFO strategy
                v = ejection_pool.pop(len(ejection_pool) - 1)

                def check_no_losses_with_v():
                    return len(set([cust.number for cust in ejection_pool] + [v.number] + [customer.number for route in routes for customer in route.route._customers])) \
                        == len(self.problem.customers)
                    
                assert check_no_losses_with_v()

                def try_to_insert_somewhere():
                    nonlocal routes

                    self.assert_zero_penalty(routes)
                    assert check_no_losses_with_v()
                    #shuffle(routes)

                    for route in routes:
                        positions = list(range(1, len(route.route._customers)))
                        #shuffle(positions)
                        # search for feasible insertion position
                        for position in positions:
                            if time.time() - start_time > self.t_max:
                                return False

                            if sum(route.get_insert_penalty(position, v)) == 0:
                                route.route._customers.insert(position, v)
                                route.recalc(route.route)
                                assert check_no_losses_without_v()
                                return True
                    assert check_no_losses_with_v()
                    return False

                def squeeze():
                    nonlocal start_time
                    nonlocal routes
                    nonlocal nearest

                    routes_initial = deepcopy(routes)

                    # insert such that penalty is minimum
                    insertions =                                            \
                        [(route, pos)                                       \
                        for route in routes                                 \
                        for pos in range(1, len(route.route._customers))]
                    route, pos = min(insertions, key=lambda p: sum(p[0].get_insert_delta(p[1], v))) # alpha = beta = 1
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

                    ctr = [None] * len(self.problem.customers)
                    for route in routes:
                        for pos, customer in enumerate(route.route._customers):
                            if customer.number == self.problem.depot.number:
                                continue
                            ctr[customer.number] = (route, pos)

                    while len(infeasible) > 0:
                        penalty_sum = sum([sum(route.get_penalty()) for route in infeasible])
                        if self.debug:
                            print(f'penalty_sum: {penalty_sum}')
                        
                        if time.time() - start_time > self.t_max:
                            routes = routes_initial
                            return False

                        v_route = infeasible.pop(randint(0, len(infeasible) - 1))
                        opt_exchange = None
                        opt_exchange_delta = inf
                        for v_pos in range(0, len(v_route.route._customers)):
                            vv = v_route.route._customers[v_pos]
                            if vv.number == self.problem.depot.number:
                                continue
                            for w in nearest[vv.number]:
                                if w.number in [customer.number for customer in ejection_pool]:
                                    continue
                                w_route, w_pos = ctr[w.number]
                                assert w_route.route._customers[w_pos].number == w.number

                                for j, exchange_gen_f in enumerate(self.exchanges):
                                    e = exchange_gen_f(v_route, v_pos, w_route, w_pos)
                                    if not exchange_appliable(e):
                                        continue
                                    p_c_delta, p_tw_delta = exchange_penalty_delta(e)
                                    if p_c_delta + p_tw_delta < opt_exchange_delta: # alpha = beta = 1
                                        opt_exchange_delta = p_c_delta + p_tw_delta # alpha = beta = 1
                                        opt_exchange = e
                        if self.debug:
                            print(f'delta: {opt_exchange_delta}')
                        if opt_exchange_delta > -1e-5:
                            # return to initial state
                            routes = routes_initial
                            return False
                        else:
                            # apply optimal exchange
                            assert check_no_losses_without_v()
                            target_c_delta, target_pw_delta = exchange_penalty_delta(opt_exchange)
                            prev = map(add, opt_exchange.v_route.get_penalty(), opt_exchange.w_route.get_penalty())
                            apply_exchange(opt_exchange)

                            for route in [opt_exchange.v_route, opt_exchange.w_route]:
                                for pos, customer in enumerate(route.route._customers):
                                    if customer.number == self.problem.depot.number:
                                        continue
                                    ctr[customer.number] = (route, pos)

                            cur = map(add, opt_exchange.v_route.get_penalty(), opt_exchange.w_route.get_penalty())
                            c_delta, pw_delta = map(sub, cur, prev)
                            if opt_exchange.v_route is opt_exchange.w_route:
                                pw_delta /= 2
                                c_delta /= 2
                            assert abs(pw_delta - target_pw_delta) < 1e-3
                            assert abs(c_delta - target_c_delta) < 1e-3
                            split_by_feasibility()
                            assert check_no_losses_without_v()
                    self.assert_zero_penalty(routes)
                    assert check_no_losses_without_v()
                    return True

                assert check_no_losses_with_v() 

                if try_to_insert_somewhere():
                    if self.debug:
                        print("insert")
                    self.assert_zero_penalty(routes)
                    assert check_no_losses_without_v() 
                    continue
                elif squeeze():
                    if self.debug:
                        print("squeeze")
                    self.assert_zero_penalty(routes)
                    assert check_no_losses_without_v()
                    continue
                else:
                    if self.debug:
                        print("insertion-ejection")
                    self.p[v.number] += 1
                    # finding the best insertion-ejection combination
                    opt_insertion_ejection = None
                    opt_insertion_ejection_psum = inf

                    for route_ind, route in enumerate(routes):
                        for insertion in range(1, len(route.route._customers)):
                            if time.time() - start_time > self.t_max:
                                return False
                    
                            r = deepcopy(route)
                            r.route._customers.insert(insertion, v)
                            r.recalc(r.route)
                            
                            for ejection, a_quote, a, total_demand, p_sum in ejections_gen(r, self.p, self.k_max):
                                print(f'{route_ind}/{len(routes)}: {ejection}')
                                assert check_ejection_metadata_is_valid(r, self.p, self.k_max, ejection, a_quote, a, total_demand, p_sum)
                                j = ejection[-1] + 1

                                if a_quote[j] <= r.route._customers[j].l and \
                                    a[j] <= r.z[j] and \
                                    r.tw_sf[j] == 0 and \
                                    total_demand <= self.problem.vehicle_capacity:

                                    # constrains satisfied
                                    if p_sum < opt_insertion_ejection_psum or \
                                        p_sum == opt_insertion_ejection_psum and \
                                        len(ejection) < len(opt_insertion_ejection.ejection):

                                        opt_insertion_ejection = InsertionEjection(route, v, insertion, deepcopy(ejection))
                                        opt_insertion_ejection_psum = p_sum

                    if opt_insertion_ejection == None:
                        return False
                    def apply_insertion_ejection(ie):
                        assert check_no_losses_with_v()
                        r = ie.route
                        r.route._customers.insert(ie.insertion, ie.v)
                        for i, pos in enumerate(ie.ejection):
                            ejection_pool.append(ie.route.route._customers.pop(pos - i))
                        ie.route.recalc(ie.route.route)
                        self.assert_zero_penalty(routes)
                        assert check_no_losses_without_v()

                    self.assert_zero_penalty(routes)
                    assert check_no_losses_with_v()

                    apply_insertion_ejection(opt_insertion_ejection)

                    def perturb():
                        def seek_for_feasible_exchange():
                            for v_route in sample(routes, len(routes)):
                                v_customers = v_route.route._customers
                                for v_pos, _ in enumerate(sample(v_customers, len(v_customers))):
                                    for w_route in sample(routes, len(routes)):
                                        w_customers = w_route.route._customers
                                        for w_pos, _ in enumerate(sample(w_customers, len(w_customers))):
                                            for exchange_gen_f in self.exchanges:
                                                e = exchange_gen_f(v_route, v_pos, w_route, w_pos)
                                                if not exchange_appliable(e):
                                                    continue
                                                p_c_delta, p_tw_delta = exchange_penalty_delta(e)
                                                if p_c_delta + p_tw_delta <= 0:
                                                    return e

                        for _ in range(self.i_rand):
                            e = seek_for_feasible_exchange()
                            apply_exchange(e)
                    perturb()
            assert len(routes) == m
            return True

        prepare()
        
        solution = None
        if lower_bound:
            lower_bound = min(lower_bound, ceil(sum([c.demand for c in self.problem.customers]) / self.problem.vehicle_capacity))
        else:
            lower_bound = ceil(sum([c.demand for c in self.problem.customers]) / self.problem.vehicle_capacity)
        # try to reduce number of routes
        while m > lower_bound:
            if self.debug:
                print(f'routes number: {m}')
            if not delete_route():
                break
            solution = deepcopy(routes)
            m = m - 1
            self.assert_zero_penalty(routes)
            assert len(set([customer.number for route in routes for customer in route.route._customers])) \
                        == len(self.problem.customers)
        return [route.route for route in solution]

    def generate_initial_population(self):
        initial_population = [None] * self.n_pop
        initial_population[0] = self.eval_min_routes_number()
        for i, _ in enumerate(initial_population[1:]):
            initial_population[i] = self.eval_min_routes_number(len(initial_population[0]))
        return initial_population

    # edge assembly crossover type
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