from eama.exchange import *
from eama.insertion_ejection import *
from eama.penalty_calculator import *
from eama.structure import *

from copy import deepcopy
from itertools import tee
from math import ceil, inf
from random import randint, sample, shuffle, choice
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count
from functools import reduce
import networkx as nx
import matplotlib.pyplot as plt

import os
import time
import sys

@dataclass
class RMHSettings:
    n_near: int = 100
    k_max: int = 5
    t_max: int = 60
    i_rand: int = 1000
    lower_bound: int = 0
    

@dataclass
class GIPSettings:
    n_pop: int = 100
    t_total: int = 600


@dataclass
class EAMASettings:
    n_ch: int = 10


class Graph:
    def __init__(self, routes=None):
        self.adjacency_table = {}
        if routes != None:
            problem = routes[0].route.problem
            self.num_vertices = len(problem.customers)
            for route in routes:
                for i, _ in enumerate(route.route._customers[:-1]):
                    u = route.route._customers[i].number
                    v = route.route._customers[i + 1].number
                    self.adjacency_table[u] = self.adjacency_table.get(u, set())
                    self.adjacency_table[v] = self.adjacency_table.get(v, set())
                    self.add_edge(u, v)

    def add_edge(self, u, v):
        self.adjacency_table[u].add(v)

    def remove_edge(self, u, v):
        self.adjacency_table[u].discard(v)

    def __xor__(self, other):
        result = Graph()
        nodes = set(self.adjacency_table.keys()).union(set(self.adjacency_table.keys()))
        result.adjacency_table = {u: self.adjacency_table.get(u, set()) ^ other.adjacency_table.get(u, set()) for u in nodes}
        return result

    def __ixor__(self, other):
        for u in self.adjacency_table.keys():
            if u in other.adjacency_table:
                self.adjacency_table[u] ^= other.adjacency_table[u]
        for u in other.adjacency_table.keys():
            if u not in self.adjacency_table:
                self.adjacency_table[u] = deepcopy(other.adjacency_table[u])
        return self
    
    def inverse(self):
        result = Graph()
        for u, edges in self.adjacency_table.items():
            result.adjacency_table[u] = result.adjacency_table.get(u, set())
            for v in edges:
                result.adjacency_table[v] = result.adjacency_table.get(v, set())               
                result.add_edge(v, u)
        return result

    def plot(self, filename, positions=None):
        G = nx.DiGraph()
        fig, ax = plt.subplots(figsize=(15,15), dpi=300)
        if positions:
            for number in self.adjacency_table.keys():
                G.add_node(number, pos=positions[number])
            pos = nx.get_node_attributes(G, 'pos')
        else:
            G.add_nodes_from(self.adjacency_table.keys())
            pos = nx.spring_layout(G)
        for u, edges in self.adjacency_table.items():
            for v in edges:
                G.add_edge(u, v)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10, arrows=True, connectionstyle='arc3', ax=ax)
        plt.savefig(filename, dpi=300)
        plt.close()


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

    def __init__(self, problem: Problem, rmh_settings: RMHSettings,
                gip_settings: GIPSettings, eama_settings: EAMASettings, obj_func=None, debug=False):
        self.problem = problem
        if not obj_func:
            obj_func = self.problem.obj_func
        self.obj_func = obj_func
        self.debug = debug
        self.p = [0] * (len(self.problem.customers) + 1)
        self.rmh_settings = rmh_settings
        self.gip_settings = gip_settings
        self.eama_settings = eama_settings

    def assert_zero_penalty(self, routes):
        assert sum([route.get_penalty(1, 1) for route in routes]) == 0

    # determine the minimum possible number of routes
    def powerful_route_minimization_heuristic(self, rmh_settings: RMHSettings):
        start_time = time.time()
        
        m = 0
        routes = []
        customers = []
        nearest = []
        alpha = 1
        beta = 1

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
                nearest[customer.number] = customers_sorted[:rmh_settings.n_near]

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
                            if time.time() - start_time > rmh_settings.t_max:
                                return False

                            if route.get_insert_penalty(position, v, 1, 1) == 0:
                                route.route._customers.insert(position, v)
                                route.recalc(route.route)
                                assert check_no_losses_without_v()
                                return True
                    assert check_no_losses_with_v()
                    return False

                def squeeze():
                    nonlocal start_time, routes, nearest, beta

                    routes_initial = deepcopy(routes)

                    # insert such that penalty is minimum
                    insertions = [(route, pos) for route in routes \
                        for pos in range(1, len(route.route._customers))]
                    route, pos = min(insertions, key=lambda p: p[0].get_insert_delta(p[1], v, alpha, beta))
                    route.route._customers.insert(pos, v)
                    route.recalc(route.route)

                    # try to reduce penalty
                    infeasible = []
                    feasible = []

                    def split_by_feasibility():
                        nonlocal infeasible, feasible

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
                        penalty_sum = sum([route.get_penalty(alpha, beta) for route in infeasible])
                        if self.debug:
                            print(f'penalty_sum: {penalty_sum}')
                        
                        if time.time() - start_time > rmh_settings.t_max:
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
                                    p_delta = exchange_penalty_delta(e, alpha, beta)
                                    if p_delta < opt_exchange_delta:
                                        opt_exchange_delta = p_delta
                                        opt_exchange = e
                        if self.debug:
                            print(f'delta: {opt_exchange_delta}')
                        if opt_exchange_delta > -1e-5:
                            # return to initial state
                            if sum([route.get_penalty(1, -1) for route in routes]) < 0:
                                beta /= 0.99
                            else:
                                beta *= 0.99
                            routes = routes_initial
                            return False
                        else:
                            # apply optimal exchange
                            assert check_no_losses_without_v()
                            target_penalty_delta = exchange_penalty_delta(opt_exchange, alpha, beta)
                            penalty_before = opt_exchange.v_route.get_penalty(alpha, beta) \
                                + opt_exchange.w_route.get_penalty(alpha, beta)
                            
                            apply_exchange(opt_exchange)

                            for route in [opt_exchange.v_route, opt_exchange.w_route]:
                                for pos, customer in enumerate(route.route._customers):
                                    if customer.number == self.problem.depot.number:
                                        continue
                                    ctr[customer.number] = (route, pos)

                            penalty_after = opt_exchange.v_route.get_penalty(alpha, beta) \
                                + opt_exchange.w_route.get_penalty(alpha, beta)
                            penalty_delta = penalty_after - penalty_before
                            if opt_exchange.v_route is opt_exchange.w_route:
                                penalty_delta /= 2
                            assert abs(penalty_delta - target_penalty_delta) < 1e-3

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
                            if time.time() - start_time > rmh_settings.t_max:
                                return False
                    
                            r = deepcopy(route)
                            r.route._customers.insert(insertion, v)
                            r.recalc(r.route)
                            
                            for ejection, a_quote, a, total_demand, p_sum in ejections_gen(r, self.p, rmh_settings.k_max):
                                assert check_ejection_metadata_is_valid(r, self.p, rmh_settings.k_max, ejection, a_quote, a, total_demand, p_sum)
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
                                                if exchange_penalty_delta(e, 1, 1) <= 0:
                                                    return e

                        for _ in range(rmh_settings.i_rand):
                            e = seek_for_feasible_exchange()
                            apply_exchange(e)
                    perturb()
            assert len(routes) == m
            return True

        prepare()
        
        solution = None
        lower_bound = max(self.rmh_settings.lower_bound, ceil(sum([c.demand for c in self.problem.customers]) / self.problem.vehicle_capacity))
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
        return solution

    def generate_initial_population(self):
        initial_population = [self.powerful_route_minimization_heuristic(self.rmh_settings)]
        rmh_settings = self.rmh_settings
        rmh_settings.t_max = 10**10
        rmh_settings.lower_bound = len(initial_population[0])
        executor = ProcessPoolExecutor(max_workers=cpu_count())
        futures = []
        n_pop = self.gip_settings.n_pop
        for _ in range(n_pop - 1):
            futures.append(executor.submit(self.powerful_route_minimization_heuristic, rmh_settings))
        wait(futures, timeout=self.gip_settings.t_total)
        executor.shutdown(wait=True, cancel_futures=True)
        initial_population += [future.result() for future in futures if future.done()]
        if self.debug:
            print(len(initial_population))        
        initial_population += [deepcopy(initial_population[-1]) for _ in range(n_pop - len(initial_population))]
        return initial_population

    # edge assembly crossover type
    def eax(self, p_a, p_b, strategy):
        problem = p_a[0].route.problem

        def split_into_cycles_dfs(graph, cycle, used, u):
            cycle.append(u)
            if u not in used:
                used.add(u)
                if len(graph.adjacency_table[u]) == 0:
                    return None
                v = choice(list(graph.adjacency_table[u]))
                return split_into_cycles_dfs(graph, cycle, used, v)
            else:
                cycle = cycle[cycle.index(cycle[-1]):]
                for u, v in zip(cycle[:-1], cycle[1:]):
                    graph.remove_edge(u, v)
                return cycle

        def split_into_routes_and_cycles(graph):
            graph = deepcopy(graph)
            nodes = list(graph.adjacency_table.keys())
            cycles = []
            while len(nodes) > 0:
                u = choice(list(nodes))
                if len(graph.adjacency_table[u]) == 0:
                    nodes.remove(u)
                    continue
                cycle = split_into_cycles_dfs(graph, [], set(), u)
                if not cycle:
                    return None
                cycles.append(cycle)
            routes, cycles = tee(cycles)
            routes = list(filter(lambda c: problem.depot.number in c, routes))
            cycles = list(filter(lambda c: problem.depot.number not in c, cycles))
            for i, route in enumerate(routes):
                ind = route.index(problem.depot.number)
                routes[i] = route[ind + 1:-1] + route[:ind]
                assert len(set(routes[i])) == len(routes[i])
                assert problem.depot.number not in routes[i]
            return routes, cycles

        def split_into_alternating_cycles_dfs(graph, graph_inv, nodes, edges, used, u, inv):
            prev = None if len(nodes) == 0 else nodes[-1][0]
            nodes.append((u, inv))
            if (u, inv) not in used:
                used.add((u, inv))
                if len(graph.adjacency_table[u]) == 0:
                    return None
                adjacency = list(graph.adjacency_table[u])
                if prev is not None:
                    adjacency.remove(prev)
                v = choice(list(adjacency))
                edges.append((v, u) if inv else (u, v))
                return split_into_alternating_cycles_dfs(graph_inv, graph, nodes, edges, used, v, not inv)
            else:
                cycle = Graph()
                ind = nodes.index((u, inv))
                if inv:
                    graph, graph_inv = graph_inv, graph
                for u, v in edges[ind:]:
                    assert v in graph.adjacency_table[u]
                    assert u in graph_inv.adjacency_table[v]
                    graph.remove_edge(u, v)
                    graph_inv.remove_edge(v, u)
                    cycle.adjacency_table[u] = cycle.adjacency_table.get(u, set())
                    cycle.adjacency_table[v] = cycle.adjacency_table.get(v, set())
                    cycle.add_edge(u, v)
                return cycle

        def split_into_alternating_cycles(graph):
            graph = deepcopy(graph)
            graph_inv = deepcopy(graph.inverse())
            nodes = list(graph.adjacency_table.keys())
            cycles = []
            inv = False
            while len(nodes) > 0:
                u = choice(list(nodes))
                if len(graph.adjacency_table[u]) == 0:
                    if len(graph_inv.adjacency_table[u]) > 0:
                        graph, graph_inv = graph_inv, graph
                        inv = not inv
                    else:
                        nodes.remove(u)
                        continue
                cycle = split_into_alternating_cycles_dfs(graph, graph_inv, [], [], set(), u, inv)
                if not cycle:
                    return None
                cycles.append(cycle)
            return cycles
        
        cycles = split_into_alternating_cycles(Graph(p_a)^Graph(p_b))
        assert cycles is not None
        E_set = []
        if strategy == 'single':
            E_set.append(choice(cycles))
        elif strategy == 'block':
            center = choice(cycles)
            center_nodes = set(center.adjacency_table.keys())
            for cycle in cycles:
                if center_nodes & set(cycle.adjacency_table.keys()):
                    E_set.append(cycle)
        intermediate_solution = reduce(lambda x, y: x^y, E_set, Graph(p_a))
        routes, cycles = split_into_routes_and_cycles(intermediate_solution)
        customers = {c.number: c for c in problem.customers}
        for i, route in enumerate(routes):
            r = Route(problem, [customers[number] for number in route])
            routes[i] = PenaltyCalculator()
            routes[i].recalc(r)
        for i, cycle in enumerate(cycles):
            cycles[i] = [customers[number] for number in cycle]

        print(len(cycles))

        #merge cycles with routes
        shuffle(cycles)
        for cycle in cycles:
            opt = None
            opt_delta = inf
            for r_ind, route in enumerate(routes):
                for i, _ in enumerate(cycle[:-1]):
                    for j, _ in enumerate(route.route._customers[:-1]):
                        r = route.route._customers[1:j + 1] + cycle[i + 1:-1] + cycle[:i + 1] + route.route._customers[j + 1:-1]
                        r = Route(problem, r)
                        new_route = PenaltyCalculator()
                        new_route.recalc(r)
                        delta = new_route.get_penalty(1.0, 1.0) - route.get_penalty(1.0, 1.0)
                        if delta < opt_delta:
                            opt = (r_ind, new_route)
                            opt_delta = delta
            routes[opt[0]] = opt[1]
            assert len(set(opt[1].route._customers[:-1])) == len(opt[1].route._customers[:-1])
        return routes


    # restore feasibility
    def repair(self, solution: Solution):
        pass

    # improve efficiency
    def local_search(self):
        pass

    def execute(self):
        population = self.generate_initial_population()
        n_pop = self.gip_settings.n_pop
        n_ch = self.eama_settings.n_ch
        while True:
            for i, _, in enumerate(population):
                p_a = deepcopy(population[i])
                p_b = population[(i + 1) % n_pop]
                for _ in range(n_ch):
                    g = self.eax(Graph(p_a), Graph(p_b))
                    sigma = self.repair(sigma)
                    sigma = self.local_search(sigma)

        pass