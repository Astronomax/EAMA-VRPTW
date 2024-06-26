from eama.exchange import *
from eama.penalty_calculator import *
from eama.structure import *
from eama.meta_wrapper import MetaWrapper, RouteWrapper, CustomerWrapper
from eama.insertion import insertions, feasible_insertions
from eama.ejection import feasible_ejections, Ejection

from itertools import tee, combinations
from math import ceil, inf
from random import randint, sample, shuffle, choice
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count

import time
import inspect


class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'


class BreakLoop(Exception):
    pass


@dataclass
class RMHSettings:
    n_near: int = 100
    k_max: int = 5
    t_max: int = 60
    i_rand: int = 10
    lower_bound: int = 0


@dataclass
class RMHState:
    alpha: float = 1
    beta: float = 1


@dataclass
class GIPSettings:
    n_pop: int = 100
    t_total: int = 600


@dataclass
class EAMASettings:
    n_ch: int = 10


class EAMA:
    def __init__(self, problem: Problem, rmh_settings: RMHSettings,
                gip_settings: GIPSettings, eama_settings: EAMASettings, debug=False):
        self.problem = problem
        self.debug = debug
        self.p = [0] * (len(self.problem.customers) + 1)
        self.rmh_settings = rmh_settings
        self.gip_settings = gip_settings
        self.eama_settings = eama_settings

    def _debug_print(self, msg, color):
        current_function = inspect.stack()[1].function
        formatted_string = f"{Colors.BLUE}'{current_function}': {color}{msg}{Colors.RESET}"
        print(formatted_string)

    def _split_by_feasibility(self, meta_wrapper: MetaWrapper):
        feasible, infeasible = [], []
        infeasible, feasible = tee(meta_wrapper._routes)
        # routes from "routes" copied by reference into
        # "infeasible" and "feasible". we use this for convenience
        infeasible = list(filter(lambda r: not r.feasible(), infeasible))
        feasible = list(filter(lambda r: r.feasible(), feasible))
        return feasible, infeasible

    def _insert_feasible(self, meta_wrapper: MetaWrapper, v: CustomerWrapper, deadline: float = inf):
        '''
        for route in meta_wrapper._routes:
            tail = None
            for node in route._route.head.iter():
                if node.tail():
                    tail = node
                assert node.value.route() is route
            assert tail is not None
            assert tail.value.number == 0
            assert route._route.tail is tail
        '''
        assert v.ejected()
        if self.debug:
            self._debug_print("started", Colors.RESET)
            #print("'insert_feasible': started")
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid(v)
        #'''
        for insertion in feasible_insertions(v, meta_wrapper=meta_wrapper):
            insertion.apply()
            assert meta_wrapper.feasible()
            if self.debug:
                self._debug_print("completed successfully", Colors.GREEN)
                #print(Colors.GREEN + "'insert_feasible': completed successfully" + Colors.RESET)
            return True
        if self.debug:
            self._debug_print("failed", Colors.RED)
            #print(Colors.RED + "'insert_feasible': failed" + Colors.RESET)
        return False
        #'''        
        '''
        try:
            insertions_list = list(feasible_insertions(v, meta_wrapper=meta_wrapper))
            #print([(insertion.penalty_delta(1, 1), insertion._index.number) for insertion in insertions_list[:6]])
            insertion = choice(insertions_list)
            insertion.apply()
            assert meta_wrapper.feasible()
            if self.debug:
                self._debug_print("completed successfully", Colors.GREEN)
                #print(Colors.GREEN + "'insert_feasible': completed successfully" + Colors.RESET)
            return True
        except IndexError:
            if self.debug:
                self._debug_print("failed", Colors.RED)
                #print(Colors.RED + "'insert_feasible': failed" + Colors.RESET)
            return False
        '''

    def _squeeze(self, meta_wrapper: MetaWrapper, v: CustomerWrapper, state: RMHState, settings: RMHSettings, deadline: float = inf):
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid(v)
        if self.debug:
            self._debug_print("started", Colors.RESET)
            #print("'squeeze': started")
        meta_wrapper = meta_wrapper.inherit()

        # insert such that penalty is minimum
        insertions_list = list(insertions(v, meta_wrapper=meta_wrapper))
        #print([(insertion.penalty_delta(state.alpha, state.beta), insertion._index.number) for insertion in insertions_list[:6]])
        #print(len(insertions_list))
        insertion=min(insertions_list,\
            key=lambda e: e.penalty_delta(state.alpha, state.beta))
        #route = insertion._index.route()
        #print([node.value.number for node in route._route.head.iter()])
        insertion.apply()
        #print([node.value.number for node in route._route.head.iter()])
        _, infeasible = self._split_by_feasibility(meta_wrapper)
        assert len(infeasible) > 0

        while len(infeasible) > 0:
            if self.debug:
                self._debug_print(f"penalty_sum: {sum([r.get_penalty(state.alpha, state.beta) for r in infeasible])}", Colors.RESET)
                #print(f"'squeeze': penalty_sum: {sum([r.get_penalty(state.alpha, state.beta) for r in infeasible])}")
            v_route = infeasible.pop(randint(0, len(infeasible) - 1))
            opt_exchange = None
            opt_exchange_delta = inf
            v_route_penalty = v_route.get_penalty(state.alpha, state.beta)
            try:
                for v in v_route:
                    for exchange in meta_wrapper.N_near(settings.n_near, v=v):
                        if time.time() > deadline:
                            raise TimeoutError()
                        exchange_delta = exchange.penalty_delta(state.alpha, state.beta)
                        if exchange_delta < opt_exchange_delta:
                            opt_exchange = exchange
                            opt_exchange_delta = exchange_delta
                            if exchange_delta <= -v_route_penalty:# + 1e-5:
                                raise BreakLoop
            except BreakLoop:
                pass

            if self.debug:
                self._debug_print(f"opt exchange delta: {opt_exchange_delta}", Colors.RESET)
                #print(f"'squeeze': opt exchange delta: {opt_exchange_delta}")
            if not opt_exchange or opt_exchange_delta > -1e-5:
                #if meta_wrapper.get_penalty(1, -1) < 0: # p_c < p_tw
                #    state.beta /= 0.99
                #else:
                #    state.beta *= 0.99
                if self.debug:
                    self._debug_print("failed", Colors.RED)
                    #print(Colors.RED + "'squeeze': failed" + Colors.RESET)
                    self._debug_print(f"beta after correction: {state.beta}", Colors.RESET)
                    #print(f"'squeeze': beta after correction: {state.beta}")
                return False, None
            #if opt_exchange._v.route() is opt_exchange._w.route():
            #    print(opt_exchange._type)
            assert opt_exchange_delta == opt_exchange.penalty_delta(state.alpha, state.beta)
            if self.debug:
                c_delta = opt_exchange.penalty_delta(1, 0)
                tw_delta = opt_exchange.penalty_delta(0, 1)
                c_before_exchange = meta_wrapper.get_penalty(1, 0)
                tw_before_exchange = meta_wrapper.get_penalty(0, 1)
            #print(opt_exchange._type)
            '''
            for route in v_route._meta_wrapper._routes:
                assert route._route.tail.value.route() is route
            '''
            opt_exchange.apply()
            '''
            for route in v_route._meta_wrapper._routes:
                assert route._route.tail.value.route() is route
            '''
            if self.debug:
                assert abs(c_delta - (meta_wrapper.get_penalty(1, 0) - c_before_exchange)) < 1e-4
                assert abs(tw_delta - (meta_wrapper.get_penalty(0, 1) - tw_before_exchange)) < 1e-4
            _, infeasible = self._split_by_feasibility(meta_wrapper)
        if self.debug:
            self._debug_print("completed successfully", Colors.GREEN)
            #print(Colors.GREEN + "'squeeze': completed successfully" + Colors.RESET)
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid()
        '''
        for route in v_route._meta_wrapper._routes:
            assert route._route.tail.value.route() is route
        '''
        return True, meta_wrapper

    def _insert_eject(self, meta_wrapper: MetaWrapper, v: CustomerWrapper, settings: RMHSettings, deadline: float = inf):
        #print(v.number)
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid(v)
        assert v.ejected()
        if self.debug:
            self._debug_print("started", Colors.RESET)
            self._debug_print(f"p[v_in] = {self.p[v.number] + 1}", Colors.RESET)
            #print("'insert_eject': started")
        self.p[v.number] += 1
        opt_insertion_ejection = None
        p_best = inf
        #for route in sample(meta_wrapper._routes, len(meta_wrapper._routes)):
        try:
            for ind, route in enumerate(meta_wrapper._routes):
                assert v.ejected()
                insertions_list = list(insertions(v, route=route))
                for insertion in sample(insertions_list, len(insertions_list)):
                    if time.time() > deadline:
                        raise TimeoutError()
                    assert v.ejected()
                    insertion.apply()
                    assert not v.ejected()
                    assert not insertion._index.route().feasible()



                    '''
                    def subsets_lexicographic_order(nums, k):
                        nums.sort()
                        n = len(nums)
                        for r in range(1, k + 1):
                            for combo in combinations(nums, r):
                                yield list(combo)



                    solution = meta_wrapper
                    n = len(route)
                    all_ejections = sorted(subsets_lexicographic_order(list(range(n)), 5))
                    ejections = list(feasible_ejections(route, self.p, 5, p_best))
                    j = 0
                    p_best_tmp = p_best
                    for ejection_list in all_ejections:
                        solution_copy = solution.inherit()
                        route_copy = solution_copy._routes[ind]
                        ejection = [route_copy[i] for i in ejection_list]
                        Ejection(solution_copy, ejection, 0, 0, 0).apply()
                        p_sum = sum([self.p[v.number] for v in ejection])
                        if route_copy.feasible():
                            if p_sum < p_best_tmp:
                                if j == len(ejections):
                                    for node in route._route.head.iter():
                                        print(node.value._customer)
                                    print(ejection_list)
                                assert j < len(ejections)
                                assert all([x == y for x, y in zip(ejections[j][0]._ejection, ejection)])
                                assert ejections[j][1] == p_sum
                                p_best_tmp = p_sum
                                j += 1
                        solution.activate()
                    assert j == len(ejections)
                    '''




                    for ejection, p_sum in feasible_ejections(route, self.p, settings.k_max, p_best):
                        #print(p_sum, p_best)
                        assert p_sum <= p_best
                        #print(p_sum)
                        #if p_sum < p_best or len(ejection._ejection) < len(opt_insertion_ejection[1]._ejection):
                        opt_insertion_ejection = (insertion, copy(ejection))
                            #if p_sum == 0 and len(ejection._ejection) == 1:
                            #    raise BreakLoop
                        p_best = p_sum
                    route.eject(v, True)
                    assert v.ejected()
        except BreakLoop:
            pass
        if self.debug:
            self._debug_print(f"opt insertion-ejection p_sum: {p_best}", Colors.RESET)
            #print(f"'insert_eject': opt insertion-ejection p_sum: {p_best}")
        if opt_insertion_ejection == None:
            if self.debug:
                self._debug_print("failed", Colors.RED)
                #print(Colors.RED + "'insert_eject': failed" + Colors.RESET)
            return False
        insertion, ejection = opt_insertion_ejection
        #print(len(ejection._ejection))
        #print([w.number for w in ejection._ejection])
        route = insertion._index.route()
        insertion.apply()
        assert not route.feasible()
        #ejection._ejection = sorted(ejection._ejection, key=lambda x: self.p[x.number])
        ejection.apply()
        assert route.feasible()
        assert meta_wrapper.feasible()
        if self.debug:
            self._debug_print("completed successfully", Colors.GREEN)
            #print(Colors.GREEN + "'insert_eject': completed successfully" + Colors.RESET)
        return True 

    def _perturb(self, meta_wrapper: MetaWrapper, i_rand: int, n_near: int, deadline: float = inf):
        assert meta_wrapper.feasible()
        if self.debug:
            self._debug_print("started", Colors.RESET)
            #print("'perturb': started")
        '''
        cnt = 0
        for _ in range(i_rand):
            if time.time() > deadline:
                raise TimeoutError()
            #e = meta_wrapper.N_random()
            feasible_exchange = None
            try:
                v_route = meta_wrapper._routes[randint(0, len(meta_wrapper._routes) - 1)]
                w_route = meta_wrapper._routes[randint(0, len(meta_wrapper._routes) - 1)]
                if v_route is w_route:
                    continue
                    for v in sample(list(v_route), len(v_route)):
                        v_route_copy = copy(v_route)
                        cust_dict = {v.number: v for v in v_route_copy}
                        v_copy = cust_dict[v.number]
                        v_eject_c_delta = v_route._pc.get_eject_delta(v, 1, 0)
                        v_eject_tw_delta = v_route._pc.get_eject_delta(v, 0, 1)
                        v_eject_dist_delta = v_route._dc.get_eject_delta(v)   
                        v_route_copy.eject(v_copy, True)
                        for w in sorted(list(w_route), key=lambda u: v.c(u)):
                            if v is w:
                                continue
                            w_copy = cust_dict[w.number]
                            assert not w_copy.ejected()
                            # only Out-Relocate supported
                            c_delta = v_eject_c_delta + v_route_copy._pc.get_insert_delta(w_copy, v_copy, 1, 0)
                            tw_delta = v_eject_tw_delta + v_route_copy._pc.get_insert_delta(w_copy, v_copy, 0, 1)
                            dist_delta = v_eject_dist_delta + v_route_copy._dc.get_insert_delta(w_copy, v_copy)
                            e = ExchangeSlow(v, w, ExchangeType.OutRelocate, c_delta, tw_delta, dist_delta)             
                            if e.appliable() and e.feasible():
                                feasible_exchange = e
                                raise BreakLoop
                            c_delta = v_eject_c_delta + v_route_copy._pc.get_insert_delta(w_copy.next(), v_copy, 1, 0)
                            tw_delta = v_eject_tw_delta + v_route_copy._pc.get_insert_delta(w_copy.next(), v_copy, 0, 1)
                            dist_delta = v_eject_dist_delta + v_route_copy._dc.get_insert_delta(w_copy.next(), v_copy)
                            e = ExchangeSlow(v, w.next(), ExchangeType.OutRelocate, c_delta, tw_delta, dist_delta)                
                            if e.appliable() and e.feasible():
                                feasible_exchange = e
                                raise BreakLoop
                else:
                    for v in sample(list(v_route), len(v_route)):
                        #print(len(w_route))
                        for w in sorted(list(w_route), key=lambda u: v.c(u)):
                            exchanges_list = list(exchanges(v, w))
                            shuffle(exchanges_list)
                            for e in exchanges_list:
                                if e.appliable() and e.feasible():
                                    feasible_exchange = e
                                    raise BreakLoop
            except BreakLoop:
                pass
            if feasible_exchange:
                cnt += 1
                assert feasible_exchange.appliable() and feasible_exchange.feasible()
                feasible_exchange.apply()
                assert meta_wrapper.feasible()
        print(cnt)
        '''
        #'''
        for _ in range(i_rand):
            if time.time() > deadline:
                raise TimeoutError()
            e = meta_wrapper.N_random()
            if e:
                assert e.appliable() and e.feasible()
                e.apply()
                assert meta_wrapper.feasible()
            else:
                break
        #'''
        if self.debug:
            self._debug_print("completed successfully", Colors.GREEN)
            #print(Colors.GREEN + "'perturb': completed successfully" + Colors.RESET)

    def _delete_route(self, meta_wrapper: MetaWrapper, state: RMHState, settings: RMHSettings, deadline: float = inf):
        if self.debug:
            self._debug_print("started", Colors.RESET)
            #print("'delete_route': started")
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid()
        '''
        for route in meta_wrapper._routes:
            assert route._route.tail.value.route() is route
        '''
        '''
        for route in meta_wrapper._routes:
            tail = None
            for node in route._route.head.iter():
                if node.tail():
                    tail = node
                assert node.value.route() is route
            assert tail is not None
            assert tail.value.number == 0
            assert route._route.tail is tail
        '''
        meta_wrapper = meta_wrapper.inherit()
        '''
        for route in meta_wrapper._routes:
            assert route._route.tail.value.route() is route
        '''
        '''
        for route in meta_wrapper._routes:
            tail = None
            for node in route._route.head.iter():
                if node.tail():
                    tail = node
                assert node.value.route() is route
            assert tail is not None
            assert tail.value.number == 0
            assert route._route.tail is tail
        '''
        # eliminate_random_route
        i = randint(0, len(meta_wrapper._routes) - 1)
        ejected = list(meta_wrapper._routes.pop(i))
        for v in ejected:
            v._index = None
        meta_wrapper._ejection_pool.extend(ejected)
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid()
        '''
        for route in meta_wrapper._routes:
            assert route._route.tail.value.route() is route
        '''
        # trying to empty ejection_pool
        while len(meta_wrapper._ejection_pool) > 0:
            if time.time() > deadline:
                raise TimeoutError()
            assert meta_wrapper.feasible()
            assert meta_wrapper.valid()
            if self.debug:
                self._debug_print(f"ejection_pool: {len(meta_wrapper._ejection_pool)}", Colors.RESET)
                #print(f"'delete_route': ejection_pool: {len(meta_wrapper._ejection_pool)}")
            # remove v from EP with the LIFO strategy
            v = meta_wrapper._ejection_pool.pop()
            assert v.ejected()
            '''
            for route in meta_wrapper._routes:
                assert route._route.tail.value.route() is route
            '''
            if self._insert_feasible(meta_wrapper, v, deadline):
                assert meta_wrapper.feasible()
                assert meta_wrapper.valid()
                '''
                for route in meta_wrapper._routes:
                    assert route._route.tail.value.route() is route
                '''
                continue

            '''
            for route in meta_wrapper._routes:
                assert route._route.tail.value.route() is route
            '''

            assert v in meta_wrapper.nearest
            ok, result = self._squeeze(meta_wrapper, v, state, settings, deadline)
            if ok:
                assert result.feasible()
                assert result.valid()

                meta_wrapper = result
                '''
                for route in meta_wrapper._routes:
                    assert route._route.tail.value.route() is route
                '''
                continue
            meta_wrapper.activate()
            v._index = None
            if self._insert_eject(meta_wrapper, v, settings, deadline):
                assert meta_wrapper.feasible()
                self._perturb(meta_wrapper, settings.i_rand, settings.n_near, deadline)
                continue
            if self.debug:
                self._debug_print("failed", Colors.RED)
                #print(Colors.RED + "'delete_route': failed" + Colors.RESET)
            return False, None
        if self.debug:
            self._debug_print("completed successfully", Colors.GREEN)
            #print(Colors.GREEN + "'delete_route': completed successfully" + Colors.RESET)
        return True, meta_wrapper

    # determine the minimum possible number of routes
    def powerful_route_minimization_heuristic(self, settings: RMHSettings):
        if self.debug:
            self._debug_print("started", Colors.RESET)
            #print(Colors.PURPLE + f"'RM heuristic': started" + Colors.RESET)
        state = RMHState()
        # each client on his own route
        meta_wrapper = MetaWrapper(problem=self.problem)
        straight_lower_bound = ceil(sum([c.demand for c in self.problem.customers]) / self.problem.vehicle_capacity)
        lower_bound = max(settings.lower_bound, straight_lower_bound)

        deadline = time.time() + settings.t_max
        try:
            # try to reduce number of routes
            while len(meta_wrapper._routes) > lower_bound:
                if time.time() > deadline:
                    raise TimeoutError()
                assert meta_wrapper.feasible()
                assert meta_wrapper.valid()
                if self.debug:
                    self._debug_print(f"routes number: {len(meta_wrapper._routes)}", Colors.PURPLE)
                    #print(Colors.PURPLE + f"'RM heuristic': routes number: {len(meta_wrapper._routes)}" + Colors.RESET)
                ok, result = self._delete_route(meta_wrapper, state, settings, deadline)
                if not ok:
                    meta_wrapper.activate()
                    break
                meta_wrapper = result
        except TimeoutError as _:
            meta_wrapper.activate()
        if self.debug:
            self._debug_print("completed successfully", Colors.GREEN)
            #print(Colors.GREEN + "'RM heuristic': completed successfully" + Colors.RESET)
        return meta_wrapper

    def generate_initial_population(self):
        if self.debug:
            self._debug_print("started", Colors.RESET)
            #print("'generate_initial_population': started")
        initial_population = [self.powerful_route_minimization_heuristic(self.rmh_settings)]
        rmh_settings = self.rmh_settings
        rmh_settings.t_max = inf
        rmh_settings.lower_bound = len(initial_population[0])
        executor = ProcessPoolExecutor(max_workers=cpu_count())
        futures = []
        n_pop = self.gip_settings.n_pop
        for _ in range(n_pop - 1):
            futures.append(executor.submit(self.powerful_route_minimization_heuristic, rmh_settings))
        wait(futures, timeout=self.gip_settings.t_total)
        executor.shutdown(wait=True, cancel_futures=True)
        initial_population.extend([future.result() for future in futures if future.done()])
        if self.debug:
            self._debug_print(f"{len(initial_population)} solutions were generated honestly", Colors.RESET)
            #print(f"'generate_initial_population': {len(initial_population)} solutions were generated honestly")
            self._debug_print(f"{n_pop - len(initial_population)} duplicates have been added", Colors.RESET)
            #print(f"'generate_initial_population': {n_pop - len(initial_population)} duplicates have been added")
        initial_population.extend([copy(initial_population[-1]) for _ in range(n_pop - len(initial_population))])
        if self.debug:
            self._debug_print("completed successfully", Colors.GREEN)
            #print(Colors.GREEN + "'generate_initial_population': completed successfully" + Colors.RESET)
        return initial_population

    # edge assembly crossover type
    def eax(self, p_a, p_b, strategy):
        pass

    # restore feasibility
    def repair(self, routes, alpha, beta, deadline):
        pass

    # improve efficiency
    def local_search(self):
        pass

    def execute(self):
        pass