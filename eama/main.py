from eama.exchange import *
from eama.penalty_calculator import *
from eama.structure import *
from eama.meta_wrapper import MetaWrapper, RouteWrapper, CustomerWrapper
from eama.insertion import insertions
from eama.ejection import feasible_ejections, Ejection

from itertools import tee
from math import ceil, inf
from random import randint, sample, shuffle, choice
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count

import time


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

    def _split_by_feasibility(self, meta_wrapper: MetaWrapper):
        feasible, infeasible = [], []
        infeasible, feasible = tee(meta_wrapper._routes)
        # routes from "routes" copied by reference into
        # "infeasible" and "feasible". we use this for convenience
        infeasible = list(filter(lambda r: not r.feasible(), infeasible))
        feasible = list(filter(lambda r: r.feasible(), feasible))
        return feasible, infeasible

    def _insert_feasible(self, meta_wrapper: MetaWrapper, v: CustomerWrapper, deadline: float = inf):
        assert v.ejected()
        if self.debug:
            print("'insert_feasible' started")
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid(v)

        for insertion in insertions(v, meta_wrapper=meta_wrapper):
            if time.time() > deadline:
                raise TimeoutError()

            if insertion.feasible():
                insertion.apply()
                assert meta_wrapper.feasible()
                if self.debug:
                    print(Colors.GREEN + "'insert_feasible' completed successfully" + Colors.RESET)
                return True
        if self.debug:
            print(Colors.RED + "'insert_feasible' failed" + Colors.RESET)
        return False

    def _squeeze(self, meta_wrapper: MetaWrapper, v: CustomerWrapper, state: RMHState, settings: RMHSettings, deadline: float = inf):
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid(v)
        if self.debug:
            print("'squeeze' started")
        meta_wrapper = meta_wrapper.inherit()

        # insert such that penalty is minimum
        insertion=min(list(insertions(v, meta_wrapper=meta_wrapper)),\
            key=lambda e: e.penalty_delta(state.alpha, state.beta))
        insertion.apply()
        _, infeasible = self._split_by_feasibility(meta_wrapper)

        while len(infeasible) > 0:
            if self.debug:
                print(f'penalty_sum: {sum([r.get_penalty(state.alpha, state.beta) for r in infeasible])}')
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
                            if exchange_delta >= v_route_penalty - 1e-5:
                                raise BreakLoop
            except BreakLoop:
                pass

            if self.debug:
                print(f"'squeeze': opt exchange delta: {opt_exchange_delta}")
            if not opt_exchange or opt_exchange_delta > -1e-5:
                if meta_wrapper.get_penalty(1, -1) < 0:
                    state.beta *= 0.99
                else:
                    state.beta /= 0.99
                if self.debug:
                    print(Colors.RED + "'squeeze' failed" + Colors.RESET)
                    print(f"'squeeze': beta after correction: {state.beta}")
                return False, None
            opt_exchange.apply()
            _, infeasible = self._split_by_feasibility(meta_wrapper)
        if self.debug:
            print(Colors.GREEN + "'squeeze' completed successfully" + Colors.RESET)
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid()
        return True, meta_wrapper

    def _insert_eject(self, meta_wrapper: MetaWrapper, v: CustomerWrapper, settings: RMHSettings, deadline: float = inf):
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid(v)
        assert v.ejected()
        if self.debug:
            print("'insert_eject' started")
        self.p[v.number] += 1
        opt_insertion_ejection = None
        p_best = inf
        for route in meta_wrapper._routes:
            assert v.ejected()
            for insertion in insertions(v, route=route):
                if time.time() > deadline:
                    raise TimeoutError()

                assert v.ejected()
                insertion.apply()
                assert not v.ejected()

                assert not insertion._index.route().feasible()
                for ejection, p_sum in feasible_ejections(route, self.p, settings.k_max, p_best):
                    if p_sum < p_best or (p_sum == p_best and len(ejection._ejection) < len(opt_insertion_ejection[1]._ejection)):
                        opt_insertion_ejection = (insertion, ejection)
                        p_best = p_sum

                route.eject(v)
                assert v.ejected()
                route._pc.update()
                route._dc.update()
        if self.debug:
            print(f"'insert_eject': opt insertion-ejection p_sum: {p_best}")
        if opt_insertion_ejection == None:
            if self.debug:
                print(Colors.RED + "'insert_eject' failed" + Colors.RESET)
            return False
        insertion, ejection = opt_insertion_ejection
        route = insertion._index.route()
        insertion.apply()
        assert not route.feasible()
        ejection.apply()
        assert route.feasible()
        assert meta_wrapper.feasible()
        if self.debug:
            print(Colors.GREEN + "'insert_eject' completed successfully" + Colors.RESET)
        return True 

    def _perturb(self, meta_wrapper: MetaWrapper, i_rand: int, n_near: int, deadline: float = inf):
        assert meta_wrapper.feasible()
        if self.debug:
            print("'perturb' started")
        exchanges_cnt = 0
        for _ in range(i_rand):
            if time.time() > deadline:
                raise TimeoutError()
            e = choice(list(meta_wrapper.N_near(n_near)))
            if e.feasible():
                e.apply()
                exchanges_cnt += 1
                assert meta_wrapper.feasible()
        if self.debug:
            print(f"'perturb': applied {exchanges_cnt} exchanges")
            print(Colors.GREEN + "'perturb' completed successfully" + Colors.RESET)

    def _delete_route(self, meta_wrapper: MetaWrapper, state: RMHState, settings: RMHSettings, deadline: float = inf):
        if self.debug:
            print("'delete_route' started")
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid()
        meta_wrapper = meta_wrapper.inherit()

        # eliminate_random_route
        i = randint(0, len(meta_wrapper._routes) - 1)
        ejected = list(meta_wrapper._routes.pop(i))
        for v in ejected:
            v._index = None
        meta_wrapper._ejection_pool.extend(ejected)
        assert meta_wrapper.feasible()
        assert meta_wrapper.valid()

        # trying to empty ejection_pool
        while len(meta_wrapper._ejection_pool) > 0:
            if time.time() > deadline:
                raise TimeoutError()
            assert meta_wrapper.feasible()
            assert meta_wrapper.valid()
            if self.debug:
                print(f'ejection_pool: {len(meta_wrapper._ejection_pool)}')
            # remove v from EP with the LIFO strategy
            v = meta_wrapper._ejection_pool.pop()
            assert v.ejected()
            if self._insert_feasible(meta_wrapper, v, deadline):
                assert meta_wrapper.feasible()
                assert meta_wrapper.valid()
                continue
            assert v in meta_wrapper.nearest
            ok, result = self._squeeze(meta_wrapper, v, state, settings, deadline)
            if ok:
                assert result.feasible()
                assert result.valid()
                meta_wrapper = result
                continue
            meta_wrapper.activate()
            v._index = None
            if self._insert_eject(meta_wrapper, v, settings, deadline):
                assert meta_wrapper.feasible()
                self._perturb(meta_wrapper, settings.i_rand, settings.n_near, deadline)
                continue
            if self.debug:
                print(Colors.RED + "'delete_route' failed" + Colors.RESET)
            return False, None
        if self.debug:
            print(Colors.GREEN + "'delete_route' completed successfully" + Colors.RESET)
        return True, meta_wrapper

    # determine the minimum possible number of routes
    def powerful_route_minimization_heuristic(self, settings: RMHSettings):
        state = RMHState()
        # each client on his own route
        meta_wrapper = MetaWrapper(self.problem)
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
                    print(Colors.PURPLE + f"'RM heuristic': routes number: {len(meta_wrapper._routes)}" + Colors.RESET)
                ok, result = self._delete_route(meta_wrapper, state, settings, deadline)
                if not ok:
                    meta_wrapper.activate()
                    break
                meta_wrapper = result
        except TimeoutError as _:
            meta_wrapper.activate()
        if self.debug:
            print(Colors.GREEN + "'powerful_route_minimization_heuristic' completed successfully" + Colors.RESET)
        return meta_wrapper

    def generate_initial_population(self):
        if self.debug:
            print("'generate_initial_population' started")
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
            print(f"'generate_initial_population': {len(initial_population)} solutions were generated honestly")
            print(f"'generate_initial_population': {n_pop - len(initial_population)} duplicates have been added")
        initial_population.extend([copy(initial_population[-1]) for _ in range(n_pop - len(initial_population))])
        if self.debug:
            print(Colors.GREEN + "'generate_initial_population' completed successfully" + Colors.RESET)
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