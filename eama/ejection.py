from eama.meta_wrapper import MetaWrapper, RouteWrapper, CustomerWrapper
from eama.modification import Modification
from math import inf


class Ejection(Modification):
    def __init__(self, meta_wrapper: MetaWrapper, ejection: list[CustomerWrapper], c_delta: float, tw_delta: float, dist_delta: float):
        self._meta_wrapper = meta_wrapper
        self._ejection = ejection
        self._c_delta = c_delta
        self._tw_delta = tw_delta
        self._dist_delta = dist_delta

    def appliable(self):
        try:
            if len(self._ejection) > 0:
                route = self._ejection[0].route()
                if not all([v.route() is route for v in self._ejection]):
                    raise Exception()
        except Exception as _:
            return False
        return True  

    def penalty_delta(self, alpha, beta):
        return alpha * self._c_delta + beta * self._tw_delta

    def distance_delta(self):
        return self._dist_delta

    def apply(self):
        if len(self._ejection) > 0:
            route = self._ejection[0].route()
            assert all([v.route() is route for v in self._ejection])
            for v in self._ejection:
                route.eject(v)
            self._meta_wrapper._ejection_pool.extend(self._ejection)
            assert all([v.ejected() for v in self._meta_wrapper._ejection_pool])
            route._pc.update()
            route._dc.update()

    def __copy__(self):
        return Ejection(self._meta_wrapper, self._ejection.copy(), self._c_delta, self._tw_delta, self._dist_delta)


# iterate over feasible ejections lexicographically
def feasible_ejections(route: 'RouteWrapper', p: list[int], k_max: int, p_best = inf):
    assert route._meta_wrapper is not None
    meta_wrapper = route._meta_wrapper
    pc = route._pc
    dc = route._dc
    c_delta = -pc.get_penalty(1, 0)
    tw_delta = -pc.get_penalty(0, 1)
    dist_before = dc.get_distance()
    route = [v.value for v in route._route.head.iter()]
    n = len(route)
    ejected = [1]
    ejection = [route[1]]
    not_ejected = [0]
    a = [0] * n
    a_quote = [0] * n
    a_quote[0] = a[0] = meta_wrapper.problem.depot.e
    q_quote = pc.demand_pf[-1]
    total_demand = q_quote
    p_sum = 0
    dist_pf = [0] * n
    dist_pf[0] = 0

    def update(j):
        nonlocal a, a_quote
        for i in range(j, j + 2):
            v, w = route[not_ejected[-1]], route[i]
            dist_pf[i] = dist_pf[not_ejected[-1]] + v.c(w)
            a_quote[i] = a[not_ejected[-1]] + v.s + v.c(w)
            a[i] = min(max(a_quote[i], w.e), w.l)

    def backtrack():
        nonlocal p_sum, total_demand
        j = ejected.pop()
        ejection.pop()
        while not_ejected and not_ejected[-1] > ejected[-1]:
            not_ejected.pop()
        p_sum -= p[route[j].number]
        total_demand += route[j].demand

    def incr_k():
        nonlocal p_sum, total_demand
        j = ejected[-1]
        ejected.append(j + 1)
        ejection.append(route[j + 1])
        p_sum += p[route[j + 1].number]
        total_demand -= route[j + 1].demand
        update(ejected[-1])

    def incr_last():
        nonlocal p_sum, total_demand
        j = ejected[-1]
        ejected[-1] = j + 1
        ejection[-1] = route[j + 1]
        not_ejected.append(j)
        p_sum -= p[route[j].number] - p[route[j + 1].number]
        total_demand += route[j].demand - route[j + 1].demand
        update(ejected[-1])

    update(ejected[-1])
    p_sum += p[route[ejected[-1]].number]
    total_demand -= route[ejected[-1]].demand

    while True:
        j = ejected[-1] + 1
        if p_sum < p_best and a_quote[j] <= route[j].l and a[j] <= pc.z[j] and pc.tw_sf[j] == 0\
            and total_demand <= meta_wrapper.problem.vehicle_capacity:
            dist_delta = dist_pf[j] + dc._dist_sf[j] - dist_before
            yield Ejection(meta_wrapper, ejection, c_delta, tw_delta, dist_delta), p_sum
            p_best = p_sum

        if p_sum < p_best and ejected[-1] < n - 2 and len(ejected) < k_max:
            incr_k()
        else:
            if ejected[-1] >= n - 2:
                if len(ejected) == 1:
                    return
                backtrack()
            prev = ejected[-1]
            incr_last()
            while route[prev].l < a_quote[prev]:
                if len(ejected) == 1:
                    return
                backtrack()
                prev = ejected[-1]
                incr_last()
            '''
            while p_sum >= p_best or route[prev].l < a_quote[prev]:# or\
                #(len(ejection) > 1 and a[prev] == pc.a[prev] and q_quote <= meta_wrapper.problem.vehicle_capacity):
                if len(ejected) == 1:
                    return
                backtrack()
                prev = ejected[-1]
                incr_last()
            '''

'''
def check_ejection_metadata_is_valid(route, p, k_max, ejection, a_quote, a, total_demand, p_sum):
    assert len(ejection) <= k_max
    j = ejection[-1] + 1
    t_a_quote = route[0].e
    t_a = route[0].e
    last = 0
    for i in range(1, j + 1):
        if i in ejection:
            continue
        t_a_quote = t_a + route[last].s + route[last].c(route[i])
        t_a = min(max(t_a_quote, route[i].e), route[i].l)
        if t_a != a[i]:
            return False
        if t_a_quote != a_quote[i]:
            return False
        if i < j and t_a_quote > route[i].l:
            return False
        last = i
    q_quote = route.demand_pf[-1]
    total_demand_target = q_quote
    for i in ejection:
        total_demand_target -= route[i].demand
    assert total_demand_target == total_demand
    p_sum_target = 0
    for i in ejection:
        p_sum_target += p[route[i].number]
    assert p_sum_target == p_sum
    return True
'''