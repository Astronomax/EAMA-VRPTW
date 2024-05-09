from eama.meta_wrapper import MetaWrapper, RouteWrapper, CustomerWrapper
from eama.modification import Modification
from math import inf
from bisect import bisect_left

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
    p_sum = 0
    dist_pf = [0] * n
    dist_pf[0] = 0


    demand_pfsum = [0] * n
    pickup_pfsum = [0] * n
    demand_pfsum[0] = route[0].demand
    pickup_pfsum[0] = max(0, -route[0].demand)


    from sortedcontainers import SortedList
    demand_pfsums = SortedList()
    demand_pfsums.add(demand_pfsum[0])

    c_tilde, x = 0, 0

    def update_x():
        nonlocal c_tilde, x
        c_tilde = pickup_pfsum[not_ejected[-1]] + pc.pickup_sfsum[ejected[-1] + 1]
        x = pc._problem.vehicle_capacity - c_tilde

    update_x()

    def update(j):
        nonlocal a, a_quote, dist_pf, demand_pfsum, pickup_pfsum
        for i in range(j, j + 2):
            v, w = route[not_ejected[-1]], route[i]
            dist_pf[i] = dist_pf[not_ejected[-1]] + v.c(w)
            a_quote[i] = a[not_ejected[-1]] + v.s + v.c(w)
            a[i] = min(max(a_quote[i], w.e), w.l)
            demand_pfsum[i] = demand_pfsum[not_ejected[-1]] + w.demand
            pickup_pfsum[i] = demand_pfsum[not_ejected[-1]] + max(0, -w.demand)

    def backtrack():
        nonlocal p_sum, demand_pfsums, c_tilde, x
        j = ejected.pop()
        ejection.pop()
        while not_ejected and not_ejected[-1] > ejected[-1]:
            demand_pfsums.remove(demand_pfsum[not_ejected[-1]])
            not_ejected.pop()
        p_sum -= p[route[j].number]
        update_x()

    def incr_k():
        nonlocal p_sum, c_tilde, x
        j = ejected[-1]
        ejected.append(j + 1)
        ejection.append(route[j + 1])
        p_sum += p[route[j + 1].number]
        update(ejected[-1])
        update_x()

    def incr_last():
        nonlocal p_sum, demand_pfsums, c_tilde, x
        j = ejected[-1]
        ejected[-1] = j + 1
        ejection[-1] = route[j + 1]
        not_ejected.append(j)
        demand_pfsums.add(demand_pfsum[j])
        p_sum -= p[route[j].number] - p[route[j + 1].number]
        update(ejected[-1])
        update_x()

    update(ejected[-1])
    p_sum += p[route[ejected[-1]].number]

    while True:
        j = ejected[-1] + 1
        sorted_sfs = pc.sorted_sfs[j]
        pfsum_pos = demand_pfsum[not_ejected[-1]]
        lower_bound = bisect_left(sorted_sfs, x - pfsum_pos)
        if p_sum < p_best and a_quote[j] <= route[j].l and a[j] <= pc.z[j] and pc.tw_sf[j] == 0\
            and lower_bound == len(sorted_sfs) and demand_pfsums.bisect_left(x) == len(demand_pfsums):

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
