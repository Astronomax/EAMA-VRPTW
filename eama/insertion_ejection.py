class InsertionEjection:
    def __init__(self, route, v, insertion, ejection):
        self.route = route
        self.v = v
        self.insertion = insertion
        self.ejection = ejection


# iterate over ejections lexicographically
def ejections_gen(r, p, k_max):
    n = len(r.route._customers)
    ejection = [1]
    not_ejected = [0]
    a = [0] * n
    a_quote = [0] * n
    q_quote = r.demand_pf[-1]
    total_demand = q_quote
    p_sum = 0

    def update(j):
        nonlocal a, a_quote
        f = not_ejected[-1]
        for i in range(j, j + 2):
            last = r.route._customers[f]
            g = r.route._customers[i]
            a_quote[i] = a[f] + last.s + last.c(g)
            a[i] = min(max(a_quote[i], g.e), g.l)

    def backtrack():
        nonlocal p_sum, total_demand
        j = ejection.pop()
        while not_ejected and not_ejected[-1] > ejection[-1]:
            not_ejected.pop()
        p_sum -= p[r.route._customers[j].number]
        total_demand += r.route._customers[j].demand

    def incr_k():
        nonlocal p_sum, total_demand
        j = ejection[-1]
        ejection.append(j + 1)
        p_sum += p[r.route._customers[j + 1].number]
        total_demand -= r.route._customers[j + 1].demand
        update(ejection[-1])

    def incr_last():
        nonlocal p_sum, total_demand
        j = ejection[-1]
        ejection[-1] = j + 1
        not_ejected.append(j)
        p_sum -= p[r.route._customers[j].number] - p[r.route._customers[j + 1].number]
        total_demand += r.route._customers[j].demand - r.route._customers[j + 1].demand
        update(ejection[-1])

    update(ejection[-1])
    p_sum += p[r.route._customers[ejection[-1]].number]
    total_demand -= r.route._customers[ejection[-1]].demand

    while True:
        yield ejection, a_quote, a, total_demand, p_sum

        if ejection[-1] < n - 2 and len(ejection) < k_max:
            incr_k()
        else:
            if ejection[-1] >= n - 2:
                if len(ejection) == 1:
                    return
                backtrack()
            prev = ejection[-1]
            incr_last()
            while r.route._customers[prev].l < a_quote[prev]:
                if len(ejection) == 1:
                    return
                backtrack()
                prev = ejection[-1]
                incr_last()

def check_ejection_metadata_is_valid(r, p, k_max, ejection, a_quote, a, total_demand, p_sum):
    assert len(ejection) <= k_max
    j = ejection[-1] + 1
    t_a_quote = r.route._customers[0].e
    t_a = r.route._customers[0].e
    last = 0
    for i in range(1, j + 1):
        if i in ejection:
            continue
        t_a_quote = t_a + r.route._customers[last].s + r.route._customers[last].c(r.route._customers[i])
        t_a = min(max(t_a_quote, r.route._customers[i].e), r.route._customers[i].l)
        if t_a != a[i]:
            return False
        if t_a_quote != a_quote[i]:
            return False
        if i < j and t_a_quote > r.route._customers[i].l:
            return False
        last = i
    q_quote = r.demand_pf[-1]
    total_demand_target = q_quote
    for i in ejection:
        total_demand_target -= r.route._customers[i].demand
    assert total_demand_target == total_demand
    p_sum_target = 0
    for i in ejection:
        p_sum_target += p[r.route._customers[i].number]
    assert p_sum_target == p_sum
    return True