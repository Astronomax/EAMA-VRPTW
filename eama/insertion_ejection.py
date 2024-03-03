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

    def incr_k():
        nonlocal p_sum
        nonlocal total_demand
        j = ejection[-1]
        ejection.append(j + 1)

        j = ejection[-1]
        f = not_ejected[-1]
        for i in range(j, j + 2):
            last = r.route._customers[f]
            g = r.route._customers[i]
            a_quote[i] = a[f] + last.s + last.c(g)
            a[i] = min(max(a_quote[i], g.e), g.l)
        p_sum += p[r.route._customers[j].number]
        total_demand -= r.route._customers[j].demand

    def incr_last():
        nonlocal p_sum
        nonlocal total_demand
        j = ejection[-1]
        ejection[-1] += 1
        not_ejected.append(j)
        p_sum -= p[r.route._customers[j].number]
        total_demand += r.route._customers[j].demand

        j = ejection[-1]
        f = not_ejected[-1]
        for i in range(j, j + 2):
            last = r.route._customers[f]
            g = r.route._customers[i]
            a_quote[i] = a[f] + last.s + last.c(g)
            a[i] = min(max(a_quote[i], g.e), g.l)
        p_sum += p[r.route._customers[j].number]
        total_demand -= r.route._customers[j].demand                    
    
    j = ejection[-1]
    f = not_ejected[-1]
    for i in range(j, j + 2):
        last = r.route._customers[f]
        g = r.route._customers[i]
        a_quote[i] = a[f] + last.s + last.c(g)
        a[i] = min(max(a_quote[i], g.e), g.l)
    p_sum += p[r.route._customers[j].number]
    total_demand -= r.route._customers[j].demand

    while True:
        yield ejection, a_quote, a, total_demand, p_sum
        
        def get_next():
            nonlocal p_sum
            nonlocal total_demand
            nonlocal a_quote
            nonlocal ejection

            def backtrack():
                nonlocal p_sum
                nonlocal total_demand
                j = ejection.pop(-1)
                while len(not_ejected) > 0 and not_ejected[-1] > ejection[-1]:
                    not_ejected.pop(-1)
                p_sum -= p[r.route._customers[j].number]
                total_demand += r.route._customers[j].demand
            
            if ejection[-1] >= n - 2:
                if len(ejection) == 1:
                    return False
                backtrack()
                prev = ejection[-1]
                incr_last()
                while r.route._customers[prev].l < a_quote[prev]:
                    if len(ejection) == 1:
                        return False
                    backtrack()
                    prev = ejection[-1]
                    incr_last()

            elif len(ejection) >= k_max - 1:
                prev = ejection[-1]
                incr_last()
                while r.route._customers[prev].l < a_quote[prev]:
                    if len(ejection) == 1:
                        return False
                    backtrack()
                    prev = ejection[-1]
                    incr_last()
            else:
                incr_k()
            return True
        
        if not get_next():
            break

def check_ejection_metadata_is_valid(r, ejection, a_quote, a):
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
    return True