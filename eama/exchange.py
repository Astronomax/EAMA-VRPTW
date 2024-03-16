from enum import Enum


class ExchangeType(Enum):
    TwoOpt = 1,
    OutRelocate = 2,
    Exchange = 3


class Exchange:
    def __init__(self, v_route, v_pos, w_route, w_pos, type):
        self.v_route = v_route
        self.v_pos = v_pos
        self.w_route = w_route
        self.w_pos = w_pos
        self.type = type


def exchange_appliable(e: Exchange):
    n_v = len(e.v_route.route._customers)
    n_w = len(e.w_route.route._customers)
    if e.v_pos < 0 or e.w_pos < 0:
        return False
    if e.type == ExchangeType.TwoOpt:
        if e.v_route is e.w_route:
            return False
        elif e.v_pos >= n_v - 1 or e.w_pos >= n_w - 1:
            return False
        elif e.v_pos == n_v - 2 and e.w_pos == 0:
            return False
        elif e.w_pos == n_w - 2 and e.v_pos == 0:
            return False
    elif e.type == ExchangeType.OutRelocate:
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
    elif e.type == ExchangeType.Exchange:
        if e.v_pos <= 0 or e.v_pos >= n_v - 1:
            return False
        elif e.w_pos <= 0 or e.w_pos >= n_w - 1:
            return False
    return True

def apply_exchange(e: Exchange):
    if e.type == ExchangeType.TwoOpt:
        if e.v_route is e.w_route:
            return
        v_pf = e.v_route.route._customers[:e.v_pos + 1]
        v_sf = e.v_route.route._customers[e.v_pos + 1:]
        w_pf = e.w_route.route._customers[:e.w_pos + 1]
        w_sf = e.w_route.route._customers[e.w_pos + 1:]
        e.v_route.route._customers = v_pf + w_sf
        e.w_route.route._customers = w_pf + v_sf
    elif e.type == ExchangeType.OutRelocate:
        w_pos = e.w_pos
        if e.v_route is e.w_route and e.v_pos < e.w_pos:
            w_pos = e.w_pos - 1
        v = e.v_route.route._customers.pop(e.v_pos)
        e.w_route.route._customers.insert(w_pos, v)
    elif e.type == ExchangeType.Exchange:
        e.v_route.route._customers[e.v_pos], e.w_route.route._customers[e.w_pos] = \
        e.w_route.route._customers[e.w_pos], e.v_route.route._customers[e.v_pos]
    e.v_route.recalc(e.v_route.route)
    e.w_route.recalc(e.w_route.route)

def exchange_penalty_delta(e: Exchange, alpha, beta):
    if e.type == ExchangeType.TwoOpt:
        return e.v_route.two_opt_penalty_delta(e.v_pos, e.w_route, e.w_pos, alpha, beta)
    elif e.type == ExchangeType.OutRelocate:
        return e.v_route.out_relocate_penalty_delta(e.v_pos, e.w_route, e.w_pos, alpha, beta)
    elif e.type == ExchangeType.Exchange:
        return e.v_route.exchange_penalty_delta(e.v_pos, e.w_route, e.w_pos, alpha, beta)