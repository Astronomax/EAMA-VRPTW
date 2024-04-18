from eama.modification import Modification

from enum import Enum


class ExchangeType(Enum):
    TwoOpt = 1,
    OutRelocate = 2,
    Exchange = 3


class Exchange:
    def __init__(self, v: 'ListNode', w: 'ListNode', type: ExchangeType):
        assert v is not None and w is not None 
        self._v = v
        self._w = w
        self._type = type

    def appliable(self):
        try:
            if self._type == ExchangeType.TwoOpt:
                if self._v.route() is self._w.route():
                    raise Exception()
                if self._v.node().tail():
                    raise Exception()
                if self._w.node().tail():
                    raise Exception()
                # the path after applying two-opt should not be empty
                if self._v.node().head() and self._w.next().node().tail():
                    raise Exception()
                if self._w.node().head() and self._v.next().node().tail():
                    raise Exception()
            elif self._type == ExchangeType.OutRelocate:
                if self._v.node().head():
                    raise Exception()
                if self._v.node().tail():
                    raise Exception()
                # no path should be empty after applying exchange
                if len(self._v.route()) <= 1:
                    raise Exception()
                # w is allowed to be tail but not allowed to be head
                if self._w.node().head():
                    raise Exception()
                if self._v is self._w:
                    raise Exception()
            elif self._type == ExchangeType.Exchange:
                if self._v.node().head():
                    raise Exception()
                if self._v.node().tail():
                    raise Exception()
                if self._w.node().head():
                    raise Exception()
                if self._w.node().tail():
                    raise Exception()
        except Exception as _:
            return False
        return True  

    def apply(self):
        if not self.appliable():
            return False
        v_route = self._v.route()
        w_route = self._w.route()
        if self._type == ExchangeType.TwoOpt:
            assert v_route is not w_route

            assert not self._v.node().tail()
            assert not self._w.node().tail()
            # the path after applying two-opt should not be empty
            assert not self._v.node().head() or not self._w.next().node().tail()
            assert not self._w.node().head() or not self._v.next().node().tail()

            v_node = self._v.node()
            w_node = self._w.node()
            v_node.next, w_node.next = w_node.next, v_node.next
            v_node.next.prev = v_node
            w_node.next.prev = w_node

            for node in self._v.node().next.iter():
                node.value._index._route = v_route
            for node in self._w.node().next.iter():
                node.value._index._route = w_route
            v_route._route.length = 0
            for node in v_route._route.head.next.iter():
                if node.tail():
                    break
                v_route._route.length += 1
            w_route._route.length = 0    
            for node in w_route._route.head.next.iter():
                if node.tail():
                    break
                w_route._route.length += 1
        elif self._type == ExchangeType.OutRelocate:
            assert not self._w.ejected()
            assert not self._v.node().head()
            assert not self._v.node().tail()
            # no path should be empty after applying exchange
            assert len(v_route) > 1
            # w is allowed to be tail but not allowed to be head
            assert not self._w.node().head()
            assert self._v is not self._w

            l = len(v_route)
            v_route.eject(self._v)
            assert len(v_route) == l - 1
            for node in v_route._route.head.iter():
                assert node.value.route() is v_route
            for node in w_route._route.head.iter():
                assert node.value.route() is w_route
            w_route.insert(self._w, self._v)
            for node in v_route._route.head.iter():
                assert node.value.route() is v_route
            for node in w_route._route.head.iter():
                assert node.value.route() is w_route
        elif self._type == ExchangeType.Exchange:
            assert not self._v.node().head()
            assert not self._v.node().tail()
            assert not self._w.node().head()
            assert not self._w.node().tail()
            
            v_node, w_node = self._v.node(), self._w.node()
            v_node.value, w_node.value = self._w, self._v
            self._v._index, self._w._index = self._w._index, self._v._index
        v_route._pc.update()
        v_route._dc.update()
        if v_route is not w_route:
            w_route._pc.update()
            w_route._dc.update()
        return True


class ExchangeFast(Modification):
    def __init__(self, v: 'ListNode', w: 'ListNode', type: ExchangeType):
        assert v is not None and w is not None
        self._v = v
        self._w = w
        self._type = type

    def appliable(self):
        return Exchange(self._v, self._w, self._type).appliable()

    def penalty_delta(self, alpha, beta):
        assert self.appliable()
        from eama.penalty_calculator import PenaltyCalculator
        if self._type == ExchangeType.TwoOpt:
            return PenaltyCalculator.two_opt_penalty_delta(self._v, self._w, alpha, beta)
        elif self._type == ExchangeType.OutRelocate:
            return PenaltyCalculator.out_relocate_penalty_delta(self._v, self._w, alpha, beta)
        elif self._type == ExchangeType.Exchange:
            return PenaltyCalculator.exchange_penalty_delta(self._v, self._w, alpha, beta)
        assert False

    def distance_delta(self):
        assert self.appliable()
        from eama.distance_calculator import DistanceCalculator
        if self._type == ExchangeType.TwoOpt:
            return DistanceCalculator.two_opt_distance_delta(self._v, self._w)
        elif self._type == ExchangeType.OutRelocate:
            return DistanceCalculator.out_relocate_distance_delta(self._v, self._w)
        elif self._type == ExchangeType.Exchange:
            return DistanceCalculator.exchange_distance_delta(self._v, self._w)
        assert False

    def apply(self):
        return Exchange(self._v, self._w, self._type).apply()

    def feasible(self):
        return (self.penalty_delta(1, 1) == 0)


class ExchangeSlow(Modification):
    def __init__(self, v: 'ListNode', w: 'ListNode', type: ExchangeType, c_delta: float, tw_delta: float, dist_delta: float):
        assert v is not None and w is not None 
        self._v = v
        self._w = w
        self._type = type
        self._c_delta = c_delta
        self._tw_delta = tw_delta
        self._dist_delta = dist_delta

    def appliable(self):
        return Exchange(self._v, self._w, self._type).appliable()

    def penalty_delta(self, alpha, beta):
        return alpha * self._c_delta + beta * self._tw_delta

    def distance_delta(self):
        return self._dist_delta

    def apply(self):
        return Exchange(self._v, self._w, self._type).apply()
    
    def feasible(self):
        return (self.penalty_delta(1, 1) == 0)


def exchanges(v: 'CustomerWrapper', w: 'CustomerWrapper'):
    return list(filter(lambda x: x.appliable(), [
        ExchangeFast(v.prev(), w, ExchangeType.TwoOpt),
        ExchangeFast(v, w.prev(), ExchangeType.TwoOpt),
        ExchangeFast(v, w, ExchangeType.OutRelocate),
        ExchangeFast(v, w.next(), ExchangeType.OutRelocate),
        ExchangeFast(v, w.prev(), ExchangeType.Exchange),
        ExchangeFast(v, w.next(), ExchangeType.Exchange),
    ]))