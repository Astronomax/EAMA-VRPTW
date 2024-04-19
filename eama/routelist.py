from eama.structure import Problem

class ListNode:
    def __init__(self, customer: 'CustomerWrapper'):
        self.value = customer
        self.next = None
        self.prev = None

    def iter(self):
        curr_node = self
        while curr_node:
            yield curr_node
            curr_node = curr_node.next

    def head(self):
        return self.prev is None
    
    def tail(self):
        return self.next is None

class RouteList:
    def __init__(self, problem: 'ProblemWrapper', list: list = []):
        self._problem = problem
        from eama.meta_wrapper import CustomerWrapper
        self.head = ListNode(CustomerWrapper(problem.depot))
        node = self.head
        for value in list:
            node.next = ListNode(value)
            node.next.prev = node
            node = node.next
        self.tail = ListNode(CustomerWrapper(problem.depot))
        node.next = self.tail
        node.next.prev = node
        self.length = len(list)

    def __len__(self):
        '''
        len = 0
        for node in self.head.next.iter():
            if node.tail():
                break
            len += 1
        assert len == self.length
        '''
        return self.length

    def __copy__(self):
        result = RouteList(self._problem)
        result.head = ListNode(self.head.value)
        prev = result.head
        for node in self.head.next.iter():
            if node.tail():
                break
            prev.next = ListNode(node.value)
            prev.next.prev = prev
            prev = prev.next
        result.tail = ListNode(self.tail.value)
        prev.next = result.tail
        result.tail.prev = prev
        result.length = self.length
        return result

    def insert(self, index, value: 'CustomerWrapper'):
        assert index is not None
        assert value is not None
        assert value.number != self._problem.depot.number
        if isinstance(index, int):
            assert index >= 0
            prev_node = self.get_node(index + 1).prev
        elif isinstance(index, ListNode):
            #print('ListNode')
            assert not index.head()
            prev_node = index.prev
            assert prev_node.next is index
        new_node = ListNode(value)
        new_node.next = prev_node.next
        prev_node.next.prev = new_node
        new_node.prev = prev_node
        prev_node.next = new_node
        self.length += 1
        if isinstance(index, ListNode):
            assert new_node.next is index
        return new_node

    def remove(self, index):
        if isinstance(index, int):
            node = self.get_node(index + 1)
        elif isinstance(index, ListNode):
            node = index
        assert node.value.number != self._problem.depot.number
        assert not node.head() and not node.tail()
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
        self.length -= 1
        return node

    def __getitem__(self, index):
        assert index >= 0
        curr_node = self.get_node(index + 1)
        assert not curr_node.head() and not curr_node.tail()
        return curr_node.value

    def __setitem__(self, index, value):
        assert index >= 0
        curr_node = self.get_node(index + 1)
        assert not curr_node.head() and not curr_node.tail()
        assert curr_node.value.number != self._problem.depot.number
        curr_node.value = value

    def get_node(self, index):
        assert index >= 0
        curr_node = self.head
        for _ in range(index):
            curr_node = curr_node.next
            if not curr_node:
                assert False
        return curr_node

    def __iter__(self):
        for node in self.head.next.iter():
            if node.tail():
                break
            yield node.value