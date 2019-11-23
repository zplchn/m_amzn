import collections


class MovingAverage:

    def __init__(self, size: int):
        self.size = size
        self.sum = 0
        self.q = collections.deque()

    def next(self, val: int) -> float:
        if len(self.q) == self.size:
            self.sum -= self.q.popleft()
        self.sum += val
        self.q.append(val)
        return self.sum // len(self.q)



class ZigzagIterator(object):

    def __init__(self, v1, v2):
        """
        Initialize your data structure here.
        :type v1: List[int]
        :type v2: List[int]
        """
        self.vecs = [v1, v2]
        self.q = collections.deque()
        # iter1, iter2 = iter(v1), iter(v2) Python iterator only has one function that is next()
        if v1:
            self.q.append((0, 0))
        if v2:
            self.q.append((1, 0))


    def next(self):
        """
        :rtype: int
        """
        v, c = self.q.popleft()
        x = self.vecs[v][c]
        c += 1
        if c < len(self.vecs[v]):
            self.q.append((v, c))
        return x

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.q) > 0