# Power of 4
# Random iterator divided by 5
# Text editor deisgn


class Set2:
    def power_of_4(self, x: int) -> bool:
        return x > 0 and (x & (x - 1)) == 0 and (x & 0x55555555) == x


class Iterator5:
    def __init__(self, iterator):
        self.iter = iterator
        self.has_cache = False
        self.cache = None

    def hasNext(self):
        if self.has_cache:
            return True
        while self.iter.hasNext():
            x = self.iter.next()
            if x % 5 == 0:
                self.cache = x
                self.has_cache = True
                return True
        return False

    def next(self):
        if self.has_cache:
            self.has_cache = False
            return self.cache
        while self.iter.hasNext():
            x = self.iter.next()
            if x % 5 == 0:
                return x

        raise StopIteration
