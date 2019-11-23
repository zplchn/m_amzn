import random


class RandomizedSet380:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hm = {}
        self.num = []

    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.hm:
            return False
        self.num.append(val)
        self.hm[val] = len(self.num) - 1
        return True

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val not in self.hm:
            return False
        idx = self.hm[val]
        self.num[idx], self.num[-1] = self.num[-1], self.num[idx]
        self.num[idx], self.num[-1] = self.num[-1], self.num[idx]
        # need update first then pop. because it's the only num, pop first will make hm leave with the number
        self.hm[self.num[idx]] = idx
        del self.hm[val]
        return True

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return random.choice(self.num)