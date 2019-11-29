import random
import string

class Codec535:

    strs = string.ascii_letters + '0123456789'

    def __init__(self):
        self.urltocode = {}
        self.codetourl = {}

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.

        :type longUrl: str
        :rtype: str
        """
        while longUrl not in self.urltocode:
            code = ''.join(random.sample(self.strs, 6)) # (52 + 10) ** 6 = 56 billion chance of collision
            if code not in self.codetourl:
                self.urltocode[longUrl] = code
                self.codetourl[code] = longUrl
        return self.urltocode[longUrl]

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.

        :type shortUrl: str
        :rtype: str
        """
        return self.codetourl[shortUrl[-6:]]

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