import random
from typing import List
import unittest
import bisect

def rand7():
    return random.randint(1, 7)

class Solution:

    '''

    1. random.random() return a float number [0.0, 1.0)

    2. random.randint(a, b) returns an int [a, b] . This is equivalant to randrange(a, b + 1)

    3. random.randrange(start=0, stop, step=1) returns an int 【start, stop) and in step. both start and step are
    optional

    4. random.choice(seq) pick one from sequence

    5. random.sample(seq, k) pick k from sequence

    6. random.shuffle(seq) shuffles a list in place. do not return anything.
    If want to return a new list, use random.sample(seq, len(seq))


    '''

    def shuffle_array(self, l: List[int]) -> None:
        random.shuffle(l)

    def rand10(self):
        '''
        steps for this sort of q.
        1. convert the original ramdom generator to zero based. this is because rand2 + rand2 will [2, 4] and not rand4
        2. expand a range that surpase the needed range. rand7 * rand7 will cover rand49. but in this way
        (rand7 - 1) * 7 + (rand7 - 1)
        this is range [0, 48]
        3. reject sampling. Now we need this range to be a multiple fo the desired range N, so we can do x % N
        . and will be a sampled range [0, N - 1]. In order to be multiple of N, we simple generate again if it's >
        N*M
        4. In the end we + 1 to the [0, N - 1] range to generate [1, N]
        :return:
        '''

        # one rand7 is not going to cover rand10. So we use 2 rand7 and after zero based cover [0, 48]
        # we reject when >= 40 and regenerate again.
        # we use the X % 10 and then + 1 to make it 1-based again

        rand40 = 40
        while rand40 >= 40:
            rand40 = (rand7() - 1) * 7 + (rand7() - 1)
        return rand40 % 10 + 1


class RandomGenerator:

    def __init__(self, minv: int, maxv: int) -> None:
        self.minv, self.maxv = minv, maxv
        self.nums = []
        self.used = set()
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.nums = [i for i in range(self.minv, self.maxv + 1)]
        self.used = set()

    def update(self, left, right) -> None:
        new_nums = {x for x in range(left, right + 1)}
        self.nums = list(new_nums - self.used)
        self.minv, self.maxv = left, right

    def __next__(self) -> int:
        if not self.nums:
            # self.reset()
            raise StopIteration
        idx = random.randrange(len(self.nums))
        res = self.nums[idx]
        self.nums[idx] = self.nums[-1]
        self.nums.pop()
        self.used.add(res)
        return res



class Solution710:
    # idea is get M = N - len(B). And for numbers between [0, M) use a hm to map to one whitelisted after M

    def __init__(self, N: int, blacklist: List[int]):
        self.m = N - len(blacklist)
        hs = {x for x in range(self.m, N)} - set(blacklist)
        self.hm = {}
        it = iter(hs)
        for b in blacklist:
            if b < self.m:
                self.hm[i] = next(it)

    def pick(self) -> int:
        x = random.randrange(self.m)
        return self.hm.get(x, x)


class TestRandom(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = [
            [1],
            [1,2,3]
        ]

    def test_shuffle(self):
        s = Solution()

        for ip in self.inputs:
            for i in range(5):
                s.shuffle_array(ip)
                self.assertEqual(len(ip), len(ip))

# s = Solution()
#
# # t.test_shuffle()
#
# r = RandomGenerator(5, 8)
# for i in range(2):
#     print(next(r))
# r.update(6, 10)
# for i in range(10):
#     print(next(r))


if __name__ == '__main__':
    unittest.main()
