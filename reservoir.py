import random
from typing import List


'''
Reservoir sampling:

When to use:
1. list length is unknown
2. Save space


How to use:

Loop through entire data stream

Use a counter cnt 
increment when meet target

if random.randrange(cnt) == 0:
    swap into reservoir
    
return the one in reservoir




'''


class Solution398:

    def __init__(self, nums: List[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        # to save space, use reservoir sampling
        res, cnt = -1, 0
        for i in range(len(self.nums)):
            if self.nums[i] == target:
                cnt += 1
                if random.randrange(cnt) == 0:
                    res = i
        return res