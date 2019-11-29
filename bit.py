from typing import List


class Solution:
    def singleNumber260(self, nums: List[int]) -> List[int]:
        # we xor everyone, and then only the two distinct x = a ^ b will stay. and then we need to find a bit in x
        # that is 1 and use this as mask to and every one and xor together, this separate all nums to two group but
        # since every number  will appear twice in each group. so we have two groups that only lives the two distince.
        if not nums:
            return []
        xor = 0
        for x in nums:
            xor ^= x
        # find the right most 1
        xor &= -xor
        res = [0, 0]
        for x in nums:
            if x & xor:
                res[1] ^= x
            else:
                res[0] ^= x
        return res