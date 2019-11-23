from typing import List
import collections


class Solution:
    def findMaxLength525(self, nums):
        # for subarray problem, use interval sum idea is often. keep a sum, when 1 +1, when 0 -1. and record in a hm
        # for sum -> i. so when the sum appear again, we know the interval in the middle has a sum of 0.
        if not nums:
            return 0
        maxv, sumv = 0, 0
        hm = {}
        hm[0] = -1 # trick for handle the first sum = 0 case
        for i in range(len(nums)):
            sumv += 1 if nums[i] == 1 else -1
            if sumv in hm:
                maxv = max(maxv, i - hm[sumv])
            else:
                hm[sumv] = i # do not update if same value appear again since we want longest
        return maxv

    def findShortestSubArray697(self, nums: List[int]) -> int:
        if not nums:
            return 0
        degree = max(collections.Counter(nums).values())
        if degree == 1:
            return 1
        hm = {}
        minv = len(nums)
        for i in range(len(nums)):
            if nums[i] in hm:
                idx, cnt = hm[nums[i]]
                cnt += 1
                if cnt == degree:
                    minv = min(minv, i - idx + 1)
                else:
                    hm[nums[i]] = (idx, cnt)
            else:
                hm[nums[i]] = (i, 1)
        return minv


