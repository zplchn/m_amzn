from typing import List
import random


class Solution:
    def findKthLargest215(self, nums: List[int], k: int) -> int:
        # quick select O(N), worst case O(N2)
        def partition(l, r):
            pivot = random.randint(l, r)
            nums[pivot], nums[r] = nums[r], nums[pivot]
            t = l
            for i in range(l, r):
                if nums[i] < nums[r]:
                    nums[t], nums[i] = nums[i], nums[t]
                    t += 1
            nums[t], nums[r] = nums[r], nums[t]
            return t

        if not nums or k <= 0 or k > len(nums):
            return 0
        # now kth is kth smallest and 0 based. 1st largest in len = 1 is 0th smallest
        l, r, kth = 0, len(nums) - 1, len(nums) - k
        while True:
            pivot = partition(l, r)
            if pivot == kth:
                return nums[pivot]
            elif pivot < kth:
                l = pivot + 1
            else:
                r = pivot - 1
