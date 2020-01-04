from typing import List


class Solution:
    def twoSumLessThanK1099(self, A: List[int], K: int) -> int:
        if len(A) < 2:
            return -1
        i, j = 0, len(A) - 1
        maxv = -1
        A.sort()
        while i < j:
            m = A[i] + A[j]
            if m >= K:
                j -= 1
            else:
                maxv = max(maxv, m)
                i += 1
        return maxv

    def sortArrayByParity905(self, A: List[int]) -> List[int]:
        i, j = 0, len(A) - 1
        while i < j:
            while i < j and A[i] % 2 == 0:
                i += 1
            while i < j and A[j] % 2:
                j -= 1
            A[i], A[j] = A[j], A[i]
            i, j = i + 1, j - 1
        return A

    def triangleNumber611(self, nums: List[int]) -> int:
        # like 3sum. each time we set outer loop from large -> small, then two pointers on left. For making a
        # triagle, need two edges sum > third edge.
        if len(nums) < 3:
            return 0
        nums.sort()
        res = 0
        for i in reversed(range(2, len(nums))):
            j, k = 0, i - 1
            while j < k:
                if nums[j] + nums[k] > nums[i]:
                    res += k - j
                    k -= 1
                else:
                    j += 1
        return res

