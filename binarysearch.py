from typing import List

class Solution:
    def search1(self, A, target):
        if not A:
            return -1
        l, r = 0, len(A) - 1
        while l <= r:
            m = l + ((r - l) >> 1)
            if A[m] == target:
                return m
            elif A[m] < A[r]:
                if A[m] < target <= A[r]:
                    l = m + 1
                else:
                    r = m - 1
            else:
                if A[l] <= target < A[m]:
                    r = m - 1
                else:
                    l = m + 1
        return -1

    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r - l) >> 1)
            if nums[m] == target:
                return m
            elif nums[m] < nums[r]:
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
            else:
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
        return -1

    def search2(self, nums: List[int], target: int) -> bool:
        if not nums:
            return False
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r - l) >> 1)
            if nums[m] == target:
                return True
            if nums[m] < nums[r]:
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
            elif nums[m] > nums[r]:
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:
                r -= 1
        return False

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        res = [-1, -1]
        if not nums:
            return res
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r - l) >> 1)
            if nums[m] <= target:
                l = m + 1
            else:
                r = m - 1
        res[1] = r
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r - l) >> 1)
            if nums[m] >= target:
                r = m - 1
            else:
                l = m + 1
        res[0] = l
        return res if res[0] <= res[1] else [-1, -1]

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        l, r = 0, len(matrix) - 1
        while l <= r: # think there is one row so we need =
            m = l + ((r - l) >> 1)
            if matrix[m][-1] == target:
                return True
            elif matrix[m][-1] < target:
                l = m + 1
            else:
                r = m - 1
        if l >= len(matrix): # think one row
            return False
        ll, r = 0, len(matrix[0]) - 1
        while ll <= r:
            m = ll + ((r - ll) >> 1)
            if matrix[l][m] == target:
                return True
            elif matrix[l][m] < target:
                ll = m + 1
            else:
                r = m - 1
        return False











































