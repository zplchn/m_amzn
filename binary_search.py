from typing import List
import collections
import bisect


class TimeMap981:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hm = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.hm[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.hm:
            return ''
        values = self.hm[key]
        l, r = 0, len(values) - 1
        if timestamp < values[l][0]:
            return ''
        elif timestamp >= values[r][0]:
            return values[r][1]

        while l < r:
            m = l + ((r - l) >> 1)
            if values[m][0] == timestamp:
                return values[m][1]
            elif values[m][0] < timestamp:
                l += 1
            else:
                r -= 1
        return values[r][1] if r >= 0 else '' # r could be -1



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

    def search702(self, reader, target):
        # we can think this problem is an array start from index 0 [1,2,3....end, INF, INF, INF] end somewhere and
        # then followed by all INF, and this array is still itself sorted. we just need to put the right boundary to
        # the max int as the return type is an int. However in python there is no limit to int, we can find the right
        # boundary first

        # l, r = 0, 2147483647
        l, r = 0, 1
        while reader.get(r) < target:
            r *= 2 # keep 2 exp so it's still logn
        while l <= r:
            m = l + ((r - l) // 2)
            x = reader.get(m)
            if x == target:
                return m
            elif target < x:
                r = m - 1
            else:
                l = m + 1
        return -1

    def findClosestElements658(self, arr: List[int], k: int, x: int) -> List[int]:
        # 3 cases: 1. x is < arr[0], so first k; 2. x > arr[-1], so arr[-k:] 3. x in the middle, bisect to find the
        # first element >= x. so take a window of k on the left of x, and a window of k on the right, shrink until = k
        # Two pointer should use the SHRINK strategy rather than expand, so edge cases(go beyong index) is handled first
        # O(logN + k)
        if not arr or k == 0:
            return []
        idx = bisect.bisect_left(arr, x)
        if idx == 0:
            return arr[:k]
        elif idx == len(arr):
            return arr[-k:]
        else:
            l, r = max(idx - k, 0), min(len(arr) - 1, idx + k - 1)
            while r - l + 1 > k:
                if abs(arr[l] - x) > abs(arr[r] - x):
                    l += 1
                else:
                    r -= 1
            return arr[l: r + 1]

    def guessNumber374(self, n: int) -> int:
        def guess(num: int) -> int:
            pass
        l, r = 1, n
        while l <= r:
            m = l + ((r - l) >> 1)
            if guess(m) == 0:
                return m
            elif guess(m) == -1:
                r = m - 1
            else:
                l = m + 1
        return -1

    def fixedPoint1064(self, A: List[int]) -> int:
        if not A:
            return -1
        l, r = 0, len(A) - 1
        while l <= r:
            m = l + ((r - l) >> 1)
            if A[m] < m:
                l = m + 1
            else:
                r = m - 1
        return l if A[l] == l < len(A) else -1 # we basically find the insertion place which [-10], [-10, 3] cases















































