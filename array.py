from typing import List
import collections

class Solution:

    class MovingAverage:

        def __init__(self, size: int):
            self.size = size
            self.sum = 0
            self.q = collections.deque()

        def next(self, val: int) -> float:
            if len(self.q) < self.size:
                self.sum += val
                self.q.append(val)
            else:
                self.sum -= self.q.popleft()
                self.sum += val
                self.q.append(val)
            return self.sum // len(self.q)

    class MyQueue:

        def __init__(self):
            """
            Initialize your data structure here.
            """
            self.st1 = []
            self.st2 = []

        def push(self, x: int) -> None:
            """
            Push element x to the back of queue. O(N)
            """
            while self.st2:
                self.st1.append(self.st2.pop())
            self.st1.append(x)
            while self.st1:
                self.st2.append(self.st1.pop())

        def pop(self) -> int:
            """
            Removes the element from in front of queue and returns that element.
            """
            return self.st2.pop()

        def peek(self) -> int:
            """
            Get the front element.
            """
            return self.st2[-1]

        def empty(self) -> bool:
            """
            Returns whether the queue is empty.
            """
            return not self.st2

    class MyStack:

        def __init__(self):
            """
            Initialize your data structure here.
            """
            self.q1 = collections.deque()
            self.q2 = collections.deque()

        def push(self, x: int) -> None:
            """
            Push element x onto stack.
            """
            self.q2.append(x)
            while self.q1:
                self.q2.append(self.q1.popleft())
            self.q1, self.q2 = self.q2, self.q1


        def pop(self) -> int:
            """
            Removes the element on top of the stack and returns that element.
            """
            return self.q1.popleft()

        def top(self) -> int:
            """
            Get the top element.
            """
            return self.q1[0]

        def empty(self) -> bool:
            """
            Returns whether the stack is empty.
            """
            return not self.q1

    class Logger:

        def __init__(self):
            """
            Initialize your data structure here.
            """
            self.hm = {}

        def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
            """
            Returns true if the message should be printed in the given timestamp, otherwise returns false.
            If this method returns false, the message will not be printed.
            The timestamp is in seconds granularity.
            """
            if message in self.hm and timestamp - self.hm[message] < 10:
                return False
            self.hm[message] = timestamp
            return True

    class TwoSum:

        def __init__(self):
            """
            Initialize your data structure here.
            """
            self.hm = collections.Counter()

        def add(self, number: int) -> None:
            """
            Add the number to an internal data structure..
            """
            self.hm.update([number]) #update must be a iterable

        def find(self, value: int) -> bool:
            """
            Find if there exists any pair of numbers which sum is equal to the value.
            """
            for x in self.hm:
                y = value - x
                if x != y and y in self.hm or x == y and self.hm[x] > 1: # hm[] for nonexist key will not throw
                    return True
            return False

    class MyCircularQueue:
        # tail always be the next insertion point, head always the head. circular just need to mod the size and
        # everything else would be the same as non-circular
        def __init__(self, k: int):
            """
            Initialize your data structure here. Set the size of the queue to be k.
            """
            self.k = k
            self.size = 0
            # self.q = [0 * k] Cannot do this way because k is a int
            self.q = [0 for _ in range(k)]
            self.head = self.tail = 0

        def enQueue(self, value: int) -> bool:
            """
            Insert an element into the circular queue. Return true if the operation is successful.
            """
            if self.isFull():
                return False
            self.q[self.tail] = value
            self.tail = (self.tail + 1) % self.k
            self.size += 1
            return True

        def deQueue(self) -> bool:
            """
            Delete an element from the circular queue. Return true if the operation is successful.
            """
            if self.isEmpty():
                return False
            self.head = (self.head + 1) % self.k
            self.size -= 1
            return True

        def Front(self) -> int:
            """
            Get the front item from the queue.
            """
            return self.q[self.head] if not self.isEmpty() else -1

        def Rear(self) -> int:
            """
            Get the last item from the queue.
            """
            return self.q[self.tail - 1] if not self.isEmpty() else -1

        def isEmpty(self) -> bool:
            """
            Checks whether the circular queue is empty or not.
            """
            return self.size == 0

        def isFull(self) -> bool:
            """
            Checks whether the circular queue is full or not.
            """
            return self.size == self.k




    def fourSum(self, numbers, target):
        res = []
        if len(numbers) < 4:
            return res
        numbers.sort()
        for i in range(len(numbers) - 3):
            if i > 0 and numbers[i] == numbers[i-1]:
                continue
            for j in range(i + 1, len(numbers) - 2):
                if j > i + 1 and numbers[j] == numbers[j - 1]:
                    continue
                k, l = j + 1, len(numbers) - 1
                while k < l:
                    sum = numbers[i] + numbers[j] + numbers[k] + numbers[l]
                    if sum < target:
                        k += 1
                    elif sum > target:
                        l -= 1
                    else:
                        res.append([numbers[i], numbers[j], numbers[k], numbers[l]])
                        k, l = k + 1, l - 1
                        while k < l and numbers[k] == numbers[k-1]:
                            k += 1
                        while k < l and numbers[l] == numbers[l+1]:
                            l -= 1
        return res

    def trapRainWater(self, heights):
        if len(heights) < 3:
            return 0
        l, r = 0, len(heights) - 1
        res = 0
        while l < r:
            minh = min(heights[l], heights[r])
            if minh == heights[l]:
                while l < r and heights[l] <= minh:
                    res += minh - heights[l]
                    l += 1
            else:
                while l < r and heights[r] <= minh:
                    res += minh - heights[r]
                    r -= 1
        return res

    def islandPerimeter(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    res += (1 if j == 0 or grid[i][j-1] == 0 else 0) + \
                    (1 if j == len(grid[0]) - 1 or grid[i][j+1] == 0 else 0) + \
                    (1 if i == 0 or grid[i-1][j] == 0 else 0) + \
                    (1 if i == len(grid) - 1 or grid[i+1][j] == 0 else 0)
        return res

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        res = []
        if not intervals:
            return res
        intervals.sort()
        res.append(intervals[0])
        for i in range(1, len(intervals)):
            if intervals[i][0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                res.append(intervals[i])
        return res

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0
        minv, res = prices[0], 0
        for i in range(1, len(prices)):
            res = max(res, prices[i] - minv)
            minv = min(minv, prices[i])
        return res

    def majorityElement(self, nums: List[int]) -> int:
        if not nums:
            return -1
        res,cnt = -1, 0
        for x in nums:
            if x == res:
                cnt += 1
            elif cnt == 0:
                res, cnt = x, 1
            else:
                cnt -= 1
        return res

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        if not nums:
            return nums
        res = [0] * len(nums)
        res[0] = 1
        for i in range(1, len(nums)):
            res[i] = res[i - 1] * nums[i - 1]
        right = 1
        for i in range(len(nums) - 2, -1, -1):
            res[i] *= right * nums[i + 1]
            right *= nums[i + 1]
        return res

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        if len(nums) < 3:
            return -1
        nums.sort()
        delta = float('inf')
        res = 0
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            j, k = i + 1, len(nums) - 1
            while j < k:
                sum = nums[i] + nums[j] + nums[k]
                if sum == target:
                    return target
                if abs(sum - target) < delta:
                    delta = abs(sum - target)
                    res = sum
                if sum < target:
                    j += 1
                else:
                    k -= 1
        return res

    def summaryRanges(self, nums: List[int]) -> List[str]:
        res = []
        if not nums:
            return res
        i = j = 0
        while j < len(nums):
            j = i + 1
            while j < len(nums) and nums[j] == nums[j-1] + 1:
                j += 1
            res.append(str(nums[i]) + '->' + str(nums[j-1]))
            i = j
        return res

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        # if not intervals:
        #     return intervals If given is empty, still can insert!
        res = []
        i = 0
        while i < len(intervals) and intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1
        while i < len(intervals) and newInterval[1] >= intervals[i][0]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        res.append(newInterval)
        res.extend(intervals[i:])
        return res

    def plusOne(self, digits: List[int]) -> List[int]:
        if not digits:
            return digits
        i = len(digits) - 1
        while i >= 0 and digits[i] == 9:
            digits[i] = 0
            i -= 1
        if i < 0:
            return [1] + [0] * len(digits)
        digits[i] += 1
        return digits

    def singleNumber(self, nums: List[int]) -> int:
        if not nums:
            return 0
        res = 0
        for i in range(32):
            mask = 1 << i
            cnt = 0
            for x in nums:
                if x & mask:
                    cnt += 1
            if cnt % 3:
                res |= mask
        if res >= 2 ** 31: # python int is infinity large, so need to check if > 2**31. then clear the first bit and neg
            res -= (1 << 32)
        return res

    def maxArea(self, height: List[int]) -> int:
        if len(height) < 2:
            return 0
        res, i, j = 0, 0, len(height) - 1
        while i < j:
            minv = min(height[i], height[j])
            res = max(res, minv * (j - i))
            if minv == height[i]:
                i += 1
            else:
                j -= 1
        return res

    def sortColors(self, nums: List[int]) -> None:
        if not nums:
            return
        i0 = i1 = 0
        i2 = len(nums) - 1
        while i1 <= i2:
            if nums[i1] == 0:
                nums[i0], nums[i1] = nums[i1], nums[i0]
                i0 += 1
                i1 += 1
            elif nums[i1] == 2:
                nums[i1], nums[i2] = nums[i2], nums[i1]
                i2 -= 1
            else:
                i1 += 1

    def missingNumber(self, nums: List[int]) -> int:
        if not nums:
            return -1
        res = 0
        for i in range(len(nums)):
            res ^= (i + 1) ^ nums[i]
        return res

    def findMin(self, nums: List[int]) -> int:
        if not nums:
            return - 1
        l, r = 0, len(nums) - 1
        while l < r:
            m = l + ((r - l) >> 1)
            if nums[m] < nums[r]:
                r = m
            else:
                l = m + 1
        return nums[l] # return the element not pivot

    def findMin2(self, nums: List[int]) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l < r:
            m = l + ((r - l) >> 1)
            if nums[m] == nums[r]:
                r -= 1
            elif nums[m] < nums[r]:
                r = m
            else:
                l = m + 1
        return nums[l]

    def majorityElement2(self, nums: List[int]) -> List[int]:
        res = []
        if not nums:
            return res
        c1, n1, c2, n2 = None, 0, None, 0
        for x in nums:
            if x == c1: # these should be first as [2,2,2] both c1 and c2 will be inited with 2
                n1 += 1
            elif x == c2:
                n2 += 1
            elif n1 == 0:
                c1, n1 = x, 1
            elif n2 == 0:
                c2, n2 = x, 1
            else:
                n1 -= 1
                n2 -= 1
        n1 = n2 = 0
        for x in nums:
            n1 += 1 if x == c1 else 0
            n2 += 1 if x == c2 else 0
        if n1 > len(nums) // 3: # this cannot have >= [1,2,2] the 1 will be outputed
            res.append(c1)
        if n2 > len(nums) // 3:
            res.append(c2)
        return res

    def findPeakElement(self, nums: List[int]) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r - l) >> 1)
            if (m == 0 or nums[m] > nums[m - 1]) and (m == len(nums) - 1 or nums[m] > nums[m + 1]):
                return m
            elif nums[m] < nums[m + 1]:
                l = m + 1
            else:
                r = m - 1
        return -1

    def peakIndexInMountainArray(self, A: List[int]) -> int:
        if len(A) < 3:
            return -1
        l, r = 0, len(A) - 1
        while l <= r:
            m = l + ((r - l) >> 1)
            if A[m - 1] < A[m] > A[m + 1]:
                return m
            elif A[m] < A[m + 1]:
                l = m + 1
            else:
                r = m - 1
        return -1

    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if not nums1 or not nums2:
            return []
        s = set()
        nums1.sort()
        nums2.sort()
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                s.add(nums1[i])
                i, j = i + 1, j + 1
        return list(s)

    def evalRPN(self, tokens: List[str]) -> int:
        if not tokens:
            return 0
        st = []
        for t in tokens:
            if t not in '+-*/': # str.isdigit() returns false for negative number
                st.append(int(t))
            else:
                y = st.pop()
                x = st.pop()
                if t == '+':
                    st.append(x + y)
                elif t == '-':
                    st.append(x - y)
                elif t == '*':
                    st.append(x * y)
                else: # -10 // 3 == -4 ; -10 // -3 == 3. if cannot full divide, always go to smaller
                    st.append(-(-x // y) if x * y < 0 else x // y)
        return st.pop()

    def findCelebrity(self, n):
        def knows(a, b):
            return False
        # suppose someone at k is the celeb. we loop i through from 1 -> k, and check a suspect knows i
        # if knows(suspect, i) means i maybe a suspect we update it. otherwise it means i must not because a suspect
        # not know it. so once we update to k, we eliminate everyone before it. and since we do not update k anymore,
        # it means k does not know anyone after it. then we do a second pass and verify people before k

        celeb = 0
        for i in range(1, n):
            if knows(celeb, i):
                celeb = i
        for i in range(n):
            if i != celeb and knows(celeb, i) or not knows(i, celeb):
                return -1
        return celeb

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if not gas or len(gas) != len(cost):
            return -1
        start = sum = total = 0
        for i in range(len(gas)):
            sum += gas[i] - cost[i]
            total += gas[i] - cost[i]
            if sum < 0:
                start = i + 1
                sum = 0
        return start if total >= 0 else -1

    def checkValidString(self, s: str) -> bool:
        if not s:
            return True
        # swipe twice, first time left to right, and all * count as (. use a cnt, ( + 1, ) -1. any time cnt < 0 means
            # ) is more so return false. after finish, means left (plus converted *) is at least more than ). Second
            # pass right to left, and treat all * as ). anytime cnt < 0 means left is more. *((). once done,
            # means right is at least more than left. So there must exist a solution.
        cnt = 0
        for i in range(len(s)):
            if s[i] == '(' or s[i] == '*':
                cnt += 1
            else:
                cnt -= 1
            if cnt < 0:
                return False
        if cnt == 0:
            return True
        cnt = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == ')' or s[i] == '*':
                cnt += 1
            else:
                cnt -= 1
            if cnt < 0:
                return False
        return True






















                                                                                                                                                                                                                                                                               gri)

