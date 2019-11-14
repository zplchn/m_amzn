from typing import List
import collections
import random


class Solution384:

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.original = nums[::]

    def reset(self) -> List[int]:
        """
        Resets the array to its original configuration and return it.
        """
        self.nums = self.original[::]
        return self.nums

    def shuffle(self) -> List[int]:
        """
        Returns a random shuffling of the array.
        """
        # iterate the array and for every num swap with [current(included), end] by random
        for i in range(len(self.nums)):
            j = random.randrange(i, len(self.nums))
            # random.randrange(start, stop) returns a random number [start, stop)
            self.nums[i], self.nums[j] = self.nums[j], self.nums[i]
        return self.nums

class Solution:

    class MovingAverage:

        def __init__(self, size: int):
            self.size = size
            self.sum = 0
            self.q = collections.deque()

        def next(self, val: int) -> float:
            if len(self.q) ==  self.size:
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

    class MaxStack:

        def __init__(self):
            """
            initialize your data structure here.
            """
            self.st = []
            self.maxst = []

        def push(self, x: int) -> None:
            self.st.append(x)
            if not self.maxst or x >= self.maxst[-1]:
                self.maxst.append(x)

        def pop(self) -> int:
            x = self.st.pop()
            if x == self.maxst[-1]:
                self.maxst.pop()
            return x

        def top(self) -> int:
            return self.st[-1]

        def peekMax(self) -> int:
            return self.maxst[-1]

        def popMax(self) -> int:
            x = self.maxst.pop()
            tst = []
            while self.top() != x:
                tst.append(self.st.pop())
            # self.pop() CANNOT USE self.pop() because maxst has already been poped!!!!
            self.st.pop()
            while tst:
                self.push(tst.pop()) # need to call push() because if [5,1] when 5 is out, need to recalculate max is 1
            return x

    # class NestedInteger(object):
    #    def isInteger(self):
    #        """
    #        @return True if this NestedInteger holds a single integer, rather than a nested list.
    #        :rtype bool
    #        """
    #
    #    def getInteger(self):
    #        """
    #        @return the single integer that this NestedInteger holds, if it holds a single integer
    #        Return None if this NestedInteger holds a nested list
    #        :rtype int
    #        """
    #
    #    def getList(self):
    #        """
    #        @return the nested list that this NestedInteger holds, if it holds a nested list
    #        Return None if this NestedInteger holds a single integer
    #        :rtype List[NestedInteger]
    #        """

    class NestedIterator(object):

        def __init__(self, nestedList):
            """
            Initialize your data structure here.
            :type nestedList: List[NestedInteger]
            """
            self.st = list(reversed(nestedList))

        def next(self):
            """
            :rtype: int
            """
            return self.st.pop()

        def hasNext(self):
            """
            :rtype: bool
            """
            while self.st:
                if self.st[-1].isInteger():
                    return True
                t = reversed(self.st.pop().getList())
                self.st.extend(t)
            return False

    class Vector2D:

        def __init__(self, v: List[List[int]]):
            self.v = v
            self.r = self.c = 0

        def next(self) -> int:
            x = self.v[self.r][self.c]
            self.c += 1
            return x

        def hasNext(self) -> bool:
            while self.r < len(self.v):
                if self.c < len(self.v[self.r]):
                    return True
                else:
                    self.r, self.c = self.r + 1, 0
            return False

    class ZigzagIterator(object):

        def __init__(self, v1, v2):
            """
            Initialize your data structure here.
            :type v1: List[int]
            :type v2: List[int]
            """
            self.vecs = [v1, v2]
            self.q = collections.deque()
            # iter1, iter2 = iter(v1), iter(v2) Python iterator only has one function that is next()
            if v1:
                self.q.append((0, 0))
            if v2:
                self.q.append((1, 0))


        def next(self):
            """
            :rtype: int
            """
            v, c = self.q.popleft()
            x = self.vecs[v][c]
            c += 1
            if c < len(self.vecs[v]):
                self.q.append((v, c))
            return x

        def hasNext(self):
            """
            :rtype: bool
            """
            return len(self.q) > 0

    class MyHashMap:
        # a number % 1000 and // 1000 then can uniquely map the number itself

        def __init__(self):
            """
            Initialize your data structure here.
            """
            self.hm = [[] for _ in range(1000)]

        def put(self, key: int, value: int) -> None:
            """
            value will always be non-negative.
            """
            hashkey = key % 1000
            if not self.hm[hashkey]:
                self.hm[hashkey] = [-1 for _ in range(1000)]
            self.hm[hashkey][key // 1000] = value

        def get(self, key: int) -> int:
            """
            Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
            """
            hashkey = key % 1000
            return self.hm[hashkey][key // 1000] if self.hm[hashkey] else -1

        def remove(self, key: int) -> None:
            """
            Removes the mapping of the specified value key if this map contains a mapping for the key
            """
            hashkey = key % 1000
            if self.hm[hashkey]:
                self.hm[hashkey][key // 1000] = -1

    class TicTacToe:

        def __init__(self, n: int):
            """
            Initialize your data structure here.
            """
            self.row = [0 for _ in range(n)]
            self.col = [0 for _ in range(n)]
            self.diag = self.revdiag = 0
            self.n = n

        def move(self, row: int, col: int, player: int) -> int:
            """
            Player {player} makes a move at ({row}, {col}).
            @param row The row of the board.
            @param col The column of the board.
            @param player The player, can be either 1 or 2.
            @return The current winning condition, can be either:
                    0: No one wins.
                    1: Player 1 wins.
                    2: Player 2 wins.
            """
            x = 1 if player == 1 else -1
            self.row[row] += x
            self.col[col] += x
            self.diag += x if row == col else 0
            self.revdiag += x if row + col == self.n else 0
            return abs(self.row[row]) == self.n or abs(self.col[col]) == self.n or abs(self.diag) == self.n or abs(
                    self.revdiag) == self.n


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

    def checkValidString678(self, s: str) -> bool:
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

    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:
        def summary(l, r):
            l, r = l + 1, r - 1
            if l == r:
                return str(l)
            elif l < r:
                return str(l) + '->' + str(r)
            else:
                return None

        res = []
        if not nums:
            res.append(summary(lower - 1, upper + 1))
        else:
            res.append(summary(lower - 1, nums[0]))
            for i in range(1, len(nums)):
                res.append(summary(nums[i-1], nums[i]))
            res.append(summary(nums[-1], upper + 1))
        return list(filter(lambda x : x is not None, res))

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if not nums1 or not nums2:
            return []
        # use a stack to store a descending array and pop when meet a larger number
        # hm to map the k -> larger and at last loop nums1 to map k to v in hm
        st = []
        hm = {}
        for x in nums2:
            while st and x > st[-1]:
                hm[st.pop()] = x
            st.append(x)
        return [hm.get(k, -1) for k in nums1]

    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        #loop twice so solve the circular problem. and st store the index and only when index < n.
        #second round only for loop up
        if not nums:
            return nums
        res = [-1 for _ in range(len(nums))]
        st = []
        for i in range(len(nums) * 2):
            while st and nums[st[-1]] < nums[i % len(nums)]:
                    res[st.pop()] = nums[i % len(nums)]
            if i < len(nums):
                st.append(i)
        return res

    def nextGreaterElement3(self, n: int) -> int:
        # this is exactly next permutation
        ca = list(str(n))
        i = len(ca) - 2
        while i >= 0 and ca[i] >= ca[i + 1]:
            i -= 1
        if i < 0:
            return -1
        j = len(ca) - 1
        while ca[j] <= ca[i]:
            j -= 1
        ca[i], ca[j] = ca[j], ca[i]
        ca[i + 1:] = ca[len(ca) - 1:i:-1]
        x = ''.join(ca)
        return int(x) if int(x) <= 2 ** 31 - 1 else -1

    def subarraySum560(self, nums: List[int], k: int) -> int:
        if not nums:
            return 0
        # nagetive possible so there will be multiple local sum intervals
        hm = {}
        hm[0] = 1
        res, sum = 0, 0
        for x in nums:
            sum += x
            # the add to hs need to happen after the check. because when k == 0, if add first, it will be true
            res += hm.get(sum - k, 0)
            # hm[sum] += 1
            hm[sum] = hm.get(sum, 0) + 1
        return res

    def maximumProduct(self, nums: List[int]) -> int:
        # there are only two possibilities, the biggest 3 or 2 negs * biggest postive. Just need to compare two cases.
        if len(nums) < 3:
            return 0
        nums.sort()
        return max(nums[0] * nums[1] * nums[-1], nums[-3] * nums[-2] * nums[-1])

    def nextPermutation(self, nums: List[int]) -> None:
        if not nums:
            return
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i < 0:
            nums[:] = nums[::-1]
            return
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
        nums[i+1:] = nums[-1:i:-1]

    def gameOfLife(self, board: List[List[int]]) -> None:
        if not board or not board[0]:
            return
        # 0: dead -> dead 1 : live -> live 2: live -> dead 3: dead -> live
        offsets = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        for i in range(len(board)):
            for j in range(len(board[0])):
                sumv = 0
                for o in offsets:
                    x, y = i + o[0], j + o[1]
                    if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] in [1, 2]: # need to check the
                        # used to live case!!
                        sumv += 1
                if board[i][j] == 1 and (sumv < 2 or sumv > 3):
                    board[i][j] = 2
                elif board[i][j] == 0 and sumv == 3:
                    board[i][j] = 3
        for i in range(len(board)):
            for j in range(len(board[0])):
                board[i][j] %= 2

    def rotate(self, nums: List[int], k: int) -> None:
        k %= len(nums)
        if k == 0:
            return # need this check
        nums[:-k] = nums[-k-1::-1]
        nums[-k:] = nums[:-k-1:-1]
        nums[:] = nums[::-1]

    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        res = []
        i = len(A) - 1
        carry = 0
        while i >= 0 or K > 0 or carry:
            sum = carry
            if i >= 0:
                sum += A[i]
                i -= 1
            if K > 0:
                sum += K % 10
                K //= 10
            res.append(sum % 10)
            carry = sum // 10
        return res[::-1]

    def generate(self, numRows: int) -> List[List[int]]:
        if numRows <= 0:
            return []
        res = [[1]]
        while numRows > 1:
            t = [1]
            for i in range(len(res[-1]) - 1):
                t.append(res[-1][i] + res[-1][i+1])
            t.append(1)
            numRows -= 1
            res.append(t)
        return res

    def searchMatrix(self, matrix, target):
        if not matrix or not matrix[0]:
            return False
        i, j = len(matrix) - 1, 0
        while i >= 0 and j < len(matrix[0]):
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] < target:
                j += 1
            else:
                i -= 1
        return False

    def wiggleSort(self, nums: List[int]) -> None:
        if not nums:
            return
        for i in range(1, len(nums)):
            if i % 2 == 1 and nums[i] < nums[i - 1] or i % 2 == 0 and nums[i] > nums[i - 1]:
                nums[i], nums[i - 1] = nums[i - 1], nums[i]

    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort()
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[i - 1][1]:
                return False
        return True

    def distributeCandies(self, candies: List[int]) -> int:
        hs = set(candies)
        return min(len(candies) // 2, len(hs))

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        hm = {c: i for i, c in enumerate(order)}
        for i in range(1, len(words)):
            minl = min(len(words[i-1]), len(words[i]))
            for j in range(minl):
                if words[i-1][j] != words[i][j]:
                    if hm[words[i-1][j]] > hm[words[i][j]]:
                        return False
                    break
            else:
                if len(words[i-1]) > len(words[i]):
                    return False
        return True

    def sortedSquares(self, A: List[int]) -> List[int]:
        if not A:
            return A
        if A[0] >= 0:
            return [a * a for a in A]
        i, j = 0, len(A) - 1
        res = []
        while i <= j:
            if abs(A[i]) < abs(A[j]):
                res.append(A[j] * A[j])
                j -= 1
            else:
                res.append(A[i] * A[i])
                i += 1
        return res[::-1]

    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        # greedy. insert highest people first, with order smaller in front. Then lower people, just insert at order
        # position, because lower people are invisible to higher people
        if not people:
            return people
        people.sort(key=lambda p: (-p[0], p[1]))
        res = []
        # python insert works for empty or shorter list, position will be ignored and just insert, no exception thrown
        for p in people:
            res.insert(p[1], p)
        return res

    def countBattleships(self, board: List[List[str]]) -> int:
        if not board or not board[0]:
            return 0
        res = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'X':
                    if i > 0 and board[i-1][j] == 'X' or j > 0 and board[i][j - 1] == 'X':
                        continue
                    res += 1
        return res

    def fizzBuzz(self, n: int) -> List[str]:
        res = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                res.append('FizzBuzz')
            elif i % 3 == 0:
                res.append('Fizz')
            elif i % 5 == 0:
                res.append('Buzz')
            else:
                res.append(str(i))
        return res

    def validTicTacToe(self, board: List[str]) -> bool:
        # invalid cases: 1. count O != count X or count X - 1
        # 2. X win and count O != count X -1
        # 3. O win and count X != count X

        cnt_x = sum(row.count('X') for row in board)
        cnt_o = sum(row.count('O') for row in board)

        def win(player):
            for i in range(3):
                # all row or all column
                if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
                    return True
            return all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3))

        if cnt_o not in {cnt_x, cnt_x - 1}:
            return False
        if win('X') and cnt_o != cnt_x - 1:
            return False
        if win('O') and cnt_o != cnt_x:
            return False
        return True

    def isHappy(self, n: int) -> bool:
        if n <= 0:
            return False
        hs = set()
        while n != 1:
            sum = 0
            while n != 0:
                sum += (n % 10) * (n % 10)
                n //= 10
            if sum in hs:
                return False
            hs.add(sum)
            n = sum
        return True

    def findDuplicates(self, nums: List[int]) -> List[int]:
        res = []
        if not nums:
            return res
        # use position as indicator since num are between 1 and N. set to negative for nums[nums[i] - 1]
        for i in range(len(nums)):
            idx = abs(nums[i]) - 1
            if nums[idx] < 0:
                res.append(idx + 1)
            nums[idx] = -nums[idx]
        return res

    def thirdMax(self, nums: List[int]) -> int:
        if not nums:
            return -1
        first = second = third = float('-inf')
        for x in nums:
            if x > first:
                third, second, first = second, first, x
            elif second < x < first:
                third, second = second, x
            elif third < x < second:
                third = x
        return first if third == float('-inf') else third

    def computeArea(self, A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> int:
        area1, area2 = (C - A) * (D - B), (G - E) * (H - F)
        # no intersection
        if E >= C or G <= A or F >= D or H <= B:
            return area1 + area2
        return area1 + area2 - (min(C, G) - max(E, A)) * (min(D, H) - max(B, F))

    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        return not (rec2[0] >= rec1[2] or rec2[2] <=rec1[0] or rec1[1] >= rec2[3] or rec1[3] <= rec2[1])

    def maxSubarraySumCircular(self, A: List[int]) -> int:
        # so the max will be either a normal max subarray in the middle; or the total - min subarray, which is part
        # head and part tail. Cornor case is when all neg, then the latter would be zero but we still return one element
        if not A:
            return 0
        lmax, lmin, maxv, minv, total = 0, 0, float('-inf'), float('inf'), 0
        for x in A:
            lmax = max(lmax + x, x)
            maxv = max(maxv, lmax)
            lmin = min(lmin + x, x)
            minv = min(lmin, minv)
            total += x
        return max(maxv, total - minv) if maxv >= 0 else maxv

    def sortArrayByParityII(self, A: List[int]) -> List[int]:
        # we only focus on the even and swap when it's not an even number
        if not A:
            return A
        o = 1
        for e in range(0, len(A), 2):
            if A[e] % 2 == 1:
                while A[o] % 2 == 1:
                    o += 2
                A[e], A[o] = A[o], A[e]
        return A


































                                                                                                                                                                                                                                                                               gri)

