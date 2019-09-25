from typing import List
import collections


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class HashTable:
    def __init__(self, size):
        self.hm = [[] for _ in range(size)]

    def put(self, key, value):
        hash_key = hash(key) % len(self.hm)
        key_exists = False
        bucket = self.hm[hash_key]
        for i, (k, v) in enumerate(bucket):
            if key == k:
                key_exists = True
                bucket[i] = (key, value)
                break
        if not key_exists:
            bucket.append((key, value))

    def get(self, key):
        hash_key = hash(key) % len(self.hm)
        bucket = self.hm[hash_key]
        for i, (k, v) in enumerate(bucket):
            if key == k:
                return v

    def delete(self, key):
        hash_key = hash(key) % len(self.hm)
        bucket = self.hm[hash_key]
        for i, (k, v) in enumerate(bucket):
            if key == k:
                del bucket[i]
                break


class Solution:
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

    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        start = maxv = 0
        hm = {}

        for i in range(len(s)):
            if s[i] not in hm or hm[s[i]] < start:
                maxv = max(maxv, i - start + 1)
            else:
                start = hm[s[i]] + 1
            hm[s[i]] = i
        return maxv

    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        if not s or k < 1:
            return 0
        hm = {}
        left = maxv = 0
        for i in range(len(s)):
            hm[s[i]] = hm.get(s[i], 0) + 1
            while len(hm) > k:
                hm[s[left]] -= 1
                if hm[s[left]] == 0:
                    del hm[s[left]]
                left += 1
            maxv = max(maxv, i - left + 1)
        return maxv

    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # postorder dfs
        def dfs(s):
            while hm[s]:
                dfs(hm[s].pop())
            res.append(s)
        res = []
        hm = collections.defaultdict(list)
        #create graph using adjcency list

        for x, y in tickets:
            hm[x].append(y)
        # python does not have dict with key sorted, so need to sort ourselves
        for _, values in hm.items():
            values.sort(reverse=True) # smaller at the end. so when pop() out first

        dfs('JFK')
        return res[::-1]

    class Node:
        def __init__(self, val, next, random):
            self.val = val
            self.next = next
            self.random = random

    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return head
        cur = head

        # 1. copy 1-> 1'-> 2 -> 2'
        while cur:
            node = self.Node(cur.val, None, None)
            node.next = cur.next
            cur.next = node
            cur = cur.next.next
        # 2. connect random
        cur = head
        while cur:
            cur.next.random = cur.random.next if cur.random else None
            cur = cur.next.next
        # 3. split
        cur = head
        dummy = pre = self.Node(0, None, None)
        while cur:
            pre.next = cur.next
            pre = pre.next
            cur.next = cur.next.next
            cur = cur.next
        return dummy.next

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if not head:
            return head
        dummy = ListNode(0)
        dummy.next = head
        l = r = dummy
        while n > 0:
            r = r.next
            n -= 1
        while r and r.next:
            l = l.next
            r = r.next
        l.next = l.next.next
        return dummy.next

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0
        minv, res = prices[0], 0
        for i in range(1, len(prices)):
            res = max(res, prices[i] - minv)
            minv = min(minv, prices[i])
        return res

    def missingNumber(self, nums: List[int]) -> int:
        if not nums:
            return -1
        res = 0
        for i in range(len(nums)):
            res ^= (i + 1) ^ nums[i]
        return res

    # def isMatch(self, s: str, p: str) -> bool:

    def mergesort(self, nums: List[int]) -> List[int]:
        def sort(l, r):
            if l < r:
                m = l + ((r - l) >> 1)
                sort(l, m)
                sort(m + 1, r)
                merge(l, m, r)

        def merge(l, m, r):
            i, j, k = l, m + 1, 0
            while i <= m and j <= r:
                if nums[i] < nums[j]:
                    t[k] = nums[i]
                    i += 1
                else:
                    t[k] = nums[j]
                    j += 1
                k += 1
            while i <= m:
                t[k] = nums[i]
                k, i = k + 1, i + 1
            while j <= r:
                t[k] = nums[j]
                k, j = k + 1, j + 1
            nums[l:r+1] = t[0:k]

        if not nums:
            return nums
        t = [0 for _ in range(len(nums))]
        sort(0, len(nums) - 1)
        return nums

    def quicksort(self, nums: List[int]) -> List[int]:
        def sort(l, r):
            if l < r:
                p = partition(l, r)
                sort(l, p - 1)
                sort(p + 1, r)

        def partition(l, r) -> int:
            start, i, j = l, l + 1, r
            while True:
                while i <= j and nums[i] <= nums[start]:
                    i += 1
                while i <= j and nums[j] >= nums[start]:
                    j -= 1
                if i <= j:
                    nums[i], nums[j] = nums[j], nums[i]
                else:
                    nums[start], nums[j] = nums[j], nums[start]
                    break
            return j

        if not nums:
            return nums
        sort(0, len(nums) - 1)
        return nums

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def kth(nums1, i1, j1, nums2, i2, j2, k):
            l1 = j1 - i1 + 1
            l2 = j2 - i2 + 1
            if l1 > l2:
                return kth(nums2, i2, j2, nums1, i1, j1, k)
            if not l1:
                return nums2[i2 + k - 1]
            if k == 1:
                return min(nums1[i1], nums2[i2])
            halfk = min(k // 2, l1)
            offset1, offset2 = i1 + halfk - 1, i2 + k - halfk - 1
            if nums1[offset1] == nums2[offset2]:
                return nums1[offset1]
            elif nums1[offset1] < nums2[offset2]:
                return kth(nums1, offset1 + 1, j1, nums2, i2, offset2, k - halfk)
            else:
                return kth(nums1, i1, offset1, nums2, offset2 + 1, j2, halfk)

        m, n = len(nums1), len(nums2)
        if (m + n) % 2:
            return kth(nums1, 0, m - 1, nums2, 0, n - 1, (m + n) // 2 + 1)
        else:
            return (kth(nums1, 0, m - 1, nums2, 0, n - 1, (m + n) // 2) + kth(nums1, 0, m - 1, nums2, 0, n - 1, (m + n) // 2 + 1)) / 2

    def maxProfit3(self, prices: List[int]) -> int:
        if not prices:
            return 0
        buy1 = buy2 = float('-inf')
        sell1 = sell2 = 0
        for p in prices:
            buy1 = max(buy1, -p)
            sell1 = max(sell1, p + buy1)
            buy2 = max(buy2, sell1 - p)
            sell2 = max(sell2, buy2 + p)
        return sell2

    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        res = []
        if not A or not B:
            return res
        i = j = 0
        while i < len(A) and j < len(B):
            l = max(A[i][0], B[j][0])
            r = min(A[i][1], B[j][1])
            if l <= r:
                res.append([l, r])
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return res




if __name__ == "__main__":
    s = Solution()
    nums = [2,6,-10,9, 100, -2, 8, 13]
    # print(s.mergesort(nums))
    # print(s.quicksort(nums))

    hm = HashTable(10)
    for x in nums:
        hm.put(x, x)
    for x in nums:
        if hm.get(x) != x:
            print('err')
    hm.delete(nums[0])
    print(hm.get(nums[0]))

