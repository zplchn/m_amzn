import collections
import heapq
from typing import List


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.minheap = []
        self.maxheap = []

    def addNum(self, num: int) -> None:
        x = heapq.heappushpop(self.minheap, num)
        heapq.heappush(self.maxheap, -x)
        if len(self.minheap) < len(self.maxheap):
            heapq.heappush(self.minheap, -heapq.heappop(self.maxheap))

    def findMedian(self) -> float:
        return self.minheap[0] if len(self.minheap) > len(self.maxheap) \
            else (self.minheap[0] - self.maxheap[0]) / 2


class Solution:
    def topKFrequentWords(self, words, k):
        if not words or k <= 0:
            return []
        c = collections.Counter(words)
        heap = [(-v, w) for w, v in c.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(k)]

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        ListNode.__eq__ = lambda x, y: x.val == y.val
        ListNode.__lt__ = lambda x, y: x.val < y.val

        if not lists:
            return None
        heap = [l for l in lists if l]
        heapq.heapify(heap)
        dummy = cur = ListNode(0)

        while heap:
            node = heapq.heappop(heap)
            cur.next = node
            cur = cur.next
            if node.next:
                heapq.heappush(heap, node.next)
        return dummy.next

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        if not nums or k < 1:
            return []
        c = collections.Counter(nums)
        return [k for k, _ in c.most_common(k)]

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        minheap = []
        for iv in intervals:
            if minheap and minheap[0] <= iv[0]:
                heapq.heappop(minheap)
            heapq.heappush(minheap, iv[1])
        return len(minheap)


