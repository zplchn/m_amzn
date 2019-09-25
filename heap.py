import collections
import heapq
from typing import List


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


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


