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


class KthLargest703:
    # minheap to store the top k largest
    def __init__(self, k: int, nums: List[int]):
        self.heap = nums
        self.k = k
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heappushpop(self.heap, val)
        return self.heap[0]


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

    def kSmallestPairs373(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        def push(i, j):
            if i < len(nums1) and j < len(nums2):
                heapq.heappush(heap, (nums1[i] + nums2[j], i, j))
        res = []
        if not nums1 or not nums2 or k <= 0:
            return res

        heap = []
        push(0, 0)
        while heap and k > 0:
            v, i, j = heapq.heappop(heap)
            res.append([nums1[i], nums2[j]])
            k -= 1
            push(i, j + 1)
            if j == 0:
                push(i + 1, j)
        return res

    def kthSmallest378(self, matrix: List[List[int]], k: int) -> int:
        def push(i, j):
            if i < len(matrix) and j < len(matrix[0]):
                heapq.heappush(heap, (matrix[i][j], i, j))
        # same thought as merge k sorted list
        if not matrix or not matrix[0] or k <= 0:
            return -1
        heap = []
        push(0, 0)
        v = 0
        while heap and k > 0:
            v, x, y = heapq.heappop(heap)
            k -= 1
            push(x, y + 1)
            if y == 0:
                push(x + 1, y)
        return -1 if k > 0 else v

    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        if not points or K <= 0:
            return []
        return heapq.nsmallest(K, points, key=lambda l: l[0] ** 2 + l[1] ** 2)

    def lastStoneWeight1046(self, stones: List[int]) -> int:
        if not stones:
            return 0
        heap = [-x for x in stones]
        heapq.heapify(heap)
        while len(heap) > 1:
            y, x = heapq.heappop(heap), heapq.heappop(heap)
            if y != x:
                heapq.heappush(heap, y - x)
        return -heapq.heappop(heap) if heap else 0


    def findCheapestPrice787(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        # first create the graph, then do dijkstra. the visited array need to store a pair(node, k)
        # reason: 0 -> 1 -> 2 -> 5
        #           ------->
        # if we limit k = 2 from 0 to 5. so even if 0 -> 1 -> 2 may produce smaller sum, we still need to do 0 -> 2
        # that may be larger, so we can finish within k = 2
        graph = collections.defaultdict(list)
        for s, e, l in flights:
            graph[s].append((e, l))
        heap = [(0, src, K + 1)]
        while heap:
            sumv, s, k = heapq.heappop(heap)
            if s == dst:
                return sumv
            if k > 0:
                for e, l in graph[s]:
                    heapq.heappush(heap, (sumv + l, e, k - 1))
        return -1

    def assignBikes1057(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
        if not workers or not bikes:
            return []
        # use a heap and throw in the nearest pair of bike/worker. the order we compare is
        # (distance, worker, bike).  and like merge k sorted ll, if bike is taken, pop next nearest

        hm = collections.defaultdict(list)
        for i, w in enumerate(workers):
            for j, b in enumerate(bikes):
                dist = abs(w[0] - b[0]) + abs(w[1] - b[1])
                hm[i].append((dist, i, j))
            hm[i].sort(reverse=True)
        heap = [hm[i].pop() for i in hm]
        heapq.heapify(heap)
        res = [-1] * len(workers)
        used = set()
        while len(used) < len(res):
            _, i, j = heapq.heappop(heap)
            if j not in used:
                res[i] = j
                used.add(j)
            else:
                heapq.heappush(heap, hm[i].pop())
        return res











