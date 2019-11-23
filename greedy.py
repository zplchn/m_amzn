from typing import List
import heapq


class Solution:
    def connectSticks1167(self, sticks: List[int]) -> int:
        # [1,2,4, 5] -> [3, 4, 5] -> [7, 5] so the small values will keep be reused, so we should always pick the
        # smallest so we reuse as minimal as possible. greedy.
        if not sticks:
            return 0
        heapq.heapify(sticks)
        res = 0
        while len(sticks) >= 2:
            s1, s2 = heapq.heappop(sticks), heapq.heappop(sticks)
            res += s1 + s2
            heapq.heappush(sticks, s1 + s2)
        return res

    def maximumMinimumPath1102(self, A: List[List[int]]) -> int:
        # use bfs and control the order dequeue. the biggest number will be visited first by a shortest path
        # this guarantee a path with all max values possible
        if not A or not A[0]:
            return -1
        heap = [(-A[0][0], 0, 0)]
        maxv = -A[0][0]
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        visited = {(0, 0)}
        while heap:
            v, i, j = heapq.heappop(heap)
            maxv = max(maxv, v)
            if i == len(A) - 1 and j == len(A[0]) - 1:
                return -maxv
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(A) and 0 <= y < len(A[0]) and (x, y) not in visited:
                    visited.add((x, y))
                    heapq.heappush(heap, (-A[x][y], x, y))
        return -1

    def eraseOverlapIntervals435(self, intervals: List[List[int]]) -> int:
        # sort the intervals [1,3], [2,4] , [3,6] when there is an overlap, always keep the one ending is smaller
        # use a counter to record how many need to be deleted
        if not intervals:
            return 0
        last, res = 0, 0
        intervals.sort()
        for i in range(1, len(intervals)):
            if intervals[i][0] < intervals[last][1]:
                res += 1
                last = i if intervals[i][1] < intervals[last][1] else last
            else:
                last = i
        return res

    def findMinArrowShots452(self, points: List[List[int]]) -> int:
        # absort and update min last end if there is overlap, and advance if there is no overlap
        if not points:
            return 0
        points.sort()
        res = 1
        end = points[0][1]
        for i in range(1, len(points)):
            if points[i][0] <= end: # [1,2] and [2,3] need just one arrow, so need =
                end = min(end, points[i][1])
            else:
                end = points[i][1]
                res += 1
        return res


