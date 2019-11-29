from typing import List
import collections
import bisect
import heapq

class Solution:
    def numKLenSubstrNoRepeats1100(self, S: str, K: int) -> int:
        # always keep a sliding window with no repeating letters, and compare the length of this window to k each time
        if not S or K <= 0 or len(S) < K:
            return 0
        hs = set()
        res = i = 0
        for j in range(len(S)):
            while S[j] in hs:
                hs.remove(S[i])
                i += 1
            hs.add(S[j])
            res += j - i + 1 >= K
        return res

    def minSubArrayLen209(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0
        minv = len(nums) + 1 # for no viable solution
        sumv = i = 0
        for j in range(len(nums)):
            sumv += nums[j]
            while i < j and sumv - nums[i] >= s:
                sumv -= nums[i]
                i += 1
            if sumv >= s:
                minv = min(minv, j - i + 1)
        return minv if minv <= len(nums) else 0 # note there might not be a solution

    def minWindow76(self, s: str, t: str) -> str:
        if not s or not t:
            return ''
        c = collections.Counter(t)
        cnt = i = 0
        minv = len(s) + 1
        res = ''
        for j in range(len(s)):
            if s[j] not in c:
                continue
            c[s[j]] -= 1
            if c[s[j]] >= 0:
                cnt += 1
            while cnt == len(t) and s[i] not in c or c[s[i]] < 0:
                if s[i] in c:
                    c[s[i]] += 1
                i += 1

            if cnt == len(t) and j - i + 1 < minv:
                minv = j - i + 1
                res = s[i: j + 1]

        return res if minv <= len(s) else ''

    def maxSlidingWindow239(self, nums: List[int], k: int) -> List[int]:
        if not nums or k < 1:
            return []
        q = collections.deque()
        i = 0
        res = []
        for j in range(len(nums)):
            # while q and nums[j] > q[0]: pop from the right to keep a descending queue
            while q and nums[j] > q[-1]:
                q.pop()
            q.append(nums[j])
            if k > 1:
                k -= 1
            else:
                res.append(q[0])
                if nums[i] == q[0]:
                    q.popleft()
                i += 1
        return res

    def medianSlidingWindow480(self, nums: List[int], k: int) -> List[float]:
        if not nums or k <= 1:
            return nums
        win = sorted(nums[:k - 1])
        i = 0
        res = []
        for j in range(k - 1, len(nums)):
            bisect.insort(win, nums[j])
            res.append((win[k // 2] + win[(k - 1) // 2]) / 2)
            win.remove(nums[i])
            i += 1
        return res

    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        '''
        Think this problem as a bunch of segments, representing roofs. Each start and each end represent an interval
        We keep a max heap of these segments (height, right). And we use a res to keep result (start, height)

            ---

          --------

        -------X-----
        ----------X-----
        Use an infinity line at h = 0 from start = 0 and right = inf to prevent heap is emtpy
        :param buildings:
        :return:
        '''
        segments = [(l, -h, r) for l, r, h in buildings] # start early first, end early first
        segments += list({(r, 0, None) for _, r, _ in buildings}) # add right for sentinel for checkpoint
        res = [[0, 0]] # (start, height)
        heap = [(0, float('inf'))] # (height, right)

        for s, negH, r in sorted(segments):
            if negH:
                heapq.heappush(heap, (negH, r))
            while s >= heap[0][1]: # for every new points, check the highest segments have ended
                heapq.heappop(heap)
            if res[-1][1] + heap[0][0] != 0: # the new highest height now is different from last result
                res.append([s, -heap[0][0]])
        return res[1:]














