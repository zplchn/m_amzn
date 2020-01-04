from typing import List
import collections
import bisect
import heapq

'''
Sliding window 

For questions like: sliding window / minimum size subarray/substring that meet some condition
Note it can only be MINIMUM not maximum, which should use
    - stack ( maximum size parenthesis)
    - presum ( longest subarray sum equals k)
    - hashmap ( array degrees)

common routine: Use two pointers for both right (j) and left side (i)
Each loop:
1. Enter right side j
    - put into deque (sliding window max)
    - bisect insort (sliding window median)
    - decrease counter (sliding window cover substring)
    
2. While meet condition, shrink left side
    - condition could be: 
        - window size == k (j == k - 1)
        - counter == len(T) 
        - sum >= target (minimum window sum
        
3. Once the window being the minimum size, output the local result as one possible candidate
        

'''


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

    def longestRepeatingSubstring1062(self, S: str) -> int:
        # to find longest, we need to find an int between 1 and n - 1. so we can use binary search to try different n.
        # for find a repeating substring, we can use rolling hash. so it's o(n) to compare string match. despite hash
        # can have collision
        base = 26
        mod = 2 ** 24

        def search(k: int) -> bool:
            rh = 0 # rolling hash
            hs = set()
            lv = base ** k % mod
            i = 0
            for j in range(len(S)):
                if j < k:
                    rh = (rh * base + ord(S[j]) - ord('a')) % mod
                else:
                    hs.add(rh)
                    rh = (rh * base - (ord(S[i]) - ord('a')) * lv + ord(S[j]) - ord('a')) % mod
                    if rh in hs:
                        return True
                    i += 1
            return False

        if not S:
            return 0
        l, r = 1, len(S) - 1
        while l <= r:
            m = l + ((r - l) >> 1)
            if search(m):
                l = m + 1
            else:
                r = m - 1
        return r

    def longestDupSubstring1044(self, S: str) -> str:
        # same as 1062, but output the string instead of length. still use rolling hash
        if not S:
            return S
        # to find a repeating string, use a hashset and store rolling hash of substrings
        base = 26
        mod = 2 ** 40 # hash can collision, need to find a real big number here or verify result

        def search(m: int):
            rh = i = 0
            hs = set()
            lv = base ** m % mod
            for j in range(len(S)):
                if j < m:
                    rh = (rh * base + ord(S[j]) - ord('a')) % mod
                else:
                    hs.add(rh)
                    rh = (rh * base - (ord(S[i]) - ord('a')) * lv + (ord(S[j]) - ord('a'))) % mod
                    i += 1
                    if rh in hs:
                        return i, j

            return -1, -1

        # to find longest, using binary search and find a number between 1 and len(S) - 1
        l, r = 1, len(S) - 1
        i = j = -1
        res = None
        while l <= r:
            m = l + ((r - l) >> 1)
            i, j = search(m)
            if i != -1:
                res = (i, j)
                l = m + 1
            else:
                r = m - 1
        return S[res[0]: res[1] + 1] if res else ''

    def checkInclusion567(self, s1: str, s2: str) -> bool:
        # like check anagram. keep a fixed length of window = len of s1
        if s1 == '':
            return s2 == ''
        elif s2 == '':
            return False
        c1 = collections.Counter(s1)
        c2 = collections.Counter()
        start = 0
        for i in range(len(s2)):
            c2[s2[i]] += 1
            if c1 == c2: # because we also remove from counter, there will be old -> 0 mappings. so must delete first
                return True
            if i - start + 1 == len(s1):
                c2[s2[start]] -= 1
                if c2[s2[start]] == 0:
                    del c2[s2[start]]
                start += 1
        return False

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

    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        if not s:
            return 0
        c = collections.Counter()
        left = maxv = 0

        for i in range(len(s)):
            c[s[i]] += 1
            while len(c) > 2:
                c[s[left]] -= 1
                if c[s[left]] == 0:
                    del c[s[left]]
                left += 1  # this needs to happen after the deletion
            maxv = max(maxv, i - left + 1)
        return maxv

    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        if not s or k < 1:
            return 0
        c = collections.Counter()
        left = maxv = 0
        for i in range(len(s)):
            c[s[i]] += 1
            while len(c) > k:
                c[s[left]] -= 1
                if c[s[left]] == 0:
                    del c[s[left]]
                left += 1
            maxv = max(maxv, i - left + 1)
        return maxv

















