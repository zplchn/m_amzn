from typing import List


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


class Solution:
    def longestValidParentheses(self, s: str) -> int:
        # use a stack to maintain (. two key points: 1. start: the first valid ( 2. peek: the last valid (
        # when stack not empty, use peek / last ( ; when stack is empty, use first (, the start. when ）is more, +start
        if not s:
            return 0
        st = []
        start, maxv = 0, 0
        for i in range(len(s)):
            if s[i] == '(':
                st.append(i)
            else:
                if not st:
                    start = i + 1
                else:
                    st.pop()
                    maxv = max(maxv, i - st[-1]) if st else max(maxv, i - start + 1)
        return maxv

    def largestRectangleArea84(self, heights: List[int]) -> int:
        # use a st, and keep ascending histogram. then each one being the next one's left edge. so the problem become
        # finding the right edge, which is when current height is smaller than peek. then pop and calculate area
        if not heights:
            return 0
        heights.append(0) # this is the trick to force empty st after the loop is finished and items left in stack
        st = []
        maxv = 0
        for i in range(len(heights)):
            while st and heights[i] <= heights[st[-1]]:
                h = heights[st.pop()]
                maxv = max(maxv, ((i - st[-1] - 1) * h if st else h * i))
            st.append(i)
        return maxv

    def maximalRectangle85(self, matrix: List[List[str]]) -> int:
        # slice the matrix horizontally every row and keep the height in a array, then the problem convert to
        # calculate max area in histogram on every slice

        def max_area(heights: List[int]):
            nonlocal maxv
            st = []
            heights.append(0)
            for i in range(len(heights)):
                while st and heights[i] <= heights[st[-1]]:
                    h = heights[st.pop()]
                    maxv = max(maxv, ((i - st[-1] - 1) * h if st else i * h))
                st.append(i)

        if not matrix or not matrix[0]:
            return 0
        heights = [0] * len(matrix[0])
        maxv = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                heights[j] = 0 if matrix[i][j] == '0' else heights[j] + 1
            max_area(heights)
        return maxv

    def removeDuplicates(self, S: str) -> str:
        if not S:
            return S
        st = []
        for s in S:
            if not st or s != st[-1]:
                st.append(s)
            else:
                st.pop()
        return ''.join(st)

    def removeDuplicates1209(self, s: str, k: int) -> str:
        # use stack to store a pair of (s, cnt) and incre the cnt, remove when cnt == k
        if not s or k <= 0:
            return s
        st = []
        for c in s:
            if not st or c != st[-1][0]:
                st.append([c, 1])
            else:
                st[-1][1] += 1
                if st[-1][1] == k:
                    st.pop()
        return ''.join([k[0] * k[1] for k in st])

    def minRemoveToMakeValid1249(self, s: str) -> str:
        # two cases: 1. ) is more, simply has to remove it, no way around. 2. ( is more, remove unpaired, use a stack
        # the char will be just noise. (() and ()) are the cases. record the index removed in a set so to reconsstruct
        if not s:
            return s
        hs = set()
        st = []
        for i in range(len(s)):
            if s[i] not in '()':
                continue
            elif s[i] == '(':
                st.append(i)
            else:
                if not st:
                    hs.add(i)
                else:
                    st.pop()
        hs = hs.union(set(st))  # set union with another set. not modify itself
        return ''.join([s[i] for i in range(len(s)) if i not in hs])

    def removeKdigits402(self, num: str, k: int) -> str:
        # suppose k = 2: case 1: increasing: 1234. remove last 2 -> 12
        # case 2: 2314, first need to remove 3. then remove 2. so use stack and pop when peek >= cur while k > 0
        if not num or k <= 0:
            return num
        st = []
        for s in num:
            while k and st and s < st[-1]: # 112 do not include =
                st.pop()
                k -= 1
            st.append(s)
        st = st[:-k] if k else st # trim k from the back
        return ''.join(st).lstrip('0') or '0' # 2304 case

    def decodeString394(self, s: str) -> str:
        if not s:
            return s
        st = []
        num, combi = 0, ''
        for x in s:
            if x.isdigit():
                num = num * 10 + ord(x) - ord('0')
            elif x == '[':
                st.append(combi)
                st.append(num)
                num, combi = 0, ''
            elif x == ']':
                num = st.pop()
                combi = st.pop() + combi * num
                num = 0
            else:
                combi += x
        return combi




