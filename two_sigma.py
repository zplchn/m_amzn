from typing import List, Tuple


class Solution:

    def findCircleNum(self, M: List[List[int]]) -> int:
        def dfs(i: int) -> None:
            visited[i] = True
            for j, y in enumerate(M[i]):
                if y and not visited[j]:
                    dfs(j)

        if not M or not M[0]:
            return 0
        res = 0
        visited = [False for _ in range(len(M))]

        for i, x in enumerate(visited):
            if not x:
                res += 1
                dfs(i)
        return res

    def longestStrChain(self, words: List[str]) -> int:
        # dp subproblem : longest chain by end of the current word
        if not words:
            return 0
        # sort by length
        words.sort(key=len)
        dp = {}  # use dict instead of array to prevent o(n2) loop
        res = 1
        for w in words:
            dp[w] = max(dp.get(w[:j] + w[j + 1:], 0) + 1 for j in range(len(w)))
            res = max(res, dp[w])
        return res

    def substrings(self, s: str) -> Tuple[str, str]:
        vowel = set('aeiou')
        low, high = '~', ''
        last_con, first_con = None, None
        for i in range(len(s) - 1, -1, -1):
            if s[i] not in vowel:
                if not last_con:
                    last_con = i
                first_con = i
            elif first_con:
                low = min(low, s[i:first_con + 1])
                high = max(high, s[i:last_con + 1])
        if low == '~':  # no vowel or no consonants
            return '', ''
        return low, high

    def missing_words(self, s: str, t: str) -> str:
        sa, ta = s.split(), t.split()
        res = []
        i = j = 0
        while i < len(sa):
            if j == len(ta) or sa[i] != ta[j]:
                res.append(sa[i])
                i += 1
            else:
                i, j = i + 1, j + 1
        return ' '.join(res)

s = Solution()

x = 'i like soft cheese and hard cheese yum'
y = 'like cheese yum'

print(s.missing_words(x, y))




