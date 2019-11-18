import collections
from typing import List

class Solution:
    def longestPalindrome5(self, s: str) -> str:
        if not s:
            return s
        maxv = 0
        res = ''
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        for i in range(len(s) - 1, -1, -1):
            for j in range(i, len(s)):
                if s[i] == s[j] and (j - i <= 2 or dp[i+1][j-1]):
                    dp[i][j] = True
                    x = j - i + 1
                    if x > maxv:
                        maxv = x
                        res = s[i: j + 1]
        return res

    def countSubstrings647(self, s: str) -> int:
        if not s:
            return 0
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        res = 0
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    res += 1
        return res

    def canPermutePalindrome266(self, s: str) -> bool:
        # only up to one odd char is allowed
        cnt_odd = sum(v % 2 for v in collections.Counter(s).values())
        return cnt_odd < 2

    def longestPalindrome409(self, s: str) -> int:
        c = collections.Counter(s)
        res, maxo = 0, 0
        for v in c.values():
            if v % 2 == 1:
                maxo = 1
            res += v // 2 * 2 # all odd will be used with x // 2 * 2 chars except add 1 at last
        return res + maxo

    def longestPalindromeSubseq516(self, s: str) -> int:
        # subproblem: if s[i] == s[j], then it's dp[i+1][j-1] + 2
        # if not, then it's either take off left or right edge. dp[i+1][j] or dp[i][j-1]
        if not s:
            return 0
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j]:
                    dp[i][j] = j - i + 1 if j - i <= 2 else dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[0][-1] # no need to compare max every time, dp[0][n - 1] already is the max length

    def partition131(self, s: str) -> List[List[str]]:
        # first set up the dp array, then dfs and word break
        def dfs(i: int, combi: List[str]) -> None:
            if i == n:
                res.append(combi[:])
                return
            for j in range(i, n):
                if dp[i][j]:
                    combi.append(s[i:j + 1])
                    dfs(j + 1, combi)
                    combi.pop()

        res = []
        if not s:
            return res
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True

        dfs(0, [])
        return res




