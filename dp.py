from typing import List

class Solution:

    def wordBreak(self, s, dict):
        hs = set(dict)
        max_len = max((len(x) for x in hs), default=0) # if we need to make '' okay, max need a default value
        dp = [False] * (len(s) + 1)
        dp[0] = True # means before the current / first index it needs to be true. as a subproblem
        for i in range(len(s)): # use string to think dp loop
            if not dp[i]:
                continue
            for j in range(i + 1, len(s) + 1):
                if s[i: j] in hs:
                    dp[j] = True
                if j + 1 - i > max_len: # when the string is very long, no need to keep substring when len > max in dict
                    break
        return dp[-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        for i in range(1, len(grid)):
            grid[i][0] += grid[i - 1][0]
        for j in range(1, len(grid[0])):
            grid[0][j] += grid[0][j - 1]
        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        return grid[-1][-1]



















































