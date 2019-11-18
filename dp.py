from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = self.right = None

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

    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        # subproblem, use current + dp[i-2]; or do not use current, use dp[i-1]
        # and because only depend on last two. so use a dp array size 2
        dp = [nums[0], max(nums[0], nums[1])]
        for i in range(2, len(nums)):
            dp[0], dp[1] = dp[1], max(dp[1], dp[0] + nums[i])
        return dp[-1]

    def minCost(self, costs: List[List[int]]) -> int:
        if not costs or not costs[0]:
            return 0
        for i in range(1, len(costs)):
            for j in range(len(costs[0])):
                costs[i][j] += min(costs[i-1][(j+1) % 3], costs[i-1][(j+2) % 3])
        return min(costs[-1][0], costs[-1][1], costs[-1][2])

    def rob2(self, nums: List[int]) -> int:
        def rob(i, j):
            if i == j:
                return nums[i]
            dp = [nums[i], max(nums[i], nums[i+1])]
            for k in range(i + 2, j + 1):
                dp[0], dp[1] = dp[1], max(dp[1], dp[0] + nums[k])
            return dp[-1]

        # because first and last cannot be robbed together, calculate twice cutting first and then cutting last
        if not nums:
            return 0
        if len(nums) == 1:
            return nums[0]
        return max(rob(0, len(nums) - 2), rob(1, len(nums) - 1))

    def rob3(self, root: TreeNode) -> int:
        # dp using hashmap. subproblem: use current root + left's left + left' right + right' left + right' right;
        # or use left + right
        def dfs(root):
            if not root:
                return 0
            if root in hm:
                return hm[root]
            x = root.val
            if root.left:
                x += dfs(root.left.left) + dfs(root.left.right)
            if root.right:
                x += dfs(root.right.left) + dfs(root.right.right)
            return max(dfs(root.left) + dfs(root.right), x)

        hm = {} # based on node
        return dfs(root)

    def wordBreak139(self, s: str, wordDict: List[str]) -> bool:
        if not s or not wordDict:
            return False
        dp = [False] * (len(s) + 1)
        dp[0] = True
        hs = set(wordDict)
        for i in range(len(s)):
            for j in range(i - 1, -2, -1): # j need to reach -1
                if dp[j + 1] and s[j + 1: i + 1] in hs:
                    dp[i + 1] = True
        return dp[-1]

    def wordBreak140(self, s: str, wordDict: List[str]) -> List[str]:
        # dfs while memo from bottom up, so
        #  b..x...catsdogs
        #  a....y.catsdogs
        # so when the first dfs has record catsdogs can be broken down to a list, the second dfs can immediately reuse
        # bottom up dfs reuse results of bottom solution, stored in a hm
        def dfs(s) -> List[str]:
            if not s:
                return ['']
            if s in hm:
                return hm[s]
            res = []
            for i in range(1, len(s) + 1):
                x = s[:i]
                if x in hs:
                    rem = dfs(s[i:])
                    for r in rem:
                        res.append(x + ' ' + r if r else x)
            hm[s] = res
            return res

        if not s or not wordDict:
            return []
        hs = set(wordDict)
        hm = {}
        return dfs(s)




























































