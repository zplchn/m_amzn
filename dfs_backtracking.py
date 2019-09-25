from typing import List

class Solution:
    def letterCombinations(self, digits):
        letter_dict = {'2': ['a', 'b', 'c'],
                       '3': ['d', 'e','f'],
                       '4': ['g', 'h', 'i'],
                       '5': ['j', 'k', 'l'],
                       '6': ['m', 'n', 'o'],
                       '7': ['p', 'q', 'r', 's'],
                       '8': ['t', 'u', 'v'],
                       '9': ['w', 'x', 'y', 'z']}

        def dfs(i, combi, res):
            if i == len(digits):
                res.append(''.join(combi)) # dont forget to join
                return
            for x in letter_dict[digits[i]]:
                combi.append(x)
                dfs(i + 1, combi, res)
                combi.pop()

        res = []
        if not digits:
            return res
        dfs(0, [], res)
        return res

    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(i, j):
            offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
            grid[i][j] = '0'
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1':
                    dfs(x, y)

        if not grid or not grid[0]:
            return 0
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    res += 1
                    dfs(i, j)
        return res

    def solve(self, board: List[List[str]]) -> None:
        def dfs(i, j):
            board[i][j] = '#'
            offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] == 'O':
                    dfs(x, y)

        if not board or not board[0]:
            return
        for j in range(len(board[0])):
            dfs(0, j) if board[0][j] == 'O' else None
            dfs(len(board) - 1, j) if board[len(board) - 1][j] == 'O' else None
        for i in range(len(board)):
            dfs(i, 0) if board[i][0] == 'O' else None
            dfs(i, len(board[0]) - 1) if board[i][len(board[0]) - 1] == 'O' else None
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == '#':
                    board[i][j] = 'O'

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i, j):
            grid[i][j] = 0
            sumv = 1
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 1:
                    sumv += dfs(x, y)
            return sumv

        if not grid or not grid[0]:
            return 0
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    res = max(res, dfs(i, j))
        return res

    def generateParenthesis(self, n: int) -> List[str]:
        def dfs(left, right, combi, res):
            if len(combi) == n * 2:
                res.append(''.join(combi))
                return
            if left < n:
                combi.append('(')
                dfs(left + 1, right, combi, res)
                combi.pop()
            if right < left:
                combi.append(')')
                dfs(left, right + 1, combi, res)
                combi.pop()

        res = []
        if n < 1:
            return res
        dfs(0, 0, [], res)
        return res

    def exist(self, board: List[List[str]], word: str) -> bool:
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i, j, n):
            if n == len(word) - 1:
                return True
            t = board[i][j]
            board[i][j] = '#' # dont forget to mark!
            res = False
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] == word[n + 1]:
                    res = dfs(x, y, n + 1)
                    if res:
                        break
            board[i][j] = t
            return res

        if not word:
            return not board
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0] and dfs(i, j, 0):
                    return True
        return False

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(start, sum, combi):
            if sum == target:
                res.append(combi[:])
                return
            for i in range(start, len(candidates)):
                if sum + candidates[i] > target:
                    break
                combi.append(candidates[i])
                dfs(i, sum + candidates[i], combi)
                combi.pop()

        res = []
        if not candidates:
            return res
        candidates.sort()
        dfs(0, 0, [])
        return res

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(start, sum, combi):
            if sum == target:
                res.append(combi[:])
                return
            for i in range(start, len(candidates)):
                if i > start and candidates[i] == candidates[i-1]:
                    continue
                if sum + candidates[i] > target:
                    break
                combi.append(candidates[i])
                dfs(i + 1, sum + candidates[i], combi)
                combi.pop()

        res = []
        if not candidates:
            return res
        candidates.sort()
        dfs(0, 0, [])
        return res

    def letterCombinations(self, digits: str) -> List[str]:
        letters = {
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }

        def dfs(i, combi):
            if i == len(digits):
                res.append(''.join(combi))
                return
            for x in letters[digits[i]]:
                combi.append(x)
                dfs(i + 1, combi)
                combi.pop()
        res = []
        if not digits:
            return res
        dfs(0, [])
        return res

    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(marker, combi):
            if len(combi) == len(nums):
                res.append(combi[:])
                return
            for i in range(len(nums)):
                if not marker[i]:
                    marker[i] = True
                    combi.append(nums[i])
                    dfs(marker, combi)
                    combi.pop()
                    marker[i] = False
        res = []
        if not nums:
            return res

        dfs([False] * len(nums), [])
        return res

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dfs(marker, combi):
            if len(combi) == len(nums):
                res.append(combi[:])
                return
            for i in range(len(nums)):
                if not marker[i] and not (i > 0 and nums[i] == nums[i - 1] and not marker[i - 1]):
                    marker[i] = True
                    combi.append(nums[i])
                    dfs(marker, combi)
                    marker[i] = False
                    combi.pop()
        res = []
        if not nums:
            return res
        nums.sort()
        dfs([False] * len(nums), [])
        return res

    def letterCasePermutation(self, S: str) -> List[str]:
        def dfs(i):
            if i == len(ca):
                res.append(''.join(ca))
                return
            dfs(i + 1)
            if ca[i].isalpha():
                ca[i] = ca[i].swapcase() # must reassign the value. str cannot mutate inplace
                dfs(i + 1)
                ca[i] = ca[i].swapcase()

        res = []
        if not S:
            return res
        ca = list(S)
        dfs(0)
        return res

    def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i: int, j: int, visited: List[List[bool]]):
            visited[i][j] = True
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and not visited[x][y] and matrix[x][y] >= matrix[i][j]:
                    # think reversely, the new node need to be larger because we start off th edge and go up hills.
                    dfs(x, y, visited)

        res = []
        if not matrix or not matrix[0]:
            return res
        pacific = [[False] * len(matrix[0]) for _ in range(len(matrix))]
        atlantic = [[False] * len(matrix[0]) for _ in range(len(matrix))]
        for j in range(len(matrix[0])):
            dfs(0, j, pacific)
            dfs(len(matrix) - 1, j, atlantic)

        for i in range(len(matrix)):
            dfs(i, 0, pacific)
            dfs(i, len(matrix[0]) - 1, atlantic)

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if pacific[i][j] and atlantic[i][j]:
                    res.append([i, j])
        return res












