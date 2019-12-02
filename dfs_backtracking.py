from typing import List
import collections

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

    def restoreIpAddresses(self, s: str) -> List[str]:
        def dfs(i, k, combi):
            if k == 3:
                ss = s[i:]
                if is_valid(ss):
                    t = combi[:]
                    t.append(ss)
                    res.append('.'.join(t))
                return
            for j in range(i + 1, i + 4):
                # when there are both range() and str slicing happens, pay attention to end condition.
                # i cannot be i + 4 and then slice cannot take i + 3
                ss = s[i:j]
                if is_valid(ss):
                    combi.append(ss)
                    dfs(j, k + 1, combi)
                    combi.pop()

        def is_valid(s) -> bool:
            return 0 < len(s) <= 3 and int(s) <= 255 and not (len(s) > 1 and s[0] == '0')

        res = []
        if not s:
            return res
        dfs(0, 0, [])
        return res

    def canPartitionKSubsets698(self, nums: List[int], k: int) -> bool:
        # recursion to find k some array with sum = sum_all / k
        def dfs(start: int, pre: int, k: int) -> bool:
            if k == 1:
                return True
            if pre == sumv:
                return dfs(0, 0, k - 1)
            res = False
            for i in range(start, len(nums)):
                if visited[i]:
                    continue
                if pre + nums[i] <= sumv:
                    visited[i] = True
                    res = dfs(i + 1, pre + nums[i], k)
                    visited[i] = False
                    if res:
                        break
                else:
                    break
            return res

        if not nums or k <= 0:
            return False
        sumv = sum(nums)
        if sumv % k != 0:
            return False
        sumv //= k
        visited = [False] * len(nums)
        nums.sort()
        return dfs(0, 0, k)

    def hasPath490(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i: int, j: int) -> bool:
            if i == destination[0] and j == destination[1]:
                return True
            maze[i][j] = -1

            for o in offsets:
                x, y = i, j
                while 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] != 1:
                    x, y = x + o[0], y + o[1]
                x, y = x - o[0], y - o[1]
                if maze[x][y] != -1:
                    if dfs(x, y):
                        return True
            return False

        if not maze or not maze[0] or not start or not destination:
            return False
        # return dfs(start[0], start[1]) use * to unzip list type param
        return dfs(*start)

    def solveNQueens51(self, n: int) -> List[List[str]]:
        def is_valid(cols: List[int], row: int) -> bool:
            for i in range(row):
                if cols[i] == cols[row] or abs(cols[i] - cols[row]) == row - i:
                    return False
            return True

        def dfs(row: int) -> None:
            if row == n:
                t = []
                for i in range(n):
                    combi = ['.'] * n
                    combi[cols[i]] = 'Q'
                    t.append(''.join(combi))
                res.append(t)
                return
            for i in range(n):
                cols[row] = i
                if is_valid(cols, row):
                    dfs(row + 1)
        if n < 1:
            return []
        cols = [0] * n
        res = []
        dfs(0)
        return res

    def totalNQueens52(self, n: int) -> int:
        def is_valid(cols: List[int], row: int) -> bool:
            for i in range(row):
                if cols[i] == cols[row] or abs(cols[i] - cols[row]) == row - i:
                    return False
            return True

        def dfs(row: int) -> None:
            nonlocal res
            if row == n:
                res += 1
                return
            for i in range(n):
                cols[row] = i
                if is_valid(cols, row):
                    dfs(row + 1)

        if n < 1:
            return 0
        res = 0
        cols = [0] * n
        dfs(0)
        return res

    def solveSudoku37(self, board: List[List[str]]) -> None:
        def is_valid(i, j) -> bool:
            for x in range(9):
                if (x != i and board[x][j] == board[i][j]) or (x != j and board[i][x] == board[i][j]):
                    return False
            for x in range(i // 3 * 3, i // 3 * 3 + 3):
                for y in range(j // 3 * 3, j // 3 * 3 + 3): # note this is // not %
                    if (not (x == i and y == j) and board[x][y] == board[i][j]):
                        return False
            return True
        def dfs(i: int, j: int) -> bool:
            if j == 9:
                return dfs(i + 1, 0)
            if i == 9:
                return True
            if board[i][j] != '.':
                return dfs(i, j + 1)
            for k in range(1, 10):
                board[i][j] = str(k)
                if is_valid(i, j):
                    if dfs(i, j + 1):
                        return True
                board[i][j] = '.'  # must backtrack because the first dfs will fill till last node and is_valid also
                # depends on the node after the current level
            return False
        if not board or not board[0]:
            return
        dfs(0, 0)

    def canPartition416(self, nums: List[int]) -> bool:
        # converts the problem of finding a combination of numbers in the list sum == sum_all // 2
        # the reason using a counter - say there are 100 of 1s, and another random number 99. without using counter,
        # then each dfs generate a tree, so there will be 100 trees. by using counter, only generate 2 tree,
        # one for 1 and one for 99.
        def dfs(x) -> bool:
            if x == 0:
                return True
            if x < 0:
                return False
            for k in c:
                if c[k] == 0:
                    continue
                c[k] -= 1
                if dfs(x - k):
                    return True
                c[k] += 1
            return False
        if not nums:
            return False
        sum_all = sum(x for x in nums)
        if sum_all % 2:
            return False
        c = collections.Counter(nums)
        return dfs(sum_all // 2)

    def canCross403(self, stones: List[int]) -> bool:
        def dfs(cur: int, speed: int) -> bool:
            if cur == stones[-1]:
                return True
            if (cur, speed) in memo:
                return False

            for s in [speed -1, speed, speed + 1]:
                if s < 1:
                    continue
                new_stone = cur + s
                if new_stone in hs:
                    if dfs(new_stone, s):
                        return True
            memo.add((cur, speed))
            return False

        if not stones:
            return False
        hs = set(stones)
        memo = set()
        return dfs(1, 1)

    def cleanRoom489(self, robot):
        # class Robot:
        #    def move(self):
        #        """
        #        Returns true if the cell in front is open and robot moves into the cell.
        #        Returns false if the cell in front is blocked and robot stays in the current cell.
        #        :rtype bool
        #        """
        #
        #    def turnLeft(self):
        #        """
        #        Robot will stay in the same cell after calling turnLeft/turnRight.
        #        Each turn will be 90 degrees.
        #        :rtype void
        #        """
        #
        #    def turnRight(self):
        #        """
        #        Robot will stay in the same cell after calling turnLeft/turnRight.
        #        Each turn will be 90 degrees.
        #        :rtype void
        #        """
        #
        #    def clean(self):
        #        """
        #        Clean the current cell.
        #        :rtype void
        #        """
        """
        Treat this problem as 4-branch tree DFS. at each level only turn right to go to next branch. To go back one
        level up, turn right -> right -> move one step -> right -> right. so keep the direction in order
        :type robot: Robot
        :rtype: None
        """
        directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]

        def dfs(i, j, curdir):
            robot.clean()
            visited.add((i, j))
            for k in range(4):
                x, y = i + directions[curdir][0], j + directions[curdir][1]
                if (x, y) not in visited and robot.move():  # keep current direction and move in if can
                    dfs(x, y, curdir)
                # turn right
                curdir = (curdir + 1) % 4
                robot.turnRight()
            # this finishes all 4 directions, need to go back one step / backtrack
            robot.turnRight()
            robot.turnRight()
            robot.move()
            robot.turnRight()
            robot.turnRight()

        visited = set()
        dfs(0, 0, 0)

    def removeInvalidParentheses301(self, s: str) -> List[str]:
        def is_valid(s: str) -> bool:
            cnt = 0
            for c in s:
                if c == '(':
                    cnt += 1
                elif c == ')':
                    cnt -= 1
                    if cnt < 0: # ())(
                        return False
            return cnt == 0

        def dfs(combi: str, start: int, c1: int, c2: int) -> None:
            if c1 == c2 == 0:
                if is_valid(combi):
                    res.append(combi)
                return
            for i in range(start, len(combi)):
                if i != start and combi[i] == combi[i - 1]:
                    continue # for (() or ())) the same paranth remvoe any one is the same
                if combi[i] == '(' and c1 > 0:
                    dfs(combi[:i] + combi[i + 1:], i, c1 - 1, c2) # start should be i since i is removed
                if combi[i] == ')' and c2 > 0:
                    dfs(combi[:i] + combi[i + 1:], i, c1, c2 - 1)

        # count the extra ( or ) using two counters c1, c2. when ( is more, c1 will be +. otherwise c2 +.
        # then using dfs and remove ( and ) when c1 or c2 is +. when all == 0. check is valid

        c1 = c2 = 0
        for c in s:
            if c == '(':
                c1 += 1
            elif c == ')':
                if c1 > 0:
                    c1 -= 1
                else:
                    c2 += 1
        # )( --> c1 = c2 = 1
        # till now we get the # of ( and ) need to be removed

        res = []
        dfs(s, 0, c1, c2)
        return res

















