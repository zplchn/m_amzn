from typing import List, Tuple
import collections, heapq
from tree import TreeNode


class Codec271:
    # use len + # + str to encode

    def encode(self, strs):
        """Encodes a list of strings to a single string.

        :type strs: List[str]
        :rtype: str
        """
        return ''.join('%d#' % len(s) + s for s in strs)

    def decode(self, s):
        """Decodes a single string to a list of strings.

        :type s: str
        :rtype: List[str]
        """
        i = 0
        res = []
        while i < len(s):
            j = s.find('#', i)
            i = j + int(s[i:j]) + 1
            res.append(s[j + 1:i])
        return res


class TicTacToe:

    def __init__(self, n: int):
        """
        Initialize your data structure here.
        """
        self.row = [0 for _ in range(n)]
        self.col = [0 for _ in range(n)]
        self.diag = self.revdiag = 0
        self.n = n

    def move(self, row: int, col: int, player: int) -> int:
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        """
        x = 1 if player == 1 else -1
        self.row[row] += x
        self.col[col] += x
        self.diag += x if row == col else 0
        self.revdiag += x if row + col == self.n - 1 else 0
        return player if (
                    abs(self.row[row]) == self.n or abs(self.col[col]) == self.n or abs(self.diag) == self.n or abs(
                self.revdiag) == self.n) else 0

    def validTicTacToe794(self, board: List[str]) -> bool:
        # invalid cases: 1. count O != count X or count X - 1
        # 2. X win and count O != count X -1
        # 3. O win and count X != count X

        cnt_x = sum(row.count('X') for row in board)
        cnt_o = sum(row.count('O') for row in board)

        def win(player):
            for i in range(3):
                # all row or all column
                if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
                    return True
            return all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3))

        if cnt_o not in {cnt_x, cnt_x - 1}:
            return False
        if win('X') and cnt_o != cnt_x - 1:
            return False
        if win('O') and cnt_o != cnt_x:
            return False
        return True

class Solution:

    def numIslands(self, grid: List[List[str]]) -> int:
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i: int, j: int) -> None:
            grid[i][j] = '#'
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1':
                    dfs(x, y)

        res = 0
        if not grid or not grid[0]:
            return 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(i, j)
                    res += 1
        return res

    def titleToNumber171(self, s: str) -> int:
        if not s:
            return 0
        res = 0
        for c in s:
            res = res * 26 + ord(c) - ord('A') + 1
        return res

    def convertToTitle168(self, n: int) -> str:
        if n < 1:
            return ''
        res = []
        while n:
            n -= 1  # must minus 1 first otherwise 26 will be AZ not Z
            res.append(chr(n % 26 + ord('A')))
            n //= 26
        return ''.join(reversed(res))

    def isIsomorphic205(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        hm = {}
        for i, c in enumerate(s):
            if c in hm:
                if hm[c] != t[i]:
                    return False
            elif t[i] in hm.values():
                return False
            else:
                hm[c] = t[i]
        return True

    def reorganizeString767(self, S: str) -> str:
        # greedy. put in maxheap pairs of letter and count. every time pop two.
        c = collections.Counter(S)
        heap = [(-v, k) for k, v in c.items()]
        heapq.heapify(heap)
        res = []
        if -heap[0][0] > (len(S) + 1) // 2:
            return ''
        while len(heap) >= 2:
            c1, s1 = heapq.heappop(heap)
            c2, s2 = heapq.heappop(heap)
            res.append(s1)
            res.append(s2)
            c1, c2 = c1 + 1, c2 + 1
            if c1:
                heapq.heappush(heap, (c1, s1))
            if c2:
                heapq.heappush(heap, (c2, s2))
        if heap:
            res.append(heapq.heappop(heap)[1])

        return ''.join(res)

    def isCousins993(self, root: TreeNode, x: int, y: int) -> bool:
        def dfs(root, parent, depth):
            if not root:
                return
            hm[root.val] = (parent, depth)
            dfs(root.left, root, depth + 1)
            dfs(root.right, root, depth + 1)

        if not root:
            return False
        hm = {}
        dfs(root, None, 0)
        vx, vy = hm[x], hm[y]
        return vx[1] == vy[1] and vx[0] != vy[0]

    def exist79(self, board: List[List[str]], word: str) -> bool:
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i, j, n):
            if n == len(word) - 1:
                return True
            t = board[i][j]
            board[i][j] = '#'
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

    def findWords212(self, board: List[List[str]], words: List[str]) -> List[str]:

        class TrieNode:
            def __init__(self):
                self.children = collections.defaultdict(TrieNode)
                self.is_word = False

        def build_trie() -> TrieNode:
            root = TrieNode()
            for word in words:
                node = root
                for w in word:
                    node = node.children[w]
                node.is_word = True
            return root

        def starts_with(prefix: str) -> Tuple[bool, TrieNode]:
            node = root
            for p in prefix:
                if p not in node.children:
                    return False, node
                node = node.children[p]
            return True, node

        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i: int, j: int, s: str):
            is_starts_with, node = starts_with(s)
            if not is_starts_with:
                return
            if node.is_word:
                res.append(s)
            if not node.children:
                return
            t = board[i][j]
            board[i][j] = '#'
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] != '#' and  board[x][y] in \
                        node.children:
                    dfs(x, y, s + board[x][y])
            board[i][j] = t

        res = []
        if not board or not board[0] or not words:
            return res

        root = build_trie()

        for i in range(len(board)):
            for j in range(len(board[0])):
                if starts_with(board[i][j]):
                    dfs(i, j, board[i][j])
        return list(set(res)) # need to filter repeat words

    def merge56(self, intervals: List[List[int]]) -> List[List[int]]:
        res = []
        if not intervals:
            return res
        intervals.sort()
        res.append(intervals[0])
        for i in range(1, len(intervals)):
            if intervals[i][0] <= res[-1][1]:
                res[-1][1] = max(res[-1][1], intervals[i][1])
            else:
                res.append(intervals[i])
        return res

    def wallsAndGates286(self, rooms: List[List[int]]) -> None:
        if not rooms or not rooms[0]:
            return
        q = collections.deque()
        for i in range(len(rooms)):
            for j in range(len(rooms[0])):
                if rooms[i][j] == 0:
                    q.append((i, j))
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        while q:
            x, y = q.popleft()
            for o in offsets:
                i, j = x + o[0], y + o[1]
                if 0 <= i < len(rooms) and 0 <= j < len(rooms[0]) and rooms[i][j] > rooms[x][y]:
                    rooms[i][j] = rooms[x][y] + 1
                    q.append((i, j))

    def isValid20(self, s: str) -> bool:
        if not s:
            return True
        st = []
        for c in s:
            if c in '([{':
                st.append(c)
            else:
                if not st:
                    return False
                x = st.pop()
                if c == ')' and x != '(' or c == '}' and x != '{' or c == ']' and x != '[':
                    return False
        return len(st) == 0

    def checkValidString678(self, s: str) -> bool:
        if not s:
            return True
        # swipe twice, first time left to right, and all * count as (. use a cnt, ( + 1, ) -1. any time cnt < 0 means
        # ) is more so return false. after finish, means left (plus converted *) is at least more than ). Second
        # pass right to left, and treat all * as ). anytime cnt < 0 means left is more. *((). once done,
        # means right is at least more than left. So there must exist a solution.
        cnt = 0
        for i in range(len(s)):
            if s[i] == '(' or s[i] == '*':
                cnt += 1
            else:
                cnt -= 1
            if cnt < 0:
                return False
        if cnt == 0:
            return True
        cnt = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == ')' or s[i] == '*':
                cnt += 1
            else:
                cnt -= 1
            if cnt < 0:
                return False
        return True

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

