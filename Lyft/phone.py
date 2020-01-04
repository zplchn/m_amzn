import collections
import heapq
import bisect
from typing import List


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None


class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ZigzagIterator(object):

    def __init__(self, v1, v2):
        """
        Initialize your data structure here.
        :type v1: List[int]
        :type v2: List[int]
        """
        self.v = [v1, v2]
        self.q = collections.deque()
        if v1:
            self.q.append((0, 0))
        if v2:
            self.q.append((1, 0)) # (v, idx)

    def next(self):
        """
        :rtype: int
        """
        vid, idx = self.q.popleft()
        if idx != len(self.v[vid]) - 1:
            self.q.append((vid, idx + 1))
        return self.v[vid][idx]

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.q) > 0


class NestedIterator341(object):

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
        return self.st.pop().getInteger()

    def hasNext(self):
        """
        :rtype: bool
        """
        # It is possible it's just nested empty stack. so need to handle from hasNext not next till find a int
        # return len(self.st) > 0
        while self.st:
            if self.st[-1].isInteger():
                return True
            self.st.extend(reversed(self.st.pop().getList()))
        return False


class Vector2D:
    #251
    def __init__(self, v: List[List[int]]):
        self.v = v
        self.r = self.c = 0

    def next(self) -> int:
        self.hasNext()  # the next() can be called without hasNext() so need to move the iter
        x = self.v[self.r][self.c]
        self.c += 1
        return x

    def hasNext(self) -> bool:
        while self.r < len(self.v):
            if self.c < len(self.v[self.r]):
                return True
            else:
                self.r, self.c = self.r + 1, 0
        return False


class MaxStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.st = []
        self.maxst = []

    def push(self, x: int) -> None:
        self.st.append(x)
        if not self.maxst or self.maxst[-1] <= x:
            self.maxst.append(x)

    def pop(self) -> int:
        x = self.st.pop()
        if x == self.maxst[-1]:
            self.maxst.pop()
        return x

    def top(self) -> int:
        return self.st[-1]

    def peekMax(self) -> int:
        return self.maxst[-1]

    def popMax(self) -> int:
        t = []
        x = self.maxst.pop()
        while self.st[-1] != x:
            t.append(self.st.pop())
        self.st.pop()
        while t:
            self.push(t.pop())
        return x


class LRUCache:
    class ListNode:
        def __init__(self, key=None, val=None):
            # python func does not support overloading! every same name func
            # replace previous one. Use default value and factory methord to construct condionally

            self.key = key  # used when try to delete node
            self.val = val
            self.pre = self.next = None

    def __init__(self, capacity: int):
        self.cap = capacity
        self.head, self.tail = self.ListNode(), self.ListNode()
        self.size = 0
        self.hm = {}
        self.head.next = self.tail
        self.tail.pre = self.head

    def get(self, key: int) -> int:
        if key not in self.hm:
            return -1
        res = self.hm[key]
        self.move_to_head(res)
        return res.val

    def put(self, key: int, value: int) -> None:
        if key in self.hm:
            x = self.hm[key]
            x.val = value
            self.move_to_head(x)
        else:
            if self.size == self.cap:
                del self.hm[self.tail.pre.key]  # dont forget to remove from hm
                self.delete_node(self.tail.pre)
                self.size -= 1

            node = self.ListNode(key, value)
            self.insert_to_head(node)
            self.hm[key] = node
            self.size += 1

    def insert_to_head(self, node):
        node.next = self.head.next
        self.head.next.pre = node
        node.pre = self.head
        self.head.next = node

    def move_to_head(self, node):
        if self.head.next == node:
            return
        self.delete_node(node)
        self.insert_to_head(node)

    def delete_node(self, node):
        node.pre.next = node.next
        node.next.pre = node.pre


class KVStoreWithTime:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hm = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.hm[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        '''
        l = [1, 5, 5, 5, 8] bisect_left and bisect operates on return the INSERTION index and keep array still sorted.
        difference is when there are duplicate, bisect_left returns index of first occurance, bisect returns index
        AFTER the last occurance.
        so bisect_left(l, 5) -> 1 bisect(l, 5) -> 4. bisect(l, 1) = 1 so only when bisect value < first value,
        bisect will return 0
        '''
        if key not in self.hm:
            return ''
        l = self.hm[key]
        idx = bisect.bisect(l, (timestamp, chr(127)))
        return l[idx-1][1] if idx > 0 else ''


class AutocompleteSystem:
    '''
    642 Given a prefix, return all descendants. In order to return all words at once, instead of use is_word flag,
    which will need to construct path, we store the words directly on the nodes.
    Two cases:
    # -> record current input, return []
    any char -> dfs from node, find any children with word is not null. In order to rank, we need to sort by (-rank,
    word). So we also want to store a rank score in every node. At the end we return up to 3 results.

    Attention: tried to keep the intermediate node, but NEED to remenber once an input c is not in dict, need to set
    this intermediate node to None, otherwise the next input will based upon the last valid node which is wrong!
    Since trie only bound the length of word, it DOES NOT save time for doing this way. So just search the whole
    prefix every time!
    '''

    class TrieNode:
        def __init__(self):
            self.children = collections.defaultdict(AutocompleteSystem.TrieNode)
            self.word, self.rank = None, 0

    def __init__(self, sentences: List[str], times: List[int]):
        self.root = self.TrieNode()
        for s, t in zip(sentences, times):
            self.add_word(s, t)
        self.prefix = ''

    def add_word(self, s: str, t: int) -> None:
        node = self.root
        for c in s:
            node = node.children[c]
        node.word = s
        node.rank -= t

    def input(self, c: str) -> List[str]:
        if c == '#':
            self.add_word(self.prefix, 1)
            self.prefix = ''
            return []
        self.prefix += c
        res = self.search_word(self.prefix)
        return [w for _, w in sorted(res)[:3]]

    def search_word(self, s):
        def dfs(node):
            if node.word:
                res.append((node.rank,node.word))
            for c in node.children.values():
                dfs(c)

        node = self.root
        for c in s:
            if c not in node.children:
                return []
            node = node.children[c]
        res = []
        dfs(node)
        return res


class RangeSum2D:
    #304
    def __init__(self, matrix: List[List[int]]):
        if not matrix or not matrix[0]:
            self.dp = None
            return
        self.dp = [[0] * (len(matrix[0]) + 1) for _ in range(len(matrix) + 1)]

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                self.dp[i + 1][j + 1] = self.dp[i][j + 1] + self.dp[i + 1][j] - self.dp[i][j] + matrix[i][j]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        if not self.dp:
            return 0
        return self.dp[row2 + 1][col2 + 1] - self.dp[row1][col2 + 1] - self.dp[row2 + 1][col1] + self.dp[row1][col1]





class Solution:
    '''
    Asteroid collision

    '''
    def asteroidCollision735(self, asteroids: List[int]) -> List[int]:
        # cases: 1. No collision(cur is +, res empty or res[-1] is -)  2. collision, when res[-1] is + and cur is -.
        # If cur is smaller, nothing happens; if equal, pop and go; if cur is greater, pop and reset to cur position
        if not asteroids:
            return []
        res = []

        i = 0
        while i < len(asteroids):
            if asteroids[i] > 0 or not res or res[-1] < 0:
                res.append(asteroids[i])
            elif -asteroids[i] == res[-1]:
                res.pop()
            elif -asteroids[i] > res[-1]:
                res.pop()
                i -= 1
            i += 1
        return res

    '''
    不过和leetcode 735 不完全一样。 space station 在右边 How many asteroids will hit station。 同时要求O(1) space solution
    '''
    def asteroidHitSpaceStation(self, asteroids: List[int]) -> int:
        if not asteroids:
            return 0
        i = 1
        while i < len(asteroids):
            if i > 0 and asteroids[i] < 0 < asteroids[i - 1]:
                if -asteroids[i] == asteroids[i - 1]: #[1, 8, -8, 2]
                    asteroids.pop(i)
                    asteroids.pop(i - 1)
                    i = i - 1
                elif -asteroids[i] < asteroids[i - 1]: #[1, 8, -2]
                    asteroids.pop(i)
                else:
                    asteroids.pop(i - 1) #[1, 2, -8]
                    i = i - 1
            else:
                i += 1
        return sum(x > 0 for x in asteroids)

    '''
    先说了除没法handle有0的情况，然而面试官说可以先写除的，写完之后写了乘的，然后面试官说乘法cost也很大，要减少用乘的，最后优化成单独记录0的位置
    
    Division problem:
    1. overflow [2, 2 ** 31 - 1]
    2. cannot handle 0,  [0, 2]

    '''
    def productExceptSelf238(self, nums: List[int]) -> List[int]:
        if not nums:
            return nums
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            dp[i] = dp[i - 1] * nums[i - 1]
        numr = 1
        for i in reversed(range(len(nums) - 1)):
            numr *= nums[i + 1]
            dp[i] *= numr
        return dp

    def numDistinctIslands694(self, grid: List[List[int]]) -> int:
        # traverse 2d array. every shape will encounter the topleft point first which can be used as origin
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(oi, oj, i, j):
            grid[i][j] = 0
            combi.append((i - oi, j - oj))
            # dfs path will be same if two graphs are the same
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 1:
                    dfs(oi, oj, x, y)

        if not grid or not grid[0]:
            return 0
        hs = set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    combi = []
                    dfs(i, j, i, j)
                    hs.add(tuple(combi))
        return len(hs)

    def trapRain42(self, height: List[int]) -> int:
        if len(height) <= 2:
            return 0
        res = 0
        i, j = 0, len(height) - 1
        while i < j:
            minv = min(height[i], height[j])
            if minv == height[i]:
                while i < j and height[i] <= minv:
                    res += minv - height[i]
                    i += 1
            else:
                while i < j and height[j] <= minv:
                    res += minv - height[j]
                    j -= 1
        return res
    # 把0的地方是个洞, 雨水会全部漏掉
    def trapRainWithHole(self, height: List[int]) -> int:
        if len(height) < 3:
            return 0
        res = 0
        l, r = 0, len(height) - 1
        while l < r:
            t = 0
            minv = min(height[l], height[r])
            has_hole = False
            if minv == height[l]:
                while l < r and height[l] <= minv:
                    if height[l] == 0:
                        has_hole = True
                    elif not has_hole:
                        t += minv - height[l]
                    l += 1
            else:
                while l < r and height[r] <= minv:
                    if height[r] == 0:
                        has_hole = True
                    elif not has_hole:
                        t += minv - height[r]
                    r -= 1
            res += t if not has_hole else 0
        return res

    def largestBSTSubtree333(self, root: TreeNode) -> int:
        def dfs(root: TreeNode):
            nonlocal maxv
            if not root:
                return True, 0, float('inf'), float('-inf')  # is_bst, # of nodes, minv, maxv
            lb, ln, lmin, lmax = dfs(root.left)
            rb, rn, rmin, rmax = dfs(root.right)
            if lb and rb and lmax < root.val < rmin:  # note none root return -infinity as maxv and +inf as minv
                # print(root.val)
                size = ln + rn + 1
                maxv = max(maxv, size)
                return True, size, min(lmin, root.val), max(rmax, root.val)
            return False, 0, float('inf'), float('-inf')

        if not root:
            return 0
        maxv = 1
        dfs(root)
        return maxv

    def findDuplicateSubtrees652(self, root: TreeNode) -> List[TreeNode]:
        # we need a hm to store the value + structure, so need serialization. the serialzied string is preorder but,
        # we use post order to get subtrees first
        def dfs(root):
            if not root:
                return '#'
            ls, rs = dfs(root.left), dfs(root.right)
            key = ','.join([str(root.val), ls, rs])
            if c[key] == 1: # only output the first time
                res.append(root)
            c[key] += 1
            return key

        if not root:
            return []
        res = []
        c = collections.Counter()
        dfs(root)
        return res

    def treeToDoublyList426(self, root: 'Node') -> 'Node':
        # we use inorder and at cur root, modify pre's right(already processed before go to next) and modify cur's left(already processed)
        def dfs(root):
            nonlocal head, pre
            if not root:
                return
            dfs(root.left)
            if not head:
                head = pre = root
            else:
                pre.right = root
                root.left = pre
                pre = root
            dfs(root.right)

        if not root:
            return root
        pre = head = None
        dfs(root)
        pre.right = head
        head.left = pre
        return head

    def letterCombinations17(self, digits: str) -> List[str]:
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

    def reverseNum7(self, x: int) -> int:
        is_neg = x < 0
        x = -x if is_neg else x
        res = 0
        while x > 0:
            res = res * 10 + x % 10
            x //= 10
        res = -res if is_neg else res
        if res > 2 ** 31 - 1 or res < -2 ** 31:
            return 0
        return res

    def intersection349(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Or return set(nums1) & set(nums2) just works
        # Or
        # hs = set(nums1)
        # res = set()
        # for x in nums2:
        #     if x in hs:
        #         res.add(x)
        # return list(res)

        # Or: Already sorted or no extra space, use two pointers
        if not nums1 or not nums2:
            return []
        s = set()
        nums1.sort()
        nums2.sort()
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                s.add(nums1[i])
                i, j = i + 1, j + 1
        return list(s)

    def regex10(self, s: str, p: str) -> bool:
        '''
        Subproblems:
        1. p[j] is *

        a
        a b *    * means 0 preceeding dp[i][j] = dp[i][j - 2]

        a b
        a b *    * means 1+ preceeding(absorb) dp[i][j] = s[i] == p[j - 1] or p[j-1] == '.' and dp[i-1][j]

        2. p[j] is not *

        a
        a

        a
        .     dp[i][j] = s[i] == p[j] or p[j] == '.' and dp[i-1][j-1]

        Base case:
        when s is empty
        ''
        'a*b*.*' dp[i][j] = p[j] == '*' and dp[0][j - 2]

        Recursive:
        def isMatch(self, text, pattern):
            if not pattern:
                return not text

            first_match = bool(text) and pattern[0] in {text[0], '.'}

            if len(pattern) >= 2 and pattern[1] == '*':
                return (self.isMatch(text, pattern[2:]) or
                        first_match and self.isMatch(text[1:], pattern))
            else:
                return first_match and self.isMatch(text[1:], pattern[1:])

        '''
        if not p:
            return not s
        dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
        dp[0][0] = True
        for j in range(1, len(p)):
            dp[0][j + 1] = p[j] == '*' and dp[0][j - 1]
        for i in range(len(s)):
            for j in range(len(p)):
                if p[j] == '*':
                    dp[i + 1][j + 1] = dp[i + 1][j - 1] or (s[i] == p[j - 1] or p[j - 1] == '.') and dp[i][j + 1]
                else:
                    dp[i + 1][j + 1] = (s[i] == p[j] or p[j] == '.') and dp[i][j]
        return dp[-1][-1]

    def numSquares279(self, n: int) -> int:
        '''
        subproblem: at current i, dp[i] = min(dp[i], dp[i - j * j] + 1) for j * j <= i
        '''
        if n < 1:
            return 0
        dp = [i for i in range(n + 1)] # cannot init to 0!
        for i in range(1, n + 1):
            j = 1
            while j * j <= i:
                dp[i] = min(dp[i], dp[i - j * j] + 1)
                j += 1
        return dp[-1]

    def wordSearch79(self, board: List[List[str]], word: str) -> bool:
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(start, i, j):
            # if start == len(word): dont push to len(word) as it cause compare next char fail or 0<x<m and 0<y<n part may not true for any direction
            if start == len(word) - 1:
                return True
            t = board[i][j]
            board[i][j] = '#'
            res = False
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < m and 0 <= y < n and board[x][y] == word[start + 1]:
                    res = dfs(start + 1, x, y)
                    if res:
                        break
            board[i][j] = t
            return res

        if not board or not board[0] or not word:
            return False
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0] and dfs(0, i, j):
                    return True
        return False

    def minWindowSubstring76(self, s: str, t: str) -> str:
        '''
        Steps:  'adabxc'
        1) Use a counter to count chars in t
        2) Right edge: We loop according to right edge. Each step, enter the right char.
        3) Adjust counter: If exist in map, decrease the counter by 1. Use a cnt, and if the value is >=0, bump the counter
        4) Shrink left: while left is either not in map or value < 0
        5) Check max/min value: as long as cnt == len(t), get a local min/max
        '''
        if not s or not t:
            return ''
        i = cnt = 0
        res = ''
        minv = len(s) + 1
        c = collections.Counter(t)
        for j in range(len(s)):
            if s[j] not in c:
                continue
            c[s[j]] -= 1
            if c[s[j]] >= 0:
                cnt += 1
            while i < j and cnt == len(t) and s[i] not in c or c[s[i]] < 0:
                if s[i] in c:
                    c[s[i]] += 1
                i += 1
            if cnt == len(t) and j - i + 1 < minv:
                minv = j - i + 1
                res = s[i:j+1]
        return res

    def minWindowSubsequence727(self, S: str, T: str) -> str:
        '''
        Because we need to find subsequence, that is the substring need to contain T and keep the relative order in
        it, so we cannot use solution like hm and cnt. Instead, we want to compare a substring to T(using find) and
        enumerate all letters in T. if we find all letters in T, we find an end position and word in between is a
        possible solution.

        The optimization here is for S = bbbbde and T = bde. once we found the first bbbbde, we do not want to just
        move the start point by 1 and find another bbbde, and again bbde and bde. Rather, once we find the first
        candidate, we immediately do a reverse find/right find, and get the bde. So next round we move the start
        pointer to after the last b.

        '''
        minv = len(S) + 1
        res = ''
        i = -1
        while i < len(S):
            for c in T:
                i = S.find(c, i + 1) # to find substring right edge, use i = -1, i = s.find(char, i + 1). Must update
                # the i in the middle of find aba to find ba, we need to make sure find the second a not first a
                if i == -1:
                    return res
            r = i
            l = i + 1
            for c in reversed(T):
                l = S.rfind(c, 0, l) # rfind(char, start, end) is to find the largest index of char in [start,
                # end) because l is not reachable in next round, it' s okay. start can set = 0 because we already
                # know we have a solution

            if r - l + 1 < minv:
                minv = r - l + 1
                res = S[l:r + 1]
            i = l + 1
        return res

    def lowestCommonAncestor236(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root
        l = self.lowestCommonAncestor236(root.left, p, q)
        r = self.lowestCommonAncestor236(root.right, p, q)
        return root if l and r else l or r

    def spiralOrder54(self, matrix: List[List[int]]) -> List[int]:
        # use 4 sentinel varible for 4 four edges index, each time handle one circle and then adjust these 4 varibles
        if not matrix or not matrix[0]:
            return []
        m, n = len(matrix), len(matrix[0])
        l, r, u, d = 0, n - 1, 0, m - 1
        res = []
        while l <= r and u <= d:
            for j in range(l, r + 1):
                res.append(matrix[u][j])
            for i in range(u + 1, d):
                res.append(matrix[i][r])
            for j in range(r, l - 1, -1):
                res.append(matrix[d][j])
            for i in range(d - 1, u, -1):
                res.append(matrix[i][l])
            l, r, u, d = l + 1, r - 1, u + 1, d - 1
        return res[:m * n]

    def spiralGenerateMatrix59(self, n: int) -> List[List[int]]:
        res = [[0] * n for _ in range(n)]
        l, r, u, d = 0, n - 1, 0, n - 1
        x = 1
        while l <= r and u <= d:
            for j in range(l, r + 1):
                res[u][j] = x
                x += 1
            for i in range(u + 1, d):
                res[i][r] = x
                x += 1
            for j in range(r, l - 1, -1):
                res[d][j] = x
                x += 1
            for i in range(d - 1, u, -1):
                res[i][l] = x
                x += 1
            l, r, u, d = l + 1, r - 1, u + 1, d - 1
        if n % 2 == 1:
            res[n // 2][n // 2] -= 1
        return res

    def findDuplicates442(self, nums: List[int]) -> List[int]:
        # since numbers are between 1 - N(length of array), we use number -1 as index and use it as hm key,and set to negative to mean it has appeared before. So we just need to append to res if the index for current val is already negative
        if not nums:
            return []
        res = []
        for i in range(len(nums)):
            idx = abs(nums[i]) - 1
            if nums[idx] < 0:
                res.append(idx + 1)
            else:
                nums[idx] = -nums[idx]
        return res

    def addStrings415(self, num1: str, num2: str) -> str:
        if not num1 or not num2:
            return num1 or num2
        i, j, carry = len(num1) - 1, len(num2) - 1, 0
        res = []
        while i >= 0 or j >= 0 or carry:
            s = carry
            if i >= 0:
                s += int(num1[i])
                i -= 1
            if j >= 0:
                s += int(num2[j])
                j -= 1
            res.append(str(s % 10))
            carry = s // 10

        return ''.join(reversed(res))

    def mergeKLists23(self, lists: List[ListNode]) -> ListNode:
        ListNode.__eq__ = lambda x, y: x.val == y.val
        ListNode.__lt__ = lambda x, y: x.val < y.val

        if not lists:
            return None
        dummy = cur = ListNode(0)
        heap = [l for l in lists if l]
        heapq.heapify(heap)
        while heap:
            x = heapq.heappop(heap)
            cur.next = x
            cur = cur.next
            if x.next:
                heapq.heappush(heap, x.next)
        return dummy.next

    def sortedListToBST109(self, head: ListNode) -> TreeNode:
        def dfs(l: int, r: int) -> TreeNode:
            nonlocal head
            if l > r:
                return None
            m = l + ((r - l) >> 1)
            lc = dfs(l, m - 1)
            root = TreeNode(head.val)
            head = head.next
            root.left = lc
            root.right = dfs(m + 1, r)
            return root
        if not head:
            return None
        cnt = 0
        cur = head
        while cur:
            cnt += 1
            cur = cur.next
        return dfs(0, cnt - 1)

    def generateParenthesis22(self, n: int) -> List[str]:
        def dfs(l, r, combi):
            if len(combi) == 2 * n:
                res.append(''.join(combi))
                return
            if l < n:
                combi.append('(')
                dfs(l + 1, r, combi)
                combi.pop()
            if r < l:
                combi.append(')')
                dfs(l, r + 1, combi)
                combi.pop()
        if n <= 0:
            return []
        res = []
        dfs(0, 0, [])
        return res

    def maxProfit121(self, prices: List[int]) -> int:
        if not prices:
            return 0
        maxv, minv = 0, prices[0]
        for i in range(1, len(prices)):
            maxv = max(maxv, prices[i] - minv)
            minv = min(minv, prices[i])
        return maxv

    def longestPalindrome5(self, s: str) -> str:
        if not s:
            return s
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        maxv = 0
        res = ''
        for i in reversed(range(n)):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    if j - i + 1 > maxv:
                        maxv = j - i + 1
                        res = s[i: j + 1]
        return res

    def findPeakElement162(self, nums: List[int]) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l + ((r - l) >> 1)
            if (m == 0 or nums[m] > nums[m - 1]) and (m == len(nums) - 1 or nums[m] > nums[m + 1]):
                return m
            elif nums[m] < nums[m + 1]:
                l = m + 1
            else:
                r = m - 1
        return -1
    '''
    莉蔻162的变种，但是一共有三问。
    find local minimum
    （1）数有几个：
    1,2,3,4 -> 1 # 1
    5,2,5 3,4-> 1 # 2, 3
    5,4,4,5,3 -> 3 # 4, 4, 3
    （2）随便输出一个，楼主没动脑子直接把第一步的结果输出了
    （3）（2）能不能优化，提示了以后楼主说用二分法做
    '''
    def find_local_minimum(self, nums: List[int]):
        if not nums:
            return -1
        res = []
        for i, x in enumerate(nums):
            if (i == 0 or x <= nums[i - 1]) and (i == len(nums) - 1 or x <= nums[i + 1]):
                res.append(x)
        return res

    def wordLadder127(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        '''
        hot -> [hog, lot, hit...] at each letter, it can transform to some words, we need to use them in bfs every
        level. So we use _ot, h_t, ho_ as key to preprocess the input dict, at each deque, we replace each char with
        _ and fetch the dict, add all to the next level. BFS guaranteee the shortest path of levels.
        '''
        hm = collections.defaultdict(list)
        for w in wordList:
            for i in range(len(w)):
                t = w[:i] + '_' + w[i+1:]
                hm[t].append(w)
        q = collections.deque([beginWord])
        level = 0
        visited = {beginWord}
        while q:
            level += 1
            for _ in range(len(q)):
                x = q.popleft()
                if x == endWord:
                    return level
                for i in range(len(x)):
                    key = x[:i] + '_' + x[i + 1:]
                    for y in hm[key]:
                        if y not in visited:
                            visited.add(y)
                            q.append(y)
        return 0

    def TextJustification68(self, words: List[str], maxWidth: int) -> List[str]:
        '''
        Use a combi [] to accumulate word in a line, and calculate the total length. Keep adding until total length
        of word + total space + new word > maxL. So we know we cannot add, now round robin the space appending space to
        each word in combi. For last row, just space seperated and then left adjust
        '''
        res = []
        combi = []
        total_chars = 0
        for w in words:
            if total_chars + len(combi) + len(w) > maxWidth:
                # distribute space
                num_space = maxWidth - total_chars
                for i in range(num_space):
                    combi[i % (len(combi) - 1 or 1)] += ' '
                res.append(''.join(combi))
                # reset combi
                combi = []
                total_chars = 0
            combi.append(w)
            total_chars += len(w)
        res.append(' '.join(combi).ljust(maxWidth))
        return res

    def maxAreaOfIsland695(self, grid: List[List[int]]) -> int:
        '''
        Island problem must full mask island and NOT backtracking! We dont enter same island again and to calcuate area

        1 1
        1 1   if we backtracking, then first 1 use path from bottom 1 get a sum then add right 1 get another sum

        '''
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i: int, j: int) -> int:
            sumv = 1
            grid[i][j] = 0 # Dont forget dfs need to MASK before enter subproblems!
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 1:
                    sumv += dfs(x, y)
            # grid[i][j] = 1 Dont backtracking!
            return sumv

        if not grid or not grid[0]:
            return 0
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    res = max(res, dfs(i, j))
        return res














s = Solution()

# Asteroid
input = [[1], [1,2], [-1], [-1, -2], [-2, 1], [1, -2], [2, -1], [2, -2], [1, 2, -10], [10, 1, -5]]
for i in input:
    # res = s.asteroidCollision735(i)
    # print('before', i)
    # res = s.asteroidHitSpaceStation(i)
    # print('after', i, res)
    # print()
    pass

#Regular expression matching
r = [('', ''), ('', 'a'), ('a', ''), ('', '.*'),('', 'a*b*'), ('a', 'b'),('a', 'ab*'),('ab', 'ab*'),('abb', 'ab*'),
     ('ab', 'a.*')]
for x, y in r:
    # print(x, y, s.regex10(x, y))
    pass

# min window subsequence
# print(s.minWindowSubsequence727('abcdebdde', 'bde'))
# print(s.minWindowSubsequence727('cnhczmccqouqadqtmjjzl', 'mm'))

# find local minimum
m = [[1,2,3], [5,2,5,3,4],[5,4,4,5,3]]
for i in m:
    # print(s.find_local_minimum(i))
    pass

# Autocomplete
ss = ["i love you","island","iroman","i love leetcode"]
t = [5,3,2,2]
# [["i"],[" "],["a"],["#"]]
ac = AutocompleteSystem(ss, t)
for t in ["i"," ","a","#"]:
    # print(ac.input(t))
    pass

# Trap rain water

h = [0, 3, 1, 1,5]
# print(s.trapRainWithHole(h))

# max area
g = [[1, 1], [1, 1]]
print(s.maxAreaOfIsland695(g))




















































