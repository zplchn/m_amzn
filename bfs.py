from typing import List
import collections


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None


class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
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
                if 0 <= i < len(rooms) and 0 <= j < len(rooms[0]) and rooms[i][j] == 2 ** 31 - 1:
                    rooms[i][j] = rooms[x][y] + 1
                    q.append((i, j))

    def orangesRotting994(self, grid: List[List[int]]) -> int:
        def check_fresh():
            return any(g.count(1) for g in grid)

        if not grid or not grid[0]:
            return 0
        if not check_fresh():
            return 0
        q = collections.deque()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 2:
                    q.append((i, j))
                    grid[i][j] = 0

        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        res = 0
        while q:
            i, j = q.popleft()
            for o in offsets:
                x, y = i + o[0], j + o[1]
                # if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and (grid[x][y] == 1 or (grid[x][y] < 0 and grid[x][
                #     y] < grid[i][j] - 1)): # BFS already guarantee shortest pass. a node will be reached first by its
                #     closest source node. It's not possible for a farther node to set a value first and then a
                #     closer node reached and shorted the value. so BFS will only need to check NEW node
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 1:
                    grid[x][y] = grid[i][j] - 1
                    res = min(res, grid[x][y])
                    q.append((x, y))
        if check_fresh():
            return -1
        return abs(res)

    def shortestDistance317(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        m, n = len(grid), len(grid[0])
        total = [[0 for _ in range(n)] for _ in range(m)]

        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        minv = -1
        target = 0
        # because we need every possible 0 to be reached, we cannot modify the original grid, so we need to bfs with
        # value in the queue. Also, since we need a spot reachable from all buildings, to simplify, we mark the first
        # round all reachable spots to be -1, next round to be all -2 (and only care spots already -1)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    minv = float('inf')
                    q = collections.deque([(i, j, 0)])
                    while q:
                        a, b, v = q.popleft()
                        v += 1
                        for o in offsets:
                            x, y = a + o[0], b + o[1]
                            if 0 <= x < m and 0 <= y < n and grid[x][y] == target:
                                grid[x][y] -= 1 # bfs must mark before enqueue, not after deque. because it can be
                                # added before it's dequed
                                total[x][y] += v
                                minv = min(minv, total[x][y])
                                q.append((x, y, v))
                    target -= 1
        return -1 if minv == float('inf') else minv

    def ladderLength127(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # shortest distance use graph + bfs to model the q. use a queue, and change letter, check in dict, enqueue
        if not beginWord or not endWord or not wordList:
            return 0
        q = collections.deque([(beginWord, 0)])
        d = collections.defaultdict(list)

        for w in wordList:
            for i in range(len(w)):
                k = w[:i] + '_' + w[i + 1:]
                d[k].append(w) # in case abc and bbc, _bc will -> [abc, bbc]

        visited = {beginWord}
        while q:
            word, cnt = q.popleft()
            cnt += 1
            for i in range(len(beginWord)):
                k = word[:i] + '_' + word[i + 1:]
                neighbors = d[k]
                for n in neighbors:
                    if n not in visited:
                        if n == endWord:
                            return cnt + 1  # the endword must in dict itself. and return length = convertion + 1
                        # print('in',(s, cnt))
                        visited.add(n)
                        q.append((n, cnt))
        return 0

    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0] or grid[0][0] == 1: # initial if 1 return -1
            return -1
        q = collections.deque([(1, 0, 0)])
        grid[0][0] = 1
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
        while q:
            l, i, j = q.popleft()
            # BFS check stop conditino when deque, cornor case is only one node in the grid!
            if i == len(grid) - 1 and j == len(grid[0]) - 1:
                return l
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0:
                    grid[x][y] = 1
                    q.append((l + 1, x, y))
        return -1

    def distanceK863(self, root: TreeNode, target: TreeNode, K: int) -> List[int]:
        if not root or not target or K < 0: # if 0 output itself
            return []
            # find distance -> BFS. For a tree, we dont know parent, so we need to first convert it to a graph

        def convert(root, pre):
            if not root:
                return
            if pre:
                graph[pre].append(root)
                graph[root].append(pre)
            convert(root.left, root)
            convert(root.right, root)

        graph = collections.defaultdict(list)
        convert(root, None)
        q = collections.deque([target])
        visited = {target}
        while q and K > 0:
            cnt = len(q)
            for _ in range(cnt):
                x = q.popleft()
                for y in graph[x]:
                    if y not in visited:
                        visited.add(y)
                        q.append(y)
            K -= 1
        res = []
        while q:
            res.append(q.popleft().val)
        return res











s = Solution()
x = s.shortestDistance317([[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]])
# print(x)

word_list = ['abc', 'def']
d = {}

for word in word_list:
    for i in range(len(word)):
        s = word[:i] + "_" + word[i+1:]
        d[s] = d.get(s, []) + [word]
print(d)




