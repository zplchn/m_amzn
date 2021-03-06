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

    def updateMatrix542(self, matrix: List[List[int]]) -> List[List[int]]:
        if not matrix or not matrix[0]:
            return []
        q = collections.deque()
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    q.append((i, j))
                else:
                    matrix[i][j] = float('inf')
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        while q:
            i, j = q.popleft()
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] > matrix[i][j] + 1:
                    matrix[x][y] = matrix[i][j] + 1
                    q.append((x, y))
        return matrix

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        # if we have a -> b, and b -> c then a/c == a -> b -> c. so this problem becomes find shortest path in graph
        def bfs(s: str, e: str) -> None:
            visited = {s}
            q = collections.deque([(s, 1)])  # node, product
            while q:
                node, product = q.popleft()
                if node == e:
                    res.append(product)
                    return
                for c, val in graph[node].items():
                    if c not in visited:
                        visited.add(c)
                        q.append((c, product * val))
            res.append(-1)

        if not equations or not values or not queries:
            return []
        graph = collections.defaultdict(dict)
        for (s, e), v in zip(equations, values):
            graph[s][e] = v
            graph[e][s] = 1 / v
        res = []
        for qs, qe in queries:
            if qs not in graph or qe not in graph:
                res.append(-1.0)
                continue
            bfs(qs, qe)
        return res

    def cutOffTree675(self, forest: List[List[int]]) -> int:
        # this bascially ask the sum of shortest path between every two nodes, in asending order. bfs to calculate
        # each pair and return the sum
        def bfs(sx, sy, ex, ey) -> int:
            if sx == ex and sy == ey:
                return 0
            q = collections.deque([(sx, sy, 0)])
            visited = {(sx, sy)}
            offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

            while q:
                x, y, dist = q.popleft()
                if x == ex and y == ey:
                    return dist
                for o in offsets:
                    i, j = x + o[0], y + o[1]
                    if 0 <= i < len(forest) and 0 <= j < len(forest[0]) and forest[i][j] != 0 and (i, j) not in visited:
                        visited.add((i, j))
                        q.append((i, j, dist + 1))
            return -1

        if not forest:
            return -1
        trees = []
        for i in range(len(forest)):
            for j in range(len(forest[0])):
                if forest[i][j] > 1: # dont forget to check only trees
                    trees.append((forest[i][j], i, j))
        trees.sort()
        res = 0
        r, c = 0, 0
        for h, i, j in trees:
            dist = bfs(r, c, i, j)
            if dist == -1:
                return -1
            res += dist
            r,c = i, j
        return res

    def canMeasureWater365(self, x: int, y: int, z: int) -> bool:
        if z > x + y:
            return False
        q = collections.deque([(0, 0)])
        visited = {(0, 0)}
        while q:
            a, b = q.popleft()
            if a + b == z:
                return True
            next_states = {(x, b), (a, y), (0, b), (a, 0), (min(x, a + b),
                            a + b - min(x, a + b)), (a + b - min(y, a + b), min(y, a + b))} - visited
            visited.update(next_states)
            q.extend(next_states)
        return False

    def numBusesToDestination815(self, routes: List[List[int]], S: int, T: int) -> int:
        # Create a graph stop -> [routes]. So problem become BFS when search fewest buses to take, each level is all
        # stops that can be reached by taking any bus from this stop. and then the next level is transfer at any stop to
        # another bus. And BFS shortest path will reach first to destination stop with fewest levels(each level is
        # all stops can be reached)
        if not routes or not routes[0] or S == T:
            return 0
        graph = collections.defaultdict(list)
        for r in range(len(routes)):
            for s in range(len(routes[r])): # here cannot use len(routes[0]) as each row length may be different
                graph[routes[r][s]].append(r)
        level = 0
        q = collections.deque([S])
        visited = set()
        while q:
            for i in range(len(q)):
                s = q.popleft()
                for r in graph[s]:
                    if r not in visited:
                        visited.add(r)
                        for stop in routes[r]:
                            if stop == T:
                                return level + 1
                            q.append(stop)
            level += 1
        return -1

    def openLock752(self, deadends: List[str], target: str) -> int:
        dead = set(deadends)
        level = 0
        if '0000' in dead:
            return -1
        q = collections.deque(['0000'])
        visited = {'0000'}

        while q:
            level += 1
            for i in range(len(q)):
                x = q.popleft()
                for i in range(4):
                    for d in [-1, 1]:
                        nx = x[:i] + str((int(x[i]) + d) % 10) + x[i + 1:] # str slice is better than convert to list
                        if nx == target:
                            return level
                        if nx not in dead and nx not in visited:
                            visited.add(nx)
                            q.append(nx)
        return -1

    def slidingPuzzle773(self, board: List[List[int]]) -> int:
        # Think of each state as a node, and find the shortest path for inital node to traverse to goal node
        combi = ''.join(str(board[i][j]) for i in range(len(board)) for j in range(len(board[0])))
        idx = combi.index('0')
        q = collections.deque([(combi, (idx // len(board[0]), idx % len(board[0])))]) # use len(board[0]) in both / %
        visited = {combi}
        level = 0
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        while q:
            for i in range(len(q)):
                combi, (i, j) = q.popleft()
                if combi == '123450':
                    return level
                id0 = i * len(board[0]) + j
                # l = list(combi)
                for o in offsets:
                    l = list(combi) # for string memoing, must every time convert to list in each chagne
                    x, y = i + o[0], j + o[1]
                    if 0 <= x < len(board) and 0 <= y < len(board[0]):
                        nid0 = x * len(board[0]) + y
                        l[nid0], l[id0] = l[id0], l[nid0]
                        t = ''.join(l)
                        if t not in visited:
                            visited.add(t)
                            q.append((t, (x, y)))
            level += 1
        return -1

    def racecar818(self, target: int) -> int:
        # because every time there are two choices, go A or go R. And if we just go A if less than target,
        # and R > target, it is like greedy, which does not guarantee shortest path. We need to enumerate all states,
        # which constructs a graph, then BFS can guarantee shortest path.
        # if AAAAA, it's 0 -> 1 -> 3 -> 7 -> 15 -> 31. (2**n - 1), we want to limit the children range, cannot < 0 or
        # > target * 2. why * 2. say target is 8 - 14, we are at 7, so we can go past 15 but no need to 31.
        q = collections.deque([(0, 1)]) # (position, speed)
        visited = {(0, 1)}
        level = 0
        while q:
            level += 1
            for i in range(len(q)):
                p, s = q.popleft()
                if p == target:
                    return level - 1 # For # of nodes, returnn level, for # of transitions/steps, return level - 1

                next_states = [(p + s, s * 2), (p, -1 if s > 0 else 1)]
                for ns in next_states:
                    if ns not in visited and 0 < ns[0] < target * 2:
                        visited.add(ns)
                        q.append(ns)
        return -1



































s = Solution()
x = s.shortestDistance317([[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]])
# print(x)

word_list = ['abc', 'def']
d = {}

for word in word_list:
    for i in range(len(word)):
        s = word[:i] + "_" + word[i+1:]
        d[s] = d.get(s, []) + [word]
# print(d)




