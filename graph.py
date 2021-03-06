import collections
from typing import List


class Node:
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbours = neighbors


class Solution:
    def canFinish(self, numCourses, prerequisites):
        if numCourses <= 1:
            return True
        indegrees = [0] * numCourses
        # graph = [[]] * numCourses
        # graph = [[]] * numCourses This is wrong, this create [l, l, l] while l all points to same list object! modify one will modify all others!
        graph = [[] for _ in range(numCourses)]
        for x, y in prerequisites:
            indegrees[x] += 1
            graph[y].append(x)
        q = collections.deque([i for i, v in enumerate(indegrees) if v == 0])
        cnt = 0
        while q:
            x = q.popleft()
            cnt += 1
            for c in graph[x]:
                indegrees[c] -= 1
                if indegrees[c] == 0:
                    q.append(c)
        return cnt == numCourses

    def shortestBridge(self, A: List[List[int]]) -> int:
        if not A or not A[0]:
            return 0

            # find one island and dfs, mark all 1 to 2, and enqueue for bfs
        q = collections.deque()
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i, j):
            A[i][j] = 2
            q.append((i, j))
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(A) and 0 <= y < len(A[0]) and A[x][y] == 1:
                    dfs(x, y)

        has_found = False
        for i in range(len(A)):
            if has_found:
                break  # the inner break is not enough. so need a flag to break outer loop
            for j in range(len(A[0])):
                if A[i][j] == 1:
                    dfs(i, j)
                    has_found = True
                    break
        res = 0
        while q:
            size = len(q)
            for _ in range(size):
                i, j = q.popleft()
                for o in offsets:
                    x, y = i + o[0], j + o[1]
                    if 0 <= x < len(A) and 0 <= y < len(A[0]):
                        if A[x][y] == 2:
                            continue
                        elif A[x][y] == 1:
                            return res
                        else:
                            A[x][y] = 2
                            q.append((x, y))
            res += 1
        return res

    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        # traverse 2d array. every shape will encounter the topleft point first which can be used as origin
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(oi, oj, i, j, combi):
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
                    dfs(i, j, i, j, combi)
                    hs.add(tuple(combi))
        return len(hs)

    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i, j):
            image[i][j] = newColor
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(image) and 0 <= y < len(image[0]) and image[x][y] == color:
                    dfs(x, y)

        if not image or not image[0] or image[sr][sc] == newColor:
            return image
        color = image[sr][sc]
        dfs(sr, sc)
        return image

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        res = []
        if numCourses < 1:
            return res
        indegrees = [0] * numCourses
        graph = [[] for _ in range(numCourses)]

        for x, y in prerequisites:
            indegrees[x] += 1
            graph[y].append(x)

        q = collections.deque([i for i, v in enumerate(indegrees) if v == 0])

        while q:
            x = q.popleft()
            res.append(x)
            for c in graph[x]:
                indegrees[c] -= 1
                if indegrees[c] == 0:
                    q.append(c)
        return res if len(res) == numCourses else []

    def findItinerary587(self, tickets: List[List[str]]) -> List[str]:
        # postorder dfs
        def dfs(s):
            while hm[s]:
                dfs(hm[s].pop())
            res.append(s)
        res = []
        hm = collections.defaultdict(list)
        #create graph using adjcency list

        for x, y in tickets:
            hm[x].append(y)
        # python does not have dict with key sorted, so need to sort ourselves
        for _, values in hm.items():
            values.sort(reverse=True) # smaller at the end. so when pop() out first

        dfs('JFK')
        return res[::-1]

    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        def dfs(i: int, t: List) -> None:
            visited[i] = True
            for j in range(1, len(accounts[i])):
                t.append(accounts[i][j])
                for k in hm[accounts[i][j]]:
                    if not visited[k]:
                        dfs(k, t)

        res = []
        if not accounts:
            return res
        visited = [False for _ in range(len(accounts))]
        # create graph based on emails . Want to combine based on what then have to create adjacency list based on it
        # and treat it as vertex.

        hm = collections.defaultdict(list)
        for i, l in enumerate(accounts):
            for j in range(1, len(l)):
                hm[l[j]].append(i)
        for i, x in enumerate(visited):
            if not x:
                t = []
                dfs(i, t)
                combi = [accounts[i][0]]
                combi.extend(sorted(set(t))) # the same email will be added wherever it appears in multiple list
                # print(t)
                res.append(combi)
        return res

    def areSentencesSimilarTwo(self, words1: List[str], words2: List[str], pairs: List[List[str]]) -> bool:
        def dfs(i: int, key: str) -> bool:
            visited.add(key)
            if words2[i] in hm[key]:
                return True
            for w in hm[key]:
                if w not in visited and dfs(i, w):
                    return True
            return False

        if len(words1) != len(words2):
            return False
        hm = collections.defaultdict(set)
        for x, y in pairs:
            hm[x].add(y)
            hm[y].add(x)
        for i in range(len(words1)):
            if words2[i] == words1[i]:
                continue
            visited = set()
            if not dfs(i, words1[i]):
                return False
        return True

    def findRedundantConnection684(self, edges: List[List[int]]) -> List[int]:
        def find(i):
            if roots[i] != i: # path compression at same time
                roots[i] = find(roots[i])
            return roots[i]

        roots = list(range(len(edges) + 1))
        for e in edges:
            ra, rb = map(find, e)
            if ra == rb:
                return e
            roots[ra] = rb
        return []

    def cloneGraph133(self, node: 'Node') -> 'Node':
        hm = {}
        hm[node] = Node(node.val, [])
        q = collections.deque([node])
        while q:
            x = q.popleft()
            for n in x.neighbors:
                if n not in hm:
                    hm[n] = Node(n.val, [])
                    q.append(n)
                hm[x].neighbors.append(hm[n])
        return hm[node]

    def findJudge997(self, N: int, trust: List[List[int]]) -> int:
        # convert problem into find a node in graph, indegree = N - 1 and adjacency list empty
        indegrees = [0] * (N + 1)
        outdegrees = [0] * (N + 1)
        for i, j in trust:
            indegrees[j] += 1
            outdegrees[i] = 1
        for i in range(1, len(indegrees)):
            if indegrees[i] == N - 1 and outdegrees[i] == 0:
                return i
        return -1








    




















