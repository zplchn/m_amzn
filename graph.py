import collections
from typing import List

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
                if 0 <= i < len(rooms) and 0 <= j < len(rooms[0]) and rooms[i][j] > rooms[x][y]:
                    rooms[i][j] = rooms[x][y] + 1
                    q.append((i, j))

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

    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
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

    




















