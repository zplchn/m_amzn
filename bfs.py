from typing import List
import collections


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






s = Solution()
x = s.shortestDistance317([[1,0,2,0,1],[0,0,0,0,0],[0,0,1,0,0]])
print(x)





