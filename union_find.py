from typing import List


class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        def find(i):
            # print('find', i)
            if roots[i] != i:
                roots[i] = find(roots[i])
            return roots[i]

        if not graph:
            return False
        roots = [i for i in range(len(graph))]
        for i in range(len(graph)):
            if graph[i]:
                ri = find(i)
                rj = find(graph[i][0])
                if ri == rj:
                    return False
                for k in range(1, len(graph[i])):
                    rk = find(graph[i][k])
                    if ri == rk:
                        return False
                    roots[graph[i][k]] = rj
                print(roots)
        return True

    def minimumCost1135(self, N: int, connections: List[List[int]]) -> int:
        # greedy, sort by cost and try to union edge if not already connected
        def find(i: int) -> int:
            if roots[i] != -1:
                roots[i] = find(roots[i])
            return roots[i]

        if N < 2 or not connections:
            return 0
        connections.sort(key=lambda l: l[2])
        roots = [-1] * (N + 1)
        res = 0
        for x, y, v in connections:
            rx, ry = find(x), find(y)
            if rx != ry:
                res += v
                roots[rx] = ry # union is to union the roots join one to the other
        return res if len({find(i) for i in range(1, N + 1)}) == 1 else -1 # set comprehension find # of roots

    def validTree261(self, n: int, edges: List[List[int]]) -> bool:
        def find(i):
            if roots[i] != i:
                roots[i] = find(roots[i])
            return roots[i]

        # for an graph to be valid tree : 1. connected 2. no circle
        if len(edges) != n - 1:
            return False # maybe all connected. if no circle
        # union find check no circle. if 2 nodes of an edge are already connected
        roots = list(range(n))
        for e in edges:
            ra, rb = map(find, e)
            if ra == rb:
                return False
            roots[rb] = ra
        return True

    def countComponents323(self, n: int, edges: List[List[int]]) -> int:
        def find(i: int) -> int:
            if roots[i] != i:
                roots[i] = find(roots[i])
            return roots[i]

        if n < 1:
            return 0
        roots = list(range(n))
        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                roots[ra] = rb
        return len({find(i) for i in range(n)})

    def numIslands2_305(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        # union find if 2d array, one way is encode to 1d array roots, or use a hashmap and store tuple
        def find(t): # pass t as a whole, t is tuple
            if roots[t] != t:
                roots[t] = find(roots[t])
            return roots[t]

        roots = {}
        res = []
        cnt = 0
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for i, j in positions:
            if (i, j) in roots:
                res.append(cnt)
                continue
            cnt += 1
            roots[(i, j)] = i, j
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if (x, y) in roots:
                    rn, rc = find((x, y)), find((i, j))
                    if rn != rc:
                        roots[rc] = rn
                        cnt -= 1
            res.append(cnt)
        return res








s = Solution()
s.isBipartite([[1,3],[0,2],[1,3],[0,2]])