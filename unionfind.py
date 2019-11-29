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


s = Solution()
s.isBipartite([[1,3],[0,2],[1,3],[0,2]])