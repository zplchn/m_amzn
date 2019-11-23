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

s = Solution()
s.isBipartite([[1,3],[0,2],[1,3],[0,2]])