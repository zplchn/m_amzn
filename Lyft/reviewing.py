import collections
import heapq
import bisect
from typing import List, Tuple


class Solution:
    def findItinerary587(self, tickets: List[List[str]]) -> List[str]:
        '''
        directed graph problems:

        course schedules : traverse by all nodes, using topological sort / BFS indegrees
        reconstruct itinerary: traverse by ALL edges, (circle allowed), using DFS
        '''
        def dfs(s: str):
            while graph[s]:
                dfs(graph[s].pop())
            res.append(s) # append after finishing all children, res is reversed order of the dfs path
        res = []
        # first create graph, order by starting nodes adjacency list, reverse ordered by ending point(for poping o(1))
        graph = collections.defaultdict(list)
        for s, e in tickets:
            graph[s].append(e)
        for k, v in graph.items():
            graph[k] = sorted(v, reverse=True)
        dfs('JFK')
        return res[::-1]
