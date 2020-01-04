import collections
from typing import List

'''
Directed graph problems:

course schedules : traverse by all nodes, using topological sort / BFS indegrees
reconstruct itinerary: traverse by ALL edges, (circle allowed), using DFS


'''


class Solution:
    def alienOrder269(self, words: List[str]) -> str:
        # use the given dict, find the different char eg. wt, wf, so we know t comes before f. This is an edge t -> f.
        # so find all edges create a graph. then the problem becomes find topological sort of this graph.
        if not words:
            return ''
        graph = collections.defaultdict(list)
        chars = set(''.join(words))
        indegrees = {c: 0 for c in chars}

        for i in range(1, len(words)):
            for x, y in zip(words[i-1], words[i]): # wt wf -> zip object (w, w), (t, f)
                if x != y:
                    graph[x].append(y)
                    indegrees[y] += 1
                    break # dont forget to break. only first difference
        q = collections.deque([k for k, v in indegrees.items() if v == 0])
        # for topological sort using bfs, no need to use visited as if there is cycle, both in degrees never = 0
        res = []

        while q:
            k = q.popleft()
            res.append(k)
            for n in graph[k]:
                indegrees[n] -= 1
                if indegrees[n] == 0:
                    q.append(n)
        return ''.join(res) if len(res) == len(chars) else ''


