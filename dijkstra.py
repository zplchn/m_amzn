import heapq
from typing import List

'''
Dijkstra

When to use: single source in directed graph with weight, calculate shortest path (in terms of weight)

key: relexation. unlike bfs, when enqueue, a node might be enqueued multiple times, but each time has to be have a 
smaller distance. when a node has been pop out a heap, it guarantees shortest path

data structure:
heap as a queue, [(sum distance, x, y)]
visited: hashset store already confirmed shortest nodes
stopped: hashmap store for each node, currently known shortest dist

'''


class Solution:
    def shortestDistance505(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        # Dijkstra: core idea is single source + relexation. we can accept the queue contain multiple entry for same
        # node, inserted in descending dist from source(cug edges), and the minheap guarantee the node will be dequed
        # first time, it IS the shortest distance from source
        if not maze or not maze[0] or not start or not destination:
            return 0
        heap = [(0, start[0], start[1])]
        stopped = {(start[0], start[1]): 0} # keep current dist
        visited = set()  # keep nodes already confirmed to be shortest path

        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        while heap:
            dist, i, j = heapq.heappop(heap) # Dijsktra once a node is dequed from the pq, it is the shortest already
            # when the node is first time dequeued; the same node may dequed multiple times but only first time is
            # the shorted path from source
            if [i, j] == destination:
                return dist
            if (i, j) in visited:
                continue
            visited.add((i, j))
            for o in offsets:
                x, y = i, j
                d = 0
                while 0 <= x + o[0] < len(maze) and 0 <= y + o[1] < len(maze[0]) and maze[x + o[0]][y + o[1]] != 1:
                    x, y = x + o[0], y + o[1]
                    d += 1
                total = dist + d
                if (x, y) not in stopped or total < stopped[(x, y)]:
                    stopped[(x, y)] = total
                    heapq.heappush(heap, (total, x, y))
        return -1
