from typing import List
import collections


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