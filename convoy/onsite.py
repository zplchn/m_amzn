import collections
import heapq
from typing import List


class Convoy:
    def watershed(self, matrix: List[List[int]]) -> List[List[int]]:
        '''

        [1, 3, 4]
        [2, 8, 9]
        [7, 6*, 5]

        Even though the condition given is 8 directions, but if 7 - 8 is a higher mountain link, 6 cannot pass
        So the solution is actually still using 4 directions

        :param matrix:
        :return:
        '''
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def is_sink(i: int, j: int) -> bool:
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] <= matrix[i][j]:
                    return False
            return True

        def bfs(si: int, sj: int):
            sink_v = matrix[si][sj]
            q = collections.deque([(si, sj)])
            hm_points[(si, sj)] = sink_v
            while q:
                i, j= q.popleft()
                hm_sinks[sink_v].add((i, j))
                for o in offsets:
                    x, y = i + o[0], j + o[1]
                    if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[x][y] > matrix[i][j]:
                        if (x, y) in hm_points:
                            if hm_points[(x, y)] <= sink_v:
                                continue
                            else:
                                hm_sinks[hm_points[(x, y)]].discard((x, y))
                        hm_points[(x, y)] = sink_v
                        q.append((x, y))

        if not matrix or not matrix[0]:
            return []

        hm_sinks = {} # sink value -> {(x, y)}
        hm_points = {} # (x, y) -> sink value

        # Find possible sink points and BFS from sink points
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if is_sink(i, j):
                    hm_sinks[matrix[i][j]] = set()
                    bfs(i, j)
                #     print(hm_points)
                #     print(hm_sinks)
                # print()
        return [list(s) for s in hm_sinks.values()]

    # wildcard matching
    def isMatch(self, s: str, p: str) -> bool:
        dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]
        dp[0][0]= True
        for j in range(len(p)):
            dp[0][j + 1] = dp[0][j] and p[j] == '*'
        for i in range(len(s)):
            for j in range(len(p)):
                if p[j] == '*':
                    dp[i + 1][j + 1] = dp[i + 1][j] or dp[i][j + 1]
                else:
                    dp[i + 1][j + 1] = (s[i] == p[j] or p[j] == '?') and dp[i][j]
        return dp[-1][-1]

    def numBusesToDestination815(self, routes: List[List[int]], S: int, T: int) -> int:
        # Create a graph stop -> [routes]. So problem become BFS when search fewest buses to take, each level is all
        # stops that can be reached by taking any bus from this stop. and then the next level is transfer at any stop to
        # another bus. And BFS shortest path will reach first to destination stop with fewest levels(each level is
        # all stops can be reached)
        if not routes or not routes[0] or S == T:
            return 0
        graph = collections.defaultdict(list)
        for r in range(len(routes)):
            for s in range(len(routes[r])): # here cannot use len(routes[0]) as each row length may be different
                graph[routes[r][s]].append(r)
        level = 0
        q = collections.deque([S])
        visited = set()
        while q:
            for i in range(len(q)):
                s = q.popleft()
                for r in graph[s]:
                    if r not in visited:
                        visited.add(r)
                        for stop in routes[r]:
                            if stop == T:
                                return level + 1
                            q.append(stop)
            level += 1
        return -1

    '''
        接口要自己定义，比如先定义一个rect class
        follow up 是如果多个矩形怎么判断，时间不太够了我就扯了点sort 完再扫一遍

        '''

    def isRectangleOverlap836(self, rec1: List[int], rec2: List[int]) -> bool:
        return not (rec2[0] >= rec1[2] or rec2[2] <= rec1[0] or rec1[1] >= rec2[3] or rec1[3] <= rec2[1])

    def findMin153(self, nums: List[int]) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l < r:
            m = l + ((r - l) >> 1)
            if nums[m] > nums[r]:
                l = m + 1
            else:
                r = m
        return nums[l]  # pay attention ask for return index or the element itself

    def findMin154(self, nums: List[int]) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l < r:
            m = l + ((r - l) >> 1)
            if nums[m] > nums[r]:
                l = m + 1
            elif nums[m] < nums[r]:
                r = m
            else:
                r -= 1
        return nums[l]

    def search33(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1

        while l <= r:
            m = l + ((r - l) >> 1)
            if target == nums[m]:
                return m
            if nums[m] < nums[r]:
                if nums[m] < target <= nums[r]:
                    l = m + 1
                else:
                    r = m - 1
            elif nums[m] > nums[r]:
                if nums[l] <= target < nums[m]:
                    r = m - 1
                else:
                    l = m + 1
            else:
                r -= 1
        return -1

    def topKFrequent692(self, words: List[str], k: int) -> List[str]:
        # O(n + klogn) time complexity
        if not words or k < 1:
            return []
        c = collections.Counter(words)
        # heap = []
        # for key, v in c.items():
        #     if len(heap) < k:
        #         heapq.heappush(heap, (v, key))
        #     else:
        #         heapq.heappushpop(heap, (v, key))
        # return [heapq.heappop(heap) for _ in range(k)][::-1] This does not work as key is reversed as well,
        # need to define customized class to wrap k, v pair and define __lt__(self, other) funtion to reverse the key
        heap = [(-v, k) for k, v in c.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(k)]


class WordDictionary:

    class TrieNode:
        def __init__(self):
            self.children = collections.defaultdict(WordDictionary.TrieNode)
            self.is_word = False

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = WordDictionary.TrieNode()

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        node = self.root
        for w in word:
            node = node.children[w]
        node.is_word = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        def dfs(i: int, node: WordDictionary.TrieNode) -> bool:
            if i == len(word):
                return node.is_word
            if word[i] != '.':
                return word[i] in node.children and dfs(i + 1, node.children[word[i]])
            else:
                return any(dfs(i + 1, v) for v in node.children.values())
        return dfs(0, self.root)

    def search_all_matched(self, word: str) -> List[str]:
        def dfs(i: int, combi: List[str], node: WordDictionary.TrieNode) -> None:
            if i == len(word):
                if node.is_word:
                    res.append(''.join(combi))
                return
            if word[i] != '?':
                if word[i] in node.children:
                    combi.append(word[i])
                    dfs(i + 1, combi, node.children[word[i]])
                    combi.pop()
            else:
                for k, v in node.children.items():
                    combi.append(k)
                    dfs(i + 1, combi, v)
                    combi.pop()
        res = []
        dfs(0, [], self.root)
        return res







c = Convoy()

input = [[10, 13, 14], [12, 18, 19], [17, 6, 5]]
res = c.watershed(input)
# for i, v in enumerate(res):
#     print(sorted(v))

wd = WordDictionary()
for s in ['a', 'ab', 'ac', 'abc']:
    wd.addWord(s)
print(wd.search_all_matched('a??'))



