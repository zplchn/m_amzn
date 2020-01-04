import collections
import bisect
from typing import List, Tuple, Optional

'''
Word substrings in dict + Stack support min/max/avg/mode/median
News feed + Game of life
Refridgrator multithreaded


'''


class WordSubset:
    class TrieNode:
        def __init__(self):
            self.children = collections.defaultdict(WordSubset.TrieNode)
            self.is_word = False

    def __init__(self):
        self.root = self.TrieNode()

    def build_trie(self, word_dict: List[str]):
        for w in word_dict:
            root = self.root
            for c in w:
                root = root.children[c]
            root.is_word = True

    def get_prefix(self, root: TrieNode, c: str) -> Optional[TrieNode]:
        return root.children[c] if c in root.children else None

    def find_substrings(self, s: str, word_dict: List[str]) -> List[str]:
        res = []
        if not word_dict:
            return res
        self.build_trie(word_dict)

        for i in range(len(s)):
            node = self.root
            for j in range(i, len(s)):
                node = self.get_prefix(node, s[j])
                if node:
                    if node.is_word:
                        res.append(s[i: j + 1])
                else:
                    break
        return res


class MultiStack:
    def __init__(self):
        self.st = []
        self.minst = []
        self.maxst = []
        self.sumv = 0

        self.hm_key = collections.Counter()
        self.hm_count = collections.defaultdict(set)
        self.maxc = 0

        self.sorted_st = []

    def push(self, x: int) -> None:
        self.st.append(x)
        if not self.minst or x <= self.minst[-1]:
            self.minst.append(x)
        if not self.maxst or x >= self.maxst[-1]:
            self.maxst.append(x)
        self.sumv += x

        self.hm_count[self.hm_key[x]].discard(x)
        self.hm_key[x] += 1
        self.hm_count[self.hm_key[x]].add(x)
        self.maxc = max(self.maxc, self.hm_key[x])

        bisect.insort(self.sorted_st, x)

    def pop(self) -> None:
        x = self.st.pop()
        if x == self.minst[-1]:
            self.minst.pop()
        if x == self.maxst[-1]:
            self.maxst.pop()
        self.sumv -= x

        self.hm_count[self.hm_key[x]].remove(x)
        if not self.hm_count[self.maxc]:
            self.maxc -= 1
        self.hm_key[x] -= 1
        self.hm_count[self.hm_key[x]].add(x)

        self.sorted_st.pop(bisect.bisect(self.sorted_st, x) - 1)

    def top(self) -> int:
        return self.st[-1]

    def get_min(self) -> int:
        return self.minst[-1]

    def get_max(self) -> int:
        return self.maxst[-1]

    def get_avg(self) -> float:
        return self.sumv / len(self.st)

    def get_mode(self) -> List[int]:
        return list(self.hm_count[self.maxc])

    def get_median(self) -> float:
        n = len(self.sorted_st)
        return (self.sorted_st[n // 2] + self.sorted_st[(n - 1) // 2]) / 2


def game_of_life(board: List[List[int]]):
    # 1: live -> live 0: dead -> dead 2: live -> dead 3: dead -> live
    offsets = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
    for i in range(len(board)):
        for j in range(len(board[0])):
            cnt = 0
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] in (1, 2):
                    cnt += 1
            if board[i][j] == 1:
                if cnt < 2 or cnt > 3:
                    board[i][j] = 2
            elif cnt == 3:
                board[i][j] = 3
    for i in range(len(board)):
        for j in range(len(board[0])):
            board[i][j] %= 2





