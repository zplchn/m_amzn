import collections
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





