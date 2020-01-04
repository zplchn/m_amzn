from typing import List, Tuple
import collections


class AutocompleteSystem:
    # 642 use a trie to store the historical words, and record the frequency

    class TrieNode:
        def __init__(self):
            self.children = collections.defaultdict(AutocompleteSystem.TrieNode)
            self.word = None
            self.rank = 0

    def add_word(self, word, time):
        node = self.root
        for w in word:
            node = node.children[w]
        node.word = word
        node.rank -= time

    def __init__(self, sentences: List[str], times: List[int]):
        self.root = self.TrieNode()
        self.keyword = ''
        for w, t in zip(sentences, times):
            self.add_word(w, t)

    def search(self, word):
        def dfs(root):
            if root.word:
                res.append((root.rank, root.word))
            for _, c in root.children.items():
                dfs(c) # children is a dict

        node = self.root
        res = []
        for w in word:
            if w not in node.children:
                return []
            node = node.children[w]
        dfs(node)
        return res

    def input(self, c: str) -> List[str]:
        if c == '#':
            self.add_word(self.keyword, 1)
            self.keyword = ''
            return []

        self.keyword += c
        res = self.search(self.keyword)
        return [r[1] for r in sorted(res)[:3]]


class Solution:

    def findWords212(self, board: List[List[str]], words: List[str]) -> List[str]:

        class TrieNode:
            def __init__(self):
                self.children = collections.defaultdict(TrieNode)
                self.is_word = False

        def build_trie() -> TrieNode:
            root = TrieNode()
            for word in words:
                node = root
                for w in word:
                    node = node.children[w]
                node.is_word = True
            return root

        def starts_with(prefix: str) -> Tuple[bool, TrieNode]:
            node = root
            for p in prefix:
                if p not in node.children:
                    return False, node
                node = node.children[p]
            return True, node

        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        def dfs(i: int, j: int, s: str):
            is_starts_with, node = starts_with(s)
            if not is_starts_with:
                return
            if node.is_word:
                res.append(s)
            if not node.children:
                return
            t = board[i][j]
            board[i][j] = '#'
            for o in offsets:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] != '#' and board[x][y] in \
                        node.children:
                    dfs(x, y, s + board[x][y])
            board[i][j] = t

        res = []
        if not board or not board[0] or not words:
            return res

        root = build_trie()

        for i in range(len(board)):
            for j in range(len(board[0])):
                if starts_with(board[i][j]):
                    dfs(i, j, board[i][j])
        return list(set(res)) # need to filter repeat words

    def replaceWords(self, dict: List[str], sentence: str) -> str:
        # 648 store the dict in a trie and check each word see if a prefix path exist, replace with shortest prefix
        class TrieNode:
            def __init__(self):
                self.children = collections.defaultdict(TrieNode)
                self.is_word = False

        def build_trie():
            root = TrieNode()

            for w in dict:
                node = root
                for c in w:
                    node = node.children[c]
                node.is_word = True
            return root

        def word_prefix(w: str):
            node = root
            prefix = []
            for c in w:
                if c in node.children:
                    prefix.append(c)
                    node = node.children[c]
                    if node.is_word:
                        break
            return ''.join(prefix)

        root = build_trie()
        return ' '.join(word_prefix(w) or w for w in sentence.split())












