import collections
from typing import List, Optional


class TreeNode:
    def __init__(self, val: int):
        self.val = val
        self.left = self.right = None


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


class NNode:
    def __init_(self, val, children):
        self.val = val
        self.children = children


class Node:
    def __def__(self, val, left, right, next=None, parent=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
        self.parent = parent

    def __init__(self, val, children):
        self.val = val
        self.children = children


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        def dfs(root):
            if not root:
                res.append('#')
                return
            res.append(str(root.val))
            dfs(root.left)
            dfs(root.right)

        res = []
        dfs(root)
        return ','.join(res)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        def dfs():
            x = q.popleft()
            if x == '#':
                return None
            root = TreeNode(int(x))
            root.left = dfs()
            root.right = dfs()
            return root

        if not data:
            return None
        q = collections.deque(data.split(','))
        return dfs()


class Codec428:
    # we still use preorder, however besides the val, we also store the children size

    def serialize(self, root: 'Node') -> str:
        """Encodes a tree to a single string.

        :type root: Node
        :rtype: str
        """
        def dfs(root: Node) -> None:
            res.append(str(root.val))
            res.append(str(len(root.children)))
            for c in root.children:
                dfs(c)

        if not root:
            return '#'
        res = []
        dfs(root)
        return '#'.join(res)

    def deserialize(self, data: str) -> 'Node':
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: Node
        """
        def dfs() -> Node:
            node = Node(int(q.popleft()), [])
            size = int(q.popleft())
            for _ in range(size):
                node.children.append(dfs())
            return node

        if data == '#':
            return None
        q = collections.deque(data.split('#'))
        return dfs()


class BSTIterator:

    def __init__(self, root: TreeNode):
        self.cur = root
        self.st = []

    def next(self) -> int:
        while self.cur:
            self.st.append(self.cur)
            self.cur = self.cur.left
        self.cur = self.st.pop()
        res = self.cur.val
        self.cur = self.cur.right
        return res

    def hasNext(self) -> bool:
        return self.cur is not None or len(self.st) > 0


class Trie:
    class TrieNode:
        def __init__(self):
            # use a dict instead of array of 26
            self.children = collections.defaultdict(Trie.TrieNode)
            self.is_word = False

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = self.TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for w in word:
            node = node.children[w] # dict[] create if node not exist
        node.is_word = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for w in word:
            if w not in node.children:
                return False
            node = node.children[w]
        return node.is_word

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for w in prefix:
            if w not in node.children:
                return False
            node = node.children[w]
        return True


class WordDictionary:
    class TrieNode:
        def __init__(self):
            self.children = collections.defaultdict(WordDictionary.TrieNode)
            self.is_word = False

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = self.TrieNode()

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
        def dfs(node, i):
            if i == len(word):
                return node.is_word
            if word[i] != '.':
                return word[i] in node.children and dfs(node.children[word[i]], i + 1)
            else:
                return any(dfs(node.children[k], i + 1) for k in node.children)
        return dfs(self.root, 0)


class MyCalendar729:
    # use a bst to simulate the treemap in java and achieve o nlogn avg
    class Node:
        def __init__(self, start, end):
            self.start = start
            self.end = end
            self.left = self.right = None

    def __init__(self):
        self.root = None

    def book(self, start: int, end: int) -> bool:
        def dfs(root) -> bool:
            if start >= root.end:
                if root.right:
                    return dfs(root.right)
                else:
                    root.right = self.Node(start, end)
                    return True
            elif end <= root.start:
                if root.left:
                    return dfs(root.left)
                else:
                    root.left = self.Node(start, end)
                    return True
            else:
                return False
        if not self.root:
            self.root = self.Node(start, end)
            return True
        return dfs(self.root)

class Solution:
    def serialize(self, root):
        def dfs(root, res):
            if not root:
                res.append('#')
                return
            res.append(str(root.val))
            dfs(root.left, res)
            dfs(root.right, res)

        res = []
        dfs(root, res)
        return ','.join(res)

    def deserialize(self, data):
        def dfs(q):
            x = q.popleft()
            if x == '#':
                return None
            root = TreeNode(int(x))
            root.left = dfs(q)
            root.right = dfs(q)
            return root

        if not data:
            return None
        q = collections.deque(data.split(','))
        return dfs(q)

    def rightSideView(self, root: TreeNode) -> List[int]:
        res = []
        if not root:
            return res
        q = collections.deque([root])
        cntc, cntn = 1, 0
        while q:
            x = q.popleft()
            cntc -= 1
            if x.left:
                q.append(x.left)
                cntn += 1
            if x.right:
                q.append(x.right)
                cntn += 1
            if cntc == 0:
                res.append(x.val)
                cntc, cntn = cntn, 0
        return res

    def sortedListToBST(self, head: ListNode) -> TreeNode:
        def dfs(l, r):
            if l > r:
                return None # dont forget end condition
            nonlocal head
            m = l + ((r - l) >> 1)
            lchild = dfs(l, m - 1)
            root = TreeNode(head.val)
            root.left = lchild
            head = head.next
            root.right = dfs(m + 1, r)
            return root

        if not head:
            return None
        # inorder solution space of height blanaced tree is this list
        len = 0
        cur = head
        while cur:
            len += 1
            cur = cur.next
        return dfs(0, len - 1)

    def widthOfBinaryTree(self, root: TreeNode) -> int:
        if not root:
            return 0
        level = [(root, 1)]
        maxv = 1
        while level:
            t = []
            width = level[-1][1] - level[0][1] + 1
            maxv = max(maxv, width)
            for node, x in level:
                if node.left:
                    t.append((node.left, x * 2))
                if node.right:
                    t.append((node.right, x * 2 + 1))
            level = t
        return maxv

    def isFullTree(self, root):
        if not root:
            return True
        if root.left and not root.right or root.right and not root.left:
            return False
        return self.isFullTree(root.left) and self.isFullTree(root.right)

    def pathSum1(self, root, sum):
        def dfs(root, combi, res, sum):
            if not root:
                return
            if not root.left and not root.right:
                if sum == root.val:
                    x = combi[:]
                    x.append(root.val)
                    res.append(x)
                return
            combi.append(root.val)
            dfs(root.left, combi, res, sum - root.val)
            dfs(root.right, combi, res, sum - root.val)
            combi.pop()

        res = []
        if not root:
            return res
        dfs(root, [], res, sum)
        return res

    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        # cannot do dfs, because it also requires left to right and up to down. if dfs, then it could be down to up
        # so use bfs

        res = []
        minv = maxv = 0
        if not root:
            return res
        hm = collections.defaultdict(list)
        q = collections.deque([(root, 0)])
        while q:
            node, x = q.popleft()
            minv = min(minv, x)
            maxv = max(maxv, x)

            hm[x].append(node.val)
            if node.left:
                q.append((node.left, x - 1))
            if node.right:
                q.append((node.right, x + 1))

        for i in range(minv, maxv + 1):
            if i in hm:
                res.append(hm[i])
        return res

    def boundaryOfBinaryTree(self, root: TreeNode) -> List[int]:
        def dfs_left(root):
            if not root or not root.left and not root.right:
                return
            res.append(root.val)
            if root.left:
                dfs_left(root.left)
            else:
                dfs_left(root.right)

        def dfs_leaves(root):
            if not root:
                return
            dfs_leaves(root.left)
            if not root.left and not root.right:
                res.append(root.val)
            dfs_leaves(root.right)

        def dfs_right(root):
            if not root or not root.left and not root.right:
                return
            if root.right:
                dfs_right(root.right)
            else:
                dfs_right(root.left)
            res.append(root.val)

        res = []
        if not root:
            return res
        if root.left or root.right:
            res.append(root.val)
        # if root not have a left/right subtree, then root itself is boundary. In other words, ignore the missing side
        dfs_left(root.left)
        dfs_leaves(root)
        dfs_right(root.right)
        return res

    def sortedListToBST1(self, head: ListNode) -> TreeNode:
        def dfs(i, j):
            nonlocal head
            if i > j:
                return None
            m = i + ((j - i) >> 1)
            l = dfs(i, m - 1)
            root = TreeNode(head.val)
            head = head.next
            root.left = l
            root.right = dfs(m + 1, j)
            return root

        if not head:
            return None
        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next
        return dfs(0, n - 1)

    def lowestCommonAncestorBT(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root
        l = self.lowestCommonAncestorBT(root.left, p, q)
        r = self.lowestCommonAncestorBT(root.right, p, q)
        if l and r:
            return root
        return l or r

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return root
        while root:
            if p.val > root.val and q.val > root.val:
                root = root.right
            elif p.val < root.val and q.val < root.val:
                root = root.left
            else:
                break
        return root

    def levelOrder(self, root: 'NNode') -> List[List[int]]:
        res = []
        if not root:
            return res
        q = collections.deque([root])
        cntc, cntn = 1, 0
        combi = []
        while q:
            node = q.popleft()
            cntc -= 1
            combi.append(node.val)
            for c in node.children:
                if c:
                    cntn += 1
                    q.append(c)
            if cntc == 0:
                res.append(combi)
                combi = []
                cntc, cntn = cntn, 0
        return res

    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if not root:
            return res
        q = collections.deque([root])
        cntc, cntn, rev = 1, 0, False
        combi = []

        while q:
            node = q.popleft()
            combi.append(node.val)
            cntc -= 1
            if node.left:
                q.append(node.left)
                cntn += 1
            if node.right:
                q.append(node.right)
                cntn += 1
            if cntc == 0:
                res.append(combi[::-1] if rev else combi)
                rev = not rev
                combi = []
                cntc, cntn = cntn, 0
        return res

    def longestUnivaluePath(self, root: TreeNode) -> int:
        def dfs(root):
            nonlocal maxv
            if not root:
                return 0
            l = dfs(root.left)
            r = dfs(root.right)
            l = l if root.left and root.left.val == root.val else 0
            r = r if root.right and root.right.val == root.val else 0
            maxv = max(maxv, 1 + l + r)
            return 1 + max(l, r)

        if not root:
            return 0
        maxv = 1
        dfs(root)
        return maxv

    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        def dfs(root, combi):
            if not root:
                return
            if not root.left and not root.right:
                c = combi[:]
                c.append(str(root.val))
                res.append(''.join(c))
                return
            combi.append(str(root.val) + '->')
            dfs(root.left, combi)
            dfs(root.right, combi)
            combi.pop()
        res = []
        if not root:
            return res
        dfs(root, [])
        return res

    def rightSideView2(self, root: TreeNode) -> List[int]:
        res = []
        if not root:
            return res
        q = collections.deque([root])
        cntc, cntn = 1, 0
        while q:
            node = q.popleft()
            cntc -= 1
            if node.left:
                q.append(node.left)
                cntn += 1
            if node.right:
                q.append(node.right)
                cntn += 1
            if cntc == 0:
                res.append(node.val)
                cntc, cntn = cntn, 0
        return res

    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        if root.right:
            root.right.next = root.next.left if root.next else None
        if root.left:
            root.left.next = root.right
        self.connect(root.right)
        self.connect(root.left)
        return root



    def levelOrder2(self, root: TreeNode) -> List[List[int]]:
        res = []
        if not root:
            return res
        q = collections.deque([root])
        cntc, cntn = 1, 0
        combi = []
        while q:
            node = q.popleft()
            cntc -= 1
            combi.append(node.val)
            if node.left:
                q.append(node.left)
                cntn += 1
            if node.right:
                q.append(node.right)
                cntn += 1
            if cntc == 0:
                res.append(combi)
                combi = []
                cntc, cntn = cntn, 0
        return res

    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        res = []
        if not root:
            return res
        q = collections.deque([root])
        cntc, cntn = 1, 0
        combi = []
        while q:
            node = q.popleft()
            cntc -= 1
            combi.append(node.val)
            if node.left:
                q.append(node.left)
                cntn += 1
            if node.right:
                q.append(node.right)
                cntn += 1
            if cntc == 0:
                res.append(combi)
                combi = []
                cntc, cntn = cntn, 0
        return res[::-1]

    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right:
            return sum == root.val
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)

    def pathSum2(self, root: TreeNode, sum: int) -> List[List[int]]:
        def dfs(root, combi, sum):
            if not root:
                return
            if not root.left and not root.right:
                if root.val == sum:
                    t = combi[:]
                    t.append(root.val)
                    res.append(t)
                return
            combi.append(root.val)
            dfs(root.left, combi, sum - root.val)
            dfs(root.right, combi, sum - root.val)
            combi.pop()

        res = []
        if not root:
            return res
        dfs(root, [], sum)
        return res

    def findSecondMinimumValue(self, root: TreeNode) -> int:
        #only on the subtree that equals root.val can exist a smaller value
        def dfs(root):
            nonlocal res
            if not root:
                return
            if minv < root.val < res: #larger subtree, no need traverse
                res = root.val
            elif root.val == minv:
                dfs(root.left)
                dfs(root.right)

        if not root:
            return -1
        res = float('inf')
        minv = root.val
        dfs(root)
        return res if res != float('inf') else -1

    def connect2(self, root: 'Node') -> 'Node':
        if not root:
            return root
        nextn = root.next
        while nextn and not nextn.left and not nextn.right:
            nextn = nextn.next
        nextn = nextn.left or nextn.right if nextn else nextn
        if root.right:
            root.right.next = nextn
        if root.left:
            root.left.next = root.right if root.right else nextn
        self.connect(root.right)
        self.connect(root.left)

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def dfs(root):
            nonlocal res
            if not root:
                return 0
            l = dfs(root.left)
            r = dfs(root.right)
            res = max(res, l + r + 1)
            return max(l, r) + 1
        if not root:
            return 0
        res = 0
        dfs(root)
        return res - 1

    def flatten(self, root: TreeNode) -> None:
        def dfs(root: TreeNode) -> None:
            nonlocal pre
            if not root:
                return
            if pre:
                pre.left = None
                pre.right = root
            pre = root
            t = root.right
            dfs(root.left)
            dfs(t)
        pre = None
        dfs(root)

    def countNodes(self, root: TreeNode) -> int:
        if not root:
            return 0
        tr = root
        ll = lr = 0
        while tr:
            ll += 1
            tr = tr.left
        tr = root
        while tr:
            lr += 1
            tr = tr.right
        if ll == lr:
            return 2 ** ll - 1
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)

    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        if not root or not p:
            return None
        res = None # this can be the last left (parent) or next right (child)
        while root:
            if root.val > p.val:
                res = root
                root = root.left
            else:
                root = root.right
        return res

    def inorderSuccessor2(self, node: 'Node') -> 'Node':
        if not node:
            return node
        # two cases: if node has right child, it's the right child's last left child
        # otherwise, it's the first parent where the cur is a left child
        if node.right:
            node = node.right
            while node and node.left:
                node = node.left
            return node
        else:
            while node.parent:
                if node.parent.left == node:
                    return node.parent
                node = node.parent
            return None # the right most node

    def averageOfLevels(self, root: TreeNode) -> List[float]:
        res = []
        if not root:
            return res
        q = collections.deque([root])
        cntc, cntn, sum, cntcc = 1, 0, 0, 1
        while q:
            node = q.popleft()
            sum += node.val
            cntc -= 1
            if node.left:
                cntn += 1
                q.append(node.left)
            if node.right:
                cntn += 1
                q.append(node.right)
            if cntc == 0:
                res.append(sum / cntcc)
                cntc = cntcc = cntn
                cntn = 0
                sum = 0
        return res

    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        def dfs(root, parent, depth):
            if not root:
                return
            hm[root.val] = (parent, depth)
            dfs(root.left, root, depth + 1)
            dfs(root.right, root, depth + 1)

        if not root:
            return False
        hm = {}
        dfs(root, None, 0)
        vx, vy = hm[x], hm[y]
        return vx[1] == vy[1] and vx[0] != vy[0]

    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        def is_same(rs, rt):
            if not rs:
                return not rt
            if not rt:
                return False
            return rs.val == rt.val and is_same(rs.left, rt.left) and is_same(rs.right, rt.right)

        if not s or not t:
            return False
        return is_same(s, t) or self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1:
            return t2
        if not t2:
            return t1 # for only one of them exist, just move the root is enough.

        t1.val += t2.val
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)
        return t1

    def pathSum(self, root: TreeNode, sum: int) -> int:
        # start from every node and do dfs, as and must take the current one. Then count an overall couner.
        def dfs(root, sum):
            if not root:
                return 0
            return (1 if sum == root.val else 0) + dfs(root.left, sum - root.val) + dfs(root.right, sum - root.val)

        if not root:
            return 0
        return dfs(root, sum) + self.pathSum(root.left, sum) + self.pathSum(root.right, sum)

    def tree2str(self, t: TreeNode) -> str:
        if not t:
            return ''
        res = str(t.val)
        if not t.left and not t.right:
            return res
        res += '(' + self.tree2str(t.left) + ')'
        if t.right:
            res += '(' + self.tree2str(t.right) + ')'
        return res

    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        if not root.left:
            return self.minDepth(root.right) + 1 # dont forget + 1 here
        if not root.right:
            return self.minDepth(root.left) + 1
        return 1 + min(self.minDepth(root.right), self.minDepth(root.left))

    def longestConsecutive(self, root: TreeNode) -> int:
        def dfs(root, parent, lmax):
            nonlocal maxv
            if not root:
                return
            if parent and parent.val + 1 == root.val:
                lmax += 1
                maxv = max(maxv, lmax)
            else:
                lmax = 1
            dfs(root.left, root, lmax)
            dfs(root.right, root, lmax)

        if not root:
            return 0
        maxv = 1
        dfs(root, None, 0)
        return maxv

    def closestValue(self, root: TreeNode, target: float) -> int:
        if not root:
            return 0
        minv = float('inf')
        res = root.val
        while root:
            delta = abs(root.val - target)
            if delta < minv:
                res = root.val
                minv = delta
            if target > root.val:
                root = root.right
            else:
                root = root.left
        return res

    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        def successor(root):
            root = root.right
            while root.left:
                root = root.left
            return root

        def predecessor(root):
            root = root.left
            while root.right:
                root = root.right
            return root

        if not root:
            return root
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        else:
            if not root.left and not root.right:
                root = None
            elif root.right:
                root.val = successor(root)
                root.right = self.deleteNode(root.right, root.val)
            else:
                root.val = predecessor(root)
                root.left = self.deleteNode(root.left, root.val)
        return root

    def isCompleteTree(self, root: TreeNode) -> bool:
        if not root:
            return True
        q = collections.deque([root])
        nc, nn = 1, 0
        has_none = False
        while q:
            node = q.popleft()
            nc -= 1
            if not node:
                has_none = True
                continue
            elif has_none: # there is a None before and then we meet a node
                return False
            q.append(node.left)
            q.append(node.right)
            nn += 2
            if nc == 0:
                nc, nn = nn, 0
        return True

    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        if not root:
            return res
        st = []
        while root or st:
            if root:
                st.append(root)
                root = root.left
            else:
                root = st.pop()
                res.append(root.val)
                root = root.right
        return res

    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        if not root:
            return res
        st = []
        while root or st:
            if root:
                res.append(root.val)
                st.append(root)
                root = root.left
            else:
                root = st.pop()
                root = root.right
        return res

    def preorder(self, root: 'Node') -> List[int]:
        res = []
        if not root:
            return res
        st = [root]
        while st:
            x = st.pop()
            res.append(x.val)
            st.extend(reversed(x.children))
        return res

    def widthOfBinaryTree662(self, root: TreeNode) -> int:
        # mark a number for each node as i, then left child would be 2i and right be 2i + 1
        if not root:
            return 0
        q = collections.deque([(root, 1)])
        res = 1

        while q:
            left = q[0][1]
            right = q[-1][1]
            n = len(q)
            for i in range(n): # for will take the new entries added to a list
                node, idx = q.popleft()
                if node.left:
                    q.append((node.left, idx * 2))
                if node.right:
                    q.append((node.right, idx * 2 + 1))
            res = max(res, right - left + 1)
        return res

    def construct427(self, grid: List[List[int]]) -> 'Node':
        class Node:
            def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
                self.val = val
                self.isLeaf = isLeaf
                self.topLeft = topLeft
                self.topRight = topRight
                self.bottomLeft = bottomLeft
                self.bottomRight = bottomRight

        # dfs, recursively check if all number is same within the square by setting i, j range. and dfs if not all same
        def dfs(i, j, n):
            if n == 0:
                return None
            for x in range(i, i + n):
                for y in range(j, j + n):
                    if grid[x][y] != grid[i][j]:
                        n //= 2
                        return Node(False, False, dfs(i, j, n), dfs(i, j + n, n), dfs(i + n, j, n), dfs(i + n, j + n,
                                                                                                        n))
            return Node(grid[i][j] == 1, True, None, None, None, None)
        if not grid or not grid[0]:
            return None
        return dfs(0, 0, len(grid))

    def sumRootToLeaf1022(self, root: TreeNode) -> int:
        def dfs(root, pre):
            nonlocal res
            if not root:
                return
            sum = pre * 2 + root.val
            if not root.left and not root.right:
                res += sum
                return
            dfs(root.left, sum)
            dfs(root.right, sum)
        res = 0
        if not root:
            return res
        dfs(root, 0)
        return res

    def postorderTraversal145(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        st = []
        res = []
        pre = None
        while root or st:
            if root:
                st.append(root)
                root = root.left
            else:
                if st[-1].right and pre != st[-1].right:
                    root = st[-1].right
                else:
                    pre = st.pop()
                    res.append(pre.val)
        return res

    def postorder590(self, root: 'Node') -> List[int]:
        def dfs(root: Node) -> None:
            if not root:
                return
            for c in root.children:
                dfs(c)
            res.append(root.val)

        res = []
        dfs(root)
        return res

    def maxLevelSum1161(self, root: TreeNode) -> int:
        if not root:
            return 0
        q = collections.deque([root])
        l = 1
        maxv = root.val
        res = 1
        while q:
            sumv = 0
            for i in range(len(q)):
                node = q.popleft()
                sumv += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            if sumv > maxv:
                maxv = sumv
                res = l
            l += 1

        return res

    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        return max((self.maxDepth(x) for x in root.children), default=0) + 1 # max needs default value if empty

    def countUnivalSubtrees250(self, root: TreeNode) -> int:
        def dfs(root: TreeNode) -> bool:
            nonlocal res
            if not root:
                return True
            lv = dfs(root.left)
            rv = dfs(root.right) # do not combine in the if because then right dfs might be short circuited
            if lv and rv and (root.left is None or root.left.val == root.val) and (
                    root.right is None or root.right.val == root.val):
                res += 1
                return True
            return False

        if not root:
            return 0
        res = 0
        dfs(root)
        return res

    def findDuplicateSubtrees652(self, root: TreeNode) -> List[TreeNode]:
        # we need a hm to store the value + structure, so need serialization. the serialzied string is preorder but,
        # we use post order to get subtrees first
        def dfs(root: TreeNode) -> str:
            if not root:
                return '#'
            lv = dfs(root.left)
            rv = dfs(root.right)
            cur = str(root.val) + lv + rv
            if hm[cur] == 1: # only output one of the duplicate
                res.append(root)
            hm[cur] += 1
            return cur

        if not root:
            return []
        res = []
        hm = collections.defaultdict(int)
        dfs(root)
        return res

    def verticalTraversal987(self, root: TreeNode) -> List[List[int]]:
        def dfs(root: TreeNode, x: int, l: int) -> None:
            if not root:
                return
            hm[x].append((l, root.val))
            dfs(root.left, x - 1, l + 1)
            dfs(root.right, x + 1, l + 1)

        if not root:
            return []
        hm = collections.defaultdict(list)
        dfs(root, 0, 0)
        res = []
        for x in sorted(hm.keys()):
            res.append([y for _, y in sorted(hm[x])])
        return res

    def findTarget653(self, root: TreeNode, k: int) -> bool:
        def dfs(root: TreeNode) -> bool:
            if not root:
                return False
            if k - root.val in hs:
                return True
            hs.add(root.val)
            return dfs(root.left) or dfs(root.right)

        if not root:
            return False
        hs = set()
        return dfs(root)

    def largestBSTSubtree333(self, root: TreeNode) -> int:
        def dfs(root: TreeNode):
            nonlocal maxv
            if not root:
                return True, 0, float('inf'), float('-inf') # is_bst, # of nodes, minv, maxv
            lb, ln, lmin, lmax = dfs(root.left)
            rb, rn, rmin, rmax = dfs(root.right)
            if lb and rb and lmax < root.val < rmin: # note none root return -infinity as maxv and +inf as minv
                size = ln + rn + 1
                maxv = max(maxv, size)
                return True, size, min(lmin, root.val), max(rmax, root.val)
            return False, 0, float('inf'), float('-inf')

        if not root:
            return 0
        maxv = 1
        dfs(root)
        return maxv

    def convertBST538(self, root: TreeNode) -> TreeNode:
        # think inorder traversal from right to left, then it's keep a accumulate sum and add to the node
        def dfs(root: TreeNode) -> None:
            nonlocal sumv
            if not root:
                return
            dfs(root.right)
            root.val += sumv
            # sumv += root.val now root.val is already the sum
            sumv = root.val
            dfs(root.left)
        sumv = 0
        dfs(root)
        return root

    def maxAncestorDiff1026(self, root: TreeNode) -> int:
        def dfs(root: TreeNode, minv: int, maxv: int) -> int:
            if not root:
                return maxv - minv # when at leaf or edge, do the math
            maxv = max(root.val, maxv)
            minv = min(root.val, minv)
            return max(dfs(root.left, minv, maxv), dfs(root.right, minv, maxv))
        if not root:
            return 0
        return dfs(root, root.val, root.val)

    def maximumAverageSubtree1120(self, root: TreeNode) -> float:
        def dfs(root: TreeNode):
            nonlocal maxv
            if not root:
                return 0, 0
            lsum, lcnt = dfs(root.left)
            rsum, rcnt = dfs(root.right)
            tsum = lsum + rsum + root.val
            tcnt = lcnt + rcnt + 1
            maxv = max(maxv, tsum / tcnt)
            return tsum, tcnt
        if not root:
            return 0
        # maxv = root.val cannot init to root.val as 10 -> 1. the avg cannot be = 10, it has to be avg'd
        maxv = float('-inf')
        dfs(root)
        return maxv

    def twoSumBSTs1214(self, root1: TreeNode, root2: TreeNode, target: int) -> bool:
        def convert_tree(root):
            if not root:
                return
            hs.add(root.val)
            convert_tree(root.left)
            convert_tree(root.right)

        def search_tree(root) -> bool:
            if not root:
                return False
            if target - root.val in hs:
                return True
            return search_tree(root.left) or search_tree(root.right)

        if not root1 or not root2:
            return False
        # convert one tree to set and then traverse another and check two sum
        hs = set()
        convert_tree(root1)
        return search_tree(root2)

    def bstToGst1038(self, root: TreeNode) -> TreeNode:
        def dfs(root: TreeNode) -> None:
            nonlocal sumv
            if not root:
                return
            dfs(root.right)
            if sumv == float('-inf'):
                sumv = root.val
            else:
                root.val = sumv
                sumv += root.val
            dfs(root.left)
        if not root:
            return root
        sumv = float('-inf')
        dfs(root)
        return root

    def sumOfLeftLeaves404(self, root: TreeNode) -> int:
        def dfs(root: TreeNode) -> None:
            nonlocal res
            if not root:
                return
            if root.left:
                if not root.left.left and not root.left.right:
                    res += root.left.val
                else:
                    dfs(root.left)
            dfs(root.right)
        if not root:
            return 0
        res = 0
        dfs(root)
        return res

    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)
        if val > root.val:
            root.right = self.insertIntoBST(root.right, val)
        else:
            root.left = self.insertIntoBST(root.left, val)
        return root

    def recoverTree99(self, root: TreeNode) -> None:
        def dfs(root: TreeNode) -> None:
            nonlocal pre, p1, p2
            if not root:
                return
            dfs(root.left)
            if pre and pre.val > root.val:
                if not p1:
                    p1, p2 = pre, root
                else:
                    p2 = root
            pre = root
            dfs(root.right)

        if not root:
            return root
        pre = p1 = p2 = None
        dfs(root)
        p1.val, p2.val = p2.val, p1.val

    def findFrequentTreeSum508(self, root: TreeNode) -> List[int]:
        def dfs(root: TreeNode) -> int:
            if not root:
                return 0
            lv, rv = dfs(root.left), dfs(root.right)
            sumv = root.val + lv + rv
            c[sumv] += 1
            return sumv

        if not root:
            return []
        c = collections.Counter()
        dfs(root)
        maxv = max(c.values())
        return [k for k, v in c.items() if v == maxv]

    def treeToDoublyList426(self, root: 'Node') -> 'Node':
        def dfs(root: Node) -> None:
            nonlocal pre, head
            if not root:
                return
            dfs(root.left)
            if not head:
                head = pre = root
            else:
                pre.right = root
                root.left = pre
                pre = root
            dfs(root.right)

        if not root:
            return root
        pre = head = None
        dfs(root)
        pre.right = head
        head.left = pre
        return head

    def printTree655(self, root: TreeNode) -> List[List[str]]:
        # the width of array is the width of tree(num of leaf nodes if it's the full tree), and height is height of tree
        # then divide and conquer and every time set value to the mid point in every interval
        def tree_depth(root: TreeNode) -> int:
            if not root:
                return 0
            return max(tree_depth(root.left), tree_depth(root.right)) + 1

        def dfs(root: TreeNode, l: int, r: int, level: int) -> None:
            if not root or l > r:
                return
            m = l + ((r - l) >> 1)
            res[level][m] = str(root.val)
            dfs(root.left, l, m - 1, level + 1)
            dfs(root.right, m + 1, r, level + 1)

        if not root:
            return []
        depth = tree_depth(root)
        res = [[''] * (2 ** depth - 1) for _ in range(depth)]
        dfs(root, 0, len(res[0]) - 1, 0)
        return res

    def verifyPreorder255(self, preorder: List[int]) -> bool:
        # like mergesort o(nlogn)
        # def dfs(start, end, minv, maxv):
        #     if start > end:
        #         return True
        #     if preorder[start] < minv or preorder[start] > maxv:
        #         return False
        #     i = start + 1
        #     while i < end and preorder[i] < preorder[start]:
        #         i += 1
        #     return dfs(start + 1, i - 1, minv, preorder[start] - 1) and dfs(i, end, preorder[start] + 1, maxv)
        #
        # if not preorder:
        #     return True
        # return dfs(0, len(preorder) - 1, float('-inf'), float('-inf'))

        # use a stack to achieve O(N). preorder does not uniquely identify a tree, but the q is whether this COULD be
        # a valid tree's perorder. so as long as we find a pattern root -> small -> large then it's good. Keep a
        # stack of nodes in left tree(descending) and pop when larger(in right tree), update the min threshold and check
        st = []
        low = float('-inf')
        for x in preorder:
            if x <= low:
                return False
            while st and x > st[-1]:
                low = st.pop()
            st.append(x)
        return True






























































