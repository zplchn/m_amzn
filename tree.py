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

    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        div = 1
        while x // div >= 10:
            div *= 10
        while div >= 10: # check div not x >= 10. to prevent cases like 1000000021
            if x // div != x % 10:
                return False
            x = x % div // 10
            div //= 100
        return True

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
        res = None # this can be the last left (parent) or next left (child)
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

    # def preorderTraversal(self, root: TreeNode) -> List[int]:

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



























































