class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = self.right = None


class Solution:

    def kthSmallest230(self, root: TreeNode, k: int) -> int:
        # inorder traversal until the kth
        if not root or k <= 0:
            return -1
        st = []
        while root or st:
            if root:
                st.append(root)
                root = root.left
            else:
                node = st.pop()
                k -= 1
                if k == 0:
                    return node.val
                root = node.right
        return -1
