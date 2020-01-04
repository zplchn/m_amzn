class LRUCache:

    class ListNode:
        def __init__(self, key=None, val=None):
            # python func does not support overloading! every same name func
            # replace previous one. Use default value and factory methord to construct condionally

            self.key = key  # used when try to delete node
            self.val = val
            self.pre = self.next = None

    def __init__(self, capacity: int):
        self.cap = capacity
        self.head, self.tail = self.ListNode(), self.ListNode()
        self.size = 0
        self.hm = {}
        self.head.next = self.tail
        self.tail.pre = self.head

    def get(self, key: int) -> int:
        if key not in self.hm:
            return -1
        res = self.hm[key]
        self.move_to_head(res)
        return res.val

    def put(self, key: int, value: int) -> None:
        if key in self.hm:
            x = self.hm[key]
            x.val = value
            self.move_to_head(x)
        else:
            if self.size == self.cap:
                del self.hm[self.tail.pre.key]  # dont forget to remove from hm
                self.delete_node(self.tail.pre)
                self.size -= 1

            node = self.ListNode(key, value)
            self.insert_to_head(node)
            self.hm[key] = node
            self.size += 1

    def insert_to_head(self, node):
        if self.head.next == node:
            return
        node.next = self.head.next
        self.head.next.pre = node
        node.pre = self.head
        self.head.next = node

    def move_to_head(self, node):
        self.delete_node(node)
        self.insert_to_head(node)

    def delete_node(self, node):
        node.pre.next = node.next
        node.next.pre = node.pre


class Solution:
    def calculate227(self, s: str) -> int:
        # use a stack, when everytime meet a op, do the calculation. for +-, simply push into stack
        # for */, pop from stack and do the math, then push back. At the end, sum all numbers in st.
        if not s:
            return 0
        s += '+0'
        st = []
        op = '+'
        num = 0
        for x in s:
            if x.isdigit():
                num = num * 10 + ord(x) - ord('0')
            elif x in '+-*/':
                if op == '+':
                    st.append(num)
                elif op == '-':
                    st.append(-num)
                elif op == '*':
                    st.append(st.pop() * num)
                else:
                    st.append(int(st.pop() / num)) # python 3 way of handling negative number division
                op = x
                num = 0 #dont forget to reset whenever meet a new op
        return sum(st)

    def countSubstrings647(self, s: str) -> int:
        if not s:
            return 0
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        res = 0
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    res += 1
        return res