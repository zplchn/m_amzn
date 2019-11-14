import random
from collections import OrderedDict, defaultdict
from typing import List


class Node:
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


class Solution382:

    def __init__(self, head: ListNode):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        self.head = head

    def getRandom(self) -> int:
        """
        Returns a random node's value.
        """
        # k = 1 resevoir sampling
        res = self.head
        cur = res.next
        count = 1
        while cur:
            if random.randint(0, count) == 0: #random find a node including cur and check if is < k (k = 1). if so
                # use it
                res = cur.val
            count += 1
            cur = cur.next
        return res


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


class LFUCache:
    class Node:
        def __init__(self, k, v, count):
            self.k, self.v, self.count = k, v, count

    def __init__(self, capacity):
        self.capacity = capacity
        self.hm_key = {}
        self.hm_count = defaultdict(OrderedDict)
        self.minc = None

    def get(self, key):
        if key not in self.hm_key:
            return -1
        node = self.hm_key[key]
        self.bump_node(node)
        return node.v

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self.hm_key:
            node = self.hm_key[key]
            node.v = value
            self.bump_node(node)
            return
        if len(self.hm_key) == self.capacity:
            k, v = self.hm_count[self.minc].popitem(last=False)
            del self.hm_key[k]
        self.hm_key[key] = self.hm_count[1][key] = self.Node(key, value, 1)
        self.minc = 1

    def bump_node(self, node: Node):
        del self.hm_count[node.count][node.k]
        if not self.hm_count[node.count]:
            del self.hm_count[node.count]

        if self.minc not in self.hm_count:
            self.minc += 1
        node.count += 1
        self.hm_count[node.count][node.k] = node


class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return head
        cur = head

        # 1. copy 1->1' -> 2 -> 2'
        while cur:
            node = Node(cur.val, None, None)
            node.next = cur.next
            cur.next = node
            cur = cur.next.next
        # 2. connect random
        cur = head
        while cur:
            cur.next.random = cur.random.next if cur.random else None
            cur = cur.next.next
        # 3. split
        cur = head
        dummy = pre = Node(0, None, None)
        while cur:
            pre.next = cur.next
            pre = pre.next
            cur.next = cur.next.next
            cur = cur.next
        return dummy.next

    def isPalindrome(self, head: ListNode) -> bool:
        if not head or not head.next:
            return True
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        pre = None
        while slow:
            next = slow.next
            slow.next = pre
            pre = slow
            slow = next
        while pre:
            if pre.val != head.val:
                return False
            pre, head = pre.next, head.next
        return True

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 or not l2:
            return l1 or l2
        dummy = cur = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next

    def addTwoNumbers1(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 or not l2:
            return l1 or l2
        dummy = cur = ListNode(0)
        carry = 0
        while l1 or l2 or carry:
            sum = carry
            if l1:
                sum += l1.val
                l1 = l1.next
            if l2:
                sum += l2.val
                l2 = l2.next
            cur.next = ListNode(sum % 10)
            cur = cur.next
            carry = sum // 10
        return dummy.next

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 or not l2:
            return l1 or l2
        st1, st2 = [], []
        while l1:
            st1.append(l1.val)
            l1 = l1.next
        while l2:
            st2.append(l2.val)
            l2 = l2.next
        carry = 0
        nextn = None
        while st1 or st2 or carry:
            sum = carry
            if st1:
                sum += st1.pop()
            if st2:
                sum += st2.pop()
            node = ListNode(sum % 10)
            carry = sum // 10
            node.next = nextn
            nextn = node
        return nextn

    def getIntersectionNode(self, headA, headB):
        if not headA or not headB:
            return None # return None if one not exist
        m, n = 0, 0
        curA, curB = headA, headB
        while curA:
            m += 1
            curA = curA.next
        while curB:
            n += 1
            curB = curB.next
        curA, curB = headA, headB
        while m - n > 0:
            curA = curA.next
            m -= 1
        while n - m > 0:
            curB = curB.next
            n -= 1
        while curA != curB:
            curA, curB = curA.next, curB.next
        return curA

    def plusOne(self, head: ListNode) -> ListNode:
        if not head:
            return head
        right, cur = None, head
        while cur:
            if cur.val != 9:
                right = cur
            cur = cur.next
        if not right:
            node = ListNode(1)
            node.next = head
            head = right = node
        else:
            right.val += 1  # dont forget to plus 1
        cur = right.next
        while cur:
            cur.val = 0
            cur = cur.next
        return head

    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        dummyo = curo = ListNode(0)
        dummye = cure = ListNode(0)
        i = 1
        while head:
            if i % 2 == 1:
                curo.next = head
                curo = curo.next
            else:
                cure.next = head
                cure = cure.next
            head = head.next
            i += 1
        curo.next = dummye.next
        cure.next = None
        return dummyo.next

    def partition(self, head: ListNode, x: int) -> ListNode:
        if not head:
            return head
        dummy_lt = clt = ListNode(0)
        dummy_gt = clg = ListNode(0)
        while head:
            if head.val < x:
                clt.next = head
                clt = clt.next
            else:
                clg.next = head
                clg = clg.next
            head = head.next
        clt.next = dummy_gt.next
        clg.next = None
        return dummy_lt.next

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        pre = dummy = ListNode(0)
        pre.next = head
        has_dup = False
        cur = head

        while cur:
            while cur.next and cur.val == cur.next.val:
                has_dup = True
                cur.next = cur.next.next
            if has_dup:
                pre.next = cur.next
            else:
                pre = cur
            cur = cur.next
            has_dup = False
        return dummy.next

    def swapPairs(self, head: ListNode) -> ListNode:
        if not head:
            return head
        dummy = pre = ListNode(0)
        dummy.next = cur = head

        while cur and cur.next:
            next = cur.next.next
            pre.next = cur.next
            cur.next.next = cur
            pre = cur
            cur = next
        pre.next = cur
        return dummy.next

    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        if not head:
            return head
        dummy = left = ListNode(0)
        dummy.next = head
        x = n - m
        while m > 1:
            left = left.next
            m -= 1
        cur = left.next
        pre = None
        while x >= 0:
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
            x -= 1
        left.next.next = cur
        left.next = pre
        return dummy.next

    def reorderList(self, head: ListNode) -> None:
        if not head or not head.next:
            return
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        pre = None
        cur = slow
        while cur:
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
        t = ListNode(0)
        n = 1
        while head and pre:
            if n % 2 == 1:
                t.next = head
                head = head.next
            else:
                t.next = pre
                pre = pre.next
            t = t.next
            n += 1

    def detectCycle(self, head: ListNode) -> ListNode:
        if not head:
            return head
        slow = fast = head
        has_cycle = False
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                has_cycle = True
                break
        if not has_cycle:
            return None
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow

    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        class Node:
            def __init__(self, val, next):
                self.val = val
                self.next = next

        if not head:
            head = Node(insertVal, None)
            head.next = head
            return head
        # two pointers, if pre <= insert <= cur, insert it; else if at pivot, insert >= pre or insert <= cur, insert
        # cyclic, while loop break when cur become head again
        pre, cur = head, head.next
        while cur != head:
            if pre.val <= insertVal <= cur.val:
                break
            if pre.val > cur.val and (insertVal >= pre.val or insertVal <= cur.val):
                break
            pre, cur = cur, cur.next
        pre.next = Node(insertVal, cur)
        return head

    def splitListToParts(self, root: ListNode, k: int) -> List[ListNode]:
        # the first N % K sub list will have an extra node. plus the avg of N // K
        res = [None] * k
        if not root:
            return res
        n = 0
        cur = root
        while cur:
            n += 1
            cur = cur.next
        avg, ext = n // k, n % k
        cur = root
        for i in range(k):
            res[i] = cur
            if cur:
                for j in range(avg + (1 if i < ext else 0) - 1):
                    cur = cur.next
                t = cur.next
                cur.next = None
                cur = t
        return res

    def nextLargerNodes(self, head: ListNode) -> List[int]:
        # maintain a st caching numbers and immmeditely pop once a larger number shows up. store index because number
        # can duplicate
        res = []
        if not head:
            return res
        while head:
            res.append(head.val)
            head = head.next
        st = []
        for i in range(len(res)):
            while st and res[st[-1]] < res[i]:
                res[st.pop()] = res[i]
            st.append(i)
        while st:
            res[st.pop()] = 0
        return res

























