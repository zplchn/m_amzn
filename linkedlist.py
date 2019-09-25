class Node:
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random

class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

class Solution:
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
        m, n = 0
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




















