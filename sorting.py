from typing import List
import random
import collections

class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


class Solution:
    def findKthLargest215(self, nums: List[int], k: int) -> int:
        # quick select O(N), worst case O(N2)
        def partition(l, r):
            pivot = random.randint(l, r)
            nums[pivot], nums[r] = nums[r], nums[pivot]
            t = l
            for i in range(l, r):
                if nums[i] < nums[r]:
                    nums[t], nums[i] = nums[i], nums[t]
                    t += 1
            nums[t], nums[r] = nums[r], nums[t]
            return t

        if not nums or k <= 0 or k > len(nums):
            return 0
        # now kth is kth smallest and 0 based. 1st largest in len = 1 is 0th smallest
        l, r, kth = 0, len(nums) - 1, len(nums) - k
        while True:
            pivot = partition(l, r)
            if pivot == kth:
                return nums[pivot]
            elif pivot < kth:
                l = pivot + 1
            else:
                r = pivot - 1

    def sortList(self, head: ListNode) -> ListNode:
        # merge sort. divide and merge two lists
        def merge(t1: ListNode, t2: ListNode) -> ListNode:
            dummy = ListNode(0)
            cur = dummy
            while t1 and t2:
                if t1.val < t2.val:
                    cur.next = t1
                    t1 = t1.next
                else:
                    cur.next = t2
                    t2 = t2.next
                cur = cur.next
            cur.next = t1 if t1 else t2
            return dummy.next

        if not head or not head.next:
            return head
        slow = fast = head
        while fast.next and fast.next.next: # note .next.next
            slow = slow.next
            fast = fast.next.next
        t1, t2 = head, slow.next
        slow.next = None
        t1 = self.sortList(t1)
        t2 = self.sortList(t2)
        return merge(t1, t2)

    def reorderLogFiles937(self, logs: List[str]) -> List[str]:
        # use a new tuple to force seperation of digit and str
        def custom_sort(log: str):
            key, rest = log.split(' ', 1)
            return (0, rest, key) if rest[0].isalpha() else (1,)

        return sorted(logs, key=custom_sort)

    def relativeSortArray1122(self, arr1: List[int], arr2: List[int]) -> List[int]:
        if not arr1 or not arr2:
            return []
        hm = {v: i for i, v in enumerate(arr2)}
        return sorted(arr1, key= lambda x: hm.get(x, 10000 + x)) # since range for x is less than 1000

    def highFive1086(self, items: List[List[int]]) -> List[List[int]]:
        if not items:
            return []
        hm = collections.defaultdict(list)
        minid, maxid = float('inf'), 0
        for id, s in items:
            hm[id].append(s)
            minid, maxid = min(id, minid), max(id, maxid)
        res = []
        for id in range(minid, maxid + 1):
            if id in hm:
                res.append([id, sum(sorted(hm[id], reverse=True)[:5]) // 5])
        return res

