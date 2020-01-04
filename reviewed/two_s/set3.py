import collections
from typing import Optional, Any, List


# LRU LFU
# Stock 3 + Median of two sorted array debug
# Design ATM
#

class LRUCache:
    class Node:
        def __init__(self, key: Optional[str] = None, value: Optional[int] = None) -> None:
            self.key = key
            self.val = value
            self.pre = self.next = None

    def __init__(self, cap: int) -> None:
        if cap <= 0:
            raise ValueError
        self.hm = {}
        self.head = LRUCache.Node()
        self.tail = LRUCache.Node()
        self.cap = cap
        self.head.next = self.tail
        self.tail.pre = self.head

    def get(self, key: str) -> int:
        if key not in self.hm:
            return -1
        x = self.hm[key]
        self.move_to_head(x)
        return x.val

    def put(self, key: str, val: int) -> None:
        if key in self.hm:
            self.hm[key].val = val
            self.move_to_head(self.hm[key])
            return
        if len(self.hm) == self.cap:
            last = self.tail.pre
            del self.hm[last.key] # dont forget to delete from both hashmap and list
            self.delete_node(last)
        node = LRUCache.Node(key, val)
        self.insert_to_head(node)
        self.hm[key] = node

    def move_to_head(self, node: Node) -> None:
        if self.head == node:
            return
        self.delete_node(node)
        self.insert_to_head(node)

    def delete_node(self, node: Node) -> None:
        node.pre.next = node.next
        node.next.pre = node.pre

    def insert_to_head(self, node: Node) -> None:
        node.next = self.head.next
        node.next.pre = node
        self.head.next = node
        node.pre = self.head


class LFUCache:
    class Node:
        def __init__(self, k: int, v: int, count: int) -> None:
            self.k, self.v, self.count = k, v, count

    def __init__(self, capacity: int) -> None:
        self.hm_key = {} # k -> node
        self.hm_count = collections.defaultdict(collections.OrderedDict) # cnt -> {k -> node}
        self.cap = capacity
        self.minc = None

    def get(self, key: int) -> Optional[int]:
        if key not in self.hm_key:
            return None
        node = self.hm_key[key]
        self.bump_node(node)
        return node.v

    def put(self, key: int, val: int) -> None:
        if self.cap == 0:
            return
        if key in self.hm_key:
            node = self.hm_key[key]
            node.v = val
            self.bump_node(node)
            return
        if len(self.hm_key) == self.cap:
            k, v = self.hm_count[self.minc].popitem(last=False)
            del self.hm_key[k]
        self.minc = 1
        self.hm_key[key] = self.hm_count[self.minc][key] = LFUCache.Node(key, val, 1)

    def bump_node(self, node: Node) -> None:
        del self.hm_count[node.count][node.k]
        if not self.hm_count[self.minc]:
            self.minc += 1
        node.count += 1
        self.hm_count[node.count][node.k] = node

def maxProfit(self, prices: List[int]) -> int:
    if not prices:
        return 0
    dp = [0] * len(prices)
    lmin, lmax = prices[0], prices[-1]
    res = 0

    for i in range(1, len(prices)):
        dp[i] = max(dp[i - 1], prices[i] - lmin)
        lmin = min(lmin, prices[i])

    for i in reversed(range(len(prices) - 1)):
        res = max(res, dp[i] + max(0, lmax - prices[i]))
        lmax = max(lmax, prices[i])
    return res


def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    # This solutionn is based on cut half nums1 every time until either nums1 is empty or K == 1
    def kth(a, b, k):
        if len(a) > len(b):
            return kth(b, a, k)
        if not a:
            return b[k - 1]
        if k == 1:
            return min(a[0], b[0])
        offa = min(k // 2, len(a))
        offb = k - offa
        if a[offa - 1] == b[offb - 1]:
            return a[offa - 1]
        elif a[offa - 1] < b[offb - 1]:
            return kth(a[offa:], b, k - offa)
        else:
            return kth(a, b[offb:], k - offb)

    l1, l2 = len(nums1), len(nums2)
    n = l1 + l2
    return (kth(nums1, nums2, (n + 1) // 2) + kth(nums1, nums2, (n + 2) // 2)) / 2

def median(A, B):
    # This solution is based on how many items we can take out from A. We make A is the shorter one and binary search.
    m, n = len(A), len(B)
    if m > n:
        A, B, m, n = B, A, n, m
    if n == 0:
        raise ValueError

    imin, imax, half_len = 0, m, (m + n + 1) / 2

    while imin <= imax:
        i = (imin + imax) / 2
        j = half_len - i
        if i < m and B[j-1] > A[i]:
            # i is too small, must increase it
            imin = i + 1
        elif i > 0 and A[i-1] > B[j]:
            # i is too big, must decrease it
            imax = i - 1
        else:
            # i is perfect

            if i == 0: max_of_left = B[j-1]
            elif j == 0: max_of_left = A[i-1]
            else: max_of_left = max(A[i-1], B[j-1])

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])

            return (max_of_left + min_of_right) / 2.0

    '''
    test cases:
    [], [3]
    [3], []
    
    [3], [8, 9]
    [8], [3, 9]
    
    [3,4], [8, 9]
    [8, 9], [3, 4]
       
    
    '''

