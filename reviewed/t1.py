#!/usr/bin/env python
# encoding:utf-8
import heapq
from collections import deque
from typing import List, Optional
import collections
import random

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class TreeNode:
    def __init__(self, x) -> None:
        self.val = x
        self.left = self.right = None


class TestLL:
    def __init__(self, l) -> None:
        self.list = l

    def create_ll(self) -> ListNode:
        lh = ListNode(0)
        ln = lh

        for x in self.list:
            ln.next = ListNode(x)
            ln = ln.next
        return lh.next


def print_ll(ln) -> None:
    while ln is not None:
        print(ln.val, end=' -> ')
        ln = ln.next

class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.minstack = []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack.append(x)
        if not self.minstack or self.minstack[-1] >= x:
            self.minstack.append(x)

    def pop(self):
        """
        :rtype: None
        """
        if not self.stack:
            return None
        if self.stack.pop() == self.minstack[-1]:
            self.minstack.pop()

    def top(self) -> Optional[int]:
        """
        :rtype: int
        """
        return self.stack[-1] if self.stack else None

    def getMin(self):
        """
        :rtype: int
        """
        return self.minstack[-1] if self.minstack else None


class Solution:

    def mergeTwoLists(self, l1, l2):
        if not l1 or not l2:
            return l1 or l2

        head = cur = ListNode(0)

        while l1 is not None and l2 is not None:
            if l1.val <= l2.val:
               cur.next = l1
               l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        return head.next

    def isValid(self, s) -> bool:
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        parmap = {'(': ')', '[': ']', '{': '}'}

        for c in s:
            if c in parmap:
                stack.append(c)
            elif len(stack) == 0:
                return False
            else:
                e = stack.pop()
                if c != parmap[e]:
                    return False
        return len(stack) == 0

    def removeElement(self, nums, val):
        tl = tr = 0
        while tr < len(nums):
            if nums[tr] != val:
                nums[tl] = nums[tr]
                tl += 1
            tr += 1
        return tl

    def searchInsert(self, nums, target):
        l, r = 0, len(nums) - 1
        while l < r:
            m = l + (r - l) >> 1
            if nums[m] == target:
                return m
            elif nums[m] < target:
                l = m + 1
            else:
                r = m - 1
        return l + 1

    def maxSubArray(self, nums):
        if not nums:
            return 0
        lmax = amax = nums[0]
        for i in range(1, len(nums)):
            lmax = max(lmax + nums[i], nums[i])
            amax = max(lmax, amax)
        return amax

    def lengthOfLastWord(self, s: str):
        return len(s.rstrip().split(' ')[-1])

    def plusOne(self, digits: List[int]) -> List[int]:
        i = len(digits) - 1
        while i >= 0:
            if digits[i] == 9:
                digits[i] = 0
            else:
                break
            i -= 1
        if i < 0:
            t = [0] * (len(digits) + 1)
            t[0] = 1
            return t
        digits[i] += 1
        return digits

    def addBinary(self, a: str, b: str) -> str:
        if not a or not b:
            return a or b
        i, j, carry, sum = len(a) - 1, len(b) - 1, 0, []
        while i >= 0 or j >= 0 or carry > 0:
            t = 0
            if i >= 0:
                t += int(a[i])
                i -= 1
            if j >= 0:
                t += int(b[j])
                j -= 1
            t += carry
            carry = t // 2
            sum.append(str(t % 2))
        return ''.join(reversed(sum))

    def climbStairs(self, n: int) -> int:
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1], dp[2] = 1, 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head is None:
            return head
        cur = head
        while cur.next is not None:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        i, j, t = m - 1, n - 1, m + n - 1
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[t] = nums1[i]
                i -= 1
            else:
                nums1[t] = nums2[j]
                j -= 1
            t -= 1

        if j >= 0:
            nums1[:t + 1] = nums2[:j + 1]

    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not (p and q):
            return p is None and q is None
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    # def isSymmetric(self, root: TreeNode) -> bool:
    #     if root is None:
    #         return True
    #     return self.isSym(root.left, root.right)
    #
    # def isSym(self, left: TreeNode, right: TreeNode) -> bool:
    #     if not left or not right:
    #         return left == right
    #     return left.val == right.val and self.isSym(left.left, right.right) and self.isSym(left.right, right.left)

    def maxDepth(self, root):
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if not root:
            return res
        q = collections.deque([root])
        while q:
            combi = []
            for i in range(len(q)):
                node = q.popleft()
                combi.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(combi)
        return res

    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        q = collections.deque([root])
        cur, next = 1, 0
        lv, res = [], []

        while q:
            node = q.popleft()
            lv.append(node.val)
            cur -= 1
            if node.left:
                q.append(node.left)
                next += 1
            if node.right:
                q.append(node.right)
                next += 1
            if cur == 0:
                cur, next = next, 0
                res.append(lv)
                lv = []
        return res[::-1]

    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        return self.buildBST(nums, 0, len(nums) - 1)

    def buildBST(self, nums, start, end):
        if start > end:
            return None
        m = start + (end - start) >> 1
        root = TreeNode(nums[m])
        root.left = self.buildBST(nums, start, m - 1)
        root.right = self.buildBST(nums, m + 1, end)
        return root

    def isBalanced(self, root):
        return self.nullableHeight(root) is not None

    def nullableHeight(self, root):
        if root is None:
            return 0
        lheight = self.nullableHeight(root.left)
        if lheight is None:
            return None
        rheight = self.nullableHeight(root.right)
        if rheight is None:
            return None
        if abs(lheight - rheight) > 1:
            return None
        return max(lheight, rheight) + 1

    def minDepth(self, root):
        if root is None:
            return 0
        if not root.left:
            return self.minDepth(root.right) + 1
        if not root.right:
            return self.minDepth(root.left) + 1
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

    def hasPathSum(self, root, sum):
        return self.pathSum(root, 0, sum)

    def pathSum(self, root, csum, sum):
        if root is None:
            return False
        if root.left is None and root.right is None:
            return csum + root.val == sum
        return self.pathSum(root.left, csum + root.val, sum) or self.pathSum(root.right, csum + root.val, sum)

    def isPalindrome(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i < j:
            if not s[i].isalnum():
                i += 1
                continue
            if not s[j].isalnum():
                j -= 1
                continue
            if s[i].lower() != s[j].lower():
                return False
            i, j = i + 1, j - 1
        return True

    def singleNumber(self, nums: List[int]) -> int:
        x = 0
        for i in nums:
            x ^= i
        return x

    def hasCycle(self, head: ListNode) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        if not headA or not headB:
            return None
        ta, tb = headA, headB
        la, lb = 0, 0
        while ta:
            la += 1
            ta = ta.next
        while tb:
            lb += 1
            tb = tb.next
        while la - lb > 0:
            headA = headA.next
            la -= 1
        while lb - la > 0:
            headB = headB.next
            lb -= 1
        while headA != headB:
            headA, headB = headA.next, headB.next
        return headA

    def twoSum(self, numbers, target):
        i, j = 0, len(numbers) - 1
        while i < j:
            sum = numbers[i] + numbers[j]
            if sum == target:
                return [i + 1, j + 1]
            elif sum < target:
                i += 1
            else:
                j -= 1
        return [-1, -1]

    def majorityElement(self, nums):
        n, c = 0, None
        for x in nums:
            if n == 0:
                c = x
            n += 1 if c == x else -1
        return c

    def convertToTitle(self, n: int) -> str:
        res = []
        while n > 0:
            n -= 1
            res.append(chr(n % 26 + ord('A')))
            n //= 26
        return ''.join(res[::-1])

    def titleToNumber(self, s: str) -> int:
        res = 0
        for c in s:
            res = res * 26 + ord(c) - ord('A') + 1
        return res

    def trailingZeroes(self, n):
        res = 0
        while n >= 5:
            res += n // 5
            n //= 5
        return res

    def rotate(self, nums, k):
        n = len(nums)
        k = k % n
        self.reverse(nums, 0, n - k - 1)
        self.reverse(nums, n - k, n - 1)
        self.reverse(nums, 0, n - 1)

    def reverse(self, nums, start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start, end = start + 1, end - 1

    def reverseBits(self, n):
        res = 0
        for i in range(32):
            res = (res << 1) | (n & 1)
            n >>= 1
        return res

    def hammingWeight(self, n):
        res = 0
        for i in range(32):
            res += (n & 1)
            n >>= 1
        return res

    def isHappy(self, n):
        s = set()
        while n not in s:
            s.add(n)
            n = self.cal_square(n)
            if n == 1:
                return True
        return False

    def cal_square(self, x):
        t = 0
        while x != 0:
            t += (x % 10) * (x % 10)
            x //= 10
        return t

    def removeElements(self, head: ListNode, val: int) -> ListNode:
        t = dummy = ListNode(0)
        dummy.next = head
        while t.next:
            if t.next.val == val:
                t.next = t.next.next
            else:
                t = t.next
        return dummy.next

    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        d = {}
        for i in range(len(t)):
            if s[i] in d and d[s[i]] != t[i] or s[i] not in d and t[i] in d.values():
                return False
            d[s[i]] = t[i]
        return True

    def reverseList(self, head: ListNode) -> ListNode:
        prev, cur = None, head
        while cur:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next
        return prev

    def containsDuplicate(self, nums: List[int]) -> bool:
        s = set()
        for x in nums:
            if x in s:
                return True
            s.add(x)
        return False

    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        d = {}
        for i in range(len(nums)):
            if nums[i] in d and i - d[nums[i]] <= k:
                return True
            d[nums[i]] = i
        return False

    def isPowerOfTwo(self, n):
        return n > 0 and (n & (n - 1)) == 0

    def invertTree(self, root: TreeNode) -> Optional[TreeNode]:
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

    def deleteNode(self, node: ListNode) -> None:
        node.val = node.next.val
        node.next = node.next.next

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
        if not root or not p or not q:
            return None

        while root:
            if p.val > root.val and q.val > root.val:
                root = root.right
            elif p.val < root.val and q.val < root.val:
                root = root.left
            else:
                break
        return root

    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        d = {}
        for i in range(len(s)):
            d[s[i]] = 1 if s[i] not in d else d[s[i]] + 1
            d[t[i]] = -1 if t[i] not in d else d[t[i]] - 1
        return all(c == 0 for c in d.values())

    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        res = []
        self.bt_paths(root, res, '')
        return res

    def bt_paths(self, root: TreeNode, paths: List[str], pre: str) -> None:
        if not root:
            return
        if not root.left and not root.right:
            pre += str(root.val)
            paths.append(pre)
            return
        pre += str(root.val) + '->'
        self.bt_paths(root.left, paths, pre)
        self.bt_paths(root.right, paths, pre)

    def isUgly(self, num: int) -> bool:
        if num < 1:
            return False
        while num % 2 == 0:
            num //= 2
        while num % 3 == 0:
            num //= 3
        while num % 5 == 0:
            num //= 5
        return num == 1

    def missingNumber(self, nums: List[int]) -> int:
        x = len(nums)
        for i in range(len(nums)):
            x ^= i ^ nums[i]
        return x

    def isPalindrome(self, head: ListNode) -> bool:
        if not head:
            return True
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        prev, cur = slow, slow.next
        while cur:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next
        slow.next = None
        while prev:
            if prev.val != head.val:
                return False
            prev = prev.next
            head = head.next
        return True

    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return list(set(nums1) & set(nums2))

    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        d = {}
        res = []
        for x in nums1:
            d[x] = 1 if x not in d else d[x] + 1
        for y in nums2:
            if y in d and d[y] > 0:
                d[y] -= 1
                res.append(y)
        return res

    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        if not root:
            return 0
        if root.left and not root.left.left and not root.left.right:
            return root.left.val + self.sumOfLeftLeaves(root.right)
        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

    def reverseString(self, s: List[str]) -> None:
        i, j = 0, len(s) - 1
        while i < j:
            s[i], s[j] = s[j], s[i]
            i, j = i + 1, j - 1

    def findTheDifference(self, s: str, t: str) -> str:
        cnt = [0] * 26
        ca = ord('a')
        for x in s:
            cnt[ord(x) - ca] += 1
        for y in t:
            cnt[ord(y) - ca] -= 1
            if cnt[ord(y) - ca] < 0:
                return y
        return ''

    def reverseVowels(self, s: str) -> str:
        l = list(s)
        i, j = 0, len(s) - 1
        while i < j:
            if l[i] not in 'aeiouAEIOU':
                i += 1
            elif l[j] not in 'aeiouAEIOU':
                j -= 1
            else:
                l[i], l[j] = l[j], l[i]
                i, j = i + 1, j - 1
        return ''.join(l)

    def longestPalindrome(self, s: str) -> int:
        d = {}
        for x in s:
            d[x] = 1 if x not in d else d[x] + 1
        res = 0
        has_odd = False
        for y in d:
            res += d[y]
            if d[y] % 2 == 1:
                res -= 1
                has_odd = True
        return res + (1 if has_odd else 0)

    def addStrings(self, num1: str, num2: str) -> str:
        res = []
        i, j, carry = len(num1) - 1, len(num2) - 1, 0
        v0 = ord('0')
        while i >= 0 or j >= 0 or carry > 0:
            sum = ((ord(num1[i]) - v0) if i >= 0 else 0) + ((ord(num2[j]) - v0) if j >= 0 else 0) + carry
            res.append(str(sum % 10))
            i, j, carry = i - 1, j - 1, sum // 10
        return ''.join(res[::-1])

    def thirdMax(self, nums: List[int]) -> int:
        first = second = third = float('-inf')
        for x in nums:
            if x > first:
                first, second, third = x, first, second
            elif first > x > second:
                second, third = x, second
            elif second > x > third:
                third = x
        return first if third == float('-inf') else third

    class Node:
        def __init__(self, val, children):
            self.val = val
            self.children = children

    def NlevelOrder(self, root: 'Node') -> List[List[int]]:
        res, lc = [], []
        if not root:
            return res
        q = deque([root])
        cc, cn = 1, 0

        while q:
            node = q.popleft()
            lc.append(node.val)
            cc -= 1
            children = [c for c in node.children if c]
            q.extend(children)
            cn += len(children)

            if cc == 0:
                cc, cn = cn, 0
                res.append(lc)
                lc = []
        return res

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
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
            carry = sum // 10
            cur = cur.next
        return dummy.next

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if not head:
            return head
        dummy = left = right = ListNode(0)
        while right and n > 0:
            right = right.next
            n -= 1
        if n > 0:
            return head

        while right:
            left, right = left.next, right.next
        left.next = left.next.next
        return dummy.next

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        dummy = cur = ListNode(0)
        dummy.next = head

        while cur:
            t = cur.next
            while t and t.next and t.val == t.next.val:
                t = t.next
            if cur.next != t:
                cur.next = t.next
            else:
                cur = cur.next
        return dummy.next

    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        dummy = pre = ListNode(0)
        dummy.next = cur = head

        while cur and cur.next:
            next = cur.next.next
            pre.next = cur.next
            pre.next.next = cur
            cur.next = next
            pre, cur = cur, next
        return dummy.next

    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return head
        dummy = pre = ListNode(0)
        dummy.next = head
        cur = head
        n = 0
        while cur:
            cur = cur.next
            n += 1
        k = k % n
        cur = dummy
        while k > 0:
            cur = cur.next
            k -= 1
        while cur.next:
            pre, cur = pre.next, cur.next
        cur.next = dummy.next
        dummy.next = pre.next
        pre.next = None
        return dummy.next

    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        return self.isSym(root.left, root.right)

    def isSym(self, left: TreeNode, right: TreeNode) -> bool:
        if not left:
            return not right
        if not right:
            return False
        return left.val == right.val and self.isSym(left.left, right.right) and self.isSym(left.right, right.left)

    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        if not root:
            return res
        q = deque([root])
        cc, cn = 1, 0
        t = []
        should_reverse = False

        while q:
            node = q.popleft()
            t.append(node.val)
            cc -= 1
            if node.left:
                q.append(node.left)
                cn += 1
            if node.right:
                q.append(node.right)
                cn += 1
            if cc == 0:
                if should_reverse:
                    t = t[::-1]
                res.append(t)
                should_reverse = not should_reverse
                cc, cn = cn, 0
                t = []
        return res

    def buildTree1(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or len(preorder) != len(inorder):
            return None
        hm = {}
        for i in range(len(inorder)):
            hm[inorder[i]] = i
        return self.build1(preorder, 0, len(preorder) - 1, inorder, 0, len(inorder) - 1, hm)

    def build1(self, pre: List[int], pl: int, pr: int, ino: List[int], il: int, ir: int, hm: dict):
        if il > ir:
            return None
        root = TreeNode(pre[pl])
        k = hm[pre[pl]]
        root.left = self.build1(pre, pl + 1, pl + k - il, ino, il, k - 1, hm)
        root.right = self.build1(pre, pl + k - il + 1, pr, ino, k + 1, ir, hm)
        return root

    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder or len(inorder) != len(postorder):
            return None
        hm = dict()
        for i in range(len(inorder)):
            hm[inorder[i]] = i
        return self.build(inorder, 0, len(inorder) - 1, postorder, 0, len(postorder) - 1, hm)

    def build(self, ino, il, ir, post, pl, pr, hm):
        if il > ir:
            return None
        root = TreeNode(post[pr])
        k = hm[post[pr]]
        root.left = self.build(ino, il, k - 1, post, pl, pl + k - il - 1, hm)
        root.right = self.build(ino, k + 1, ir, post, pl + k - il, pr - 1, hm)
        return root

    def pathSum1(self, root: TreeNode, sum: int) -> List[List[int]]:
        res = []
        if not root:
            return res
        self.path_sum_helper(root, sum, [], res)
        return res

    def path_sum_helper(self, root: TreeNode, sum: int, combi: List[int], res: List[List[int]]) -> None:
        if not root:
            return
        if not root.left and not root.right:
            if sum == root.val:
                l = combi[:]
                l.append(root.val)
                res.append(l)
            return
        combi.append(root.val)
        self.path_sum_helper(root.left, sum - root.val, combi, res)
        self.path_sum_helper(root.right, sum - root.val, combi, res)
        combi.pop()

    def __init__(self):
        self.pre = None

    def flatten(self, root: TreeNode) -> None:
        if not root:
            return
        if self.pre:
            self.pre.right = root
            self.pre.left = None
        t = root.right
        self.pre = root
        self.flatten(root.left)
        self.flatten(t)

    class NumArray:

        def __init__(self, nums: List[int]):
            self.sumleft = [0] * len(nums)
            if len(nums) > 0:
                self.sumleft[0] = nums[0]
                for i in range(1, len(nums)):
                    self.sumleft[i] = self.sumleft[i - 1] + nums[i]

        def sumRange(self, i: int, j: int) -> int:
            if i < 0 or j < 0 or i > j:
                return 0
            return self.sumleft[j] - (self.sumleft[i - 1] if i > 0 else 0)


    class RandomizedSet:

        def __init__(self):
            """
            Initialize your data structure here.
            """
            self.list = []
            self.dict = {}

        def insert(self, val: int) -> bool:
            """
            Inserts a value to the set. Returns true if the set did not already contain the specified element.
            """
            if val not in self.dict:
                self.list.append(val)
                self.dict[val] = len(self.list) - 1
                return True
            return False

        def remove(self, val: int) -> bool:
            """
            Removes a value from the set. Returns true if the set contained the specified element.
            """
            if val not in self.dict:
                return False
            k = self.dict[val]
            self.list[k] = self.list[-1]
            self.dict[self.list[k]] = k
            self.list.pop()
            del self.dict[val]
            return True

        def getRandom(self) -> int:
            """
            Get a random element from the set.
            """
            return random.choice(self.list)



class Solution2:

    class Node:
        def __init__(self, val, left, right, next):
            self.val = val
            self.left = left
            self.right = right
            self.next = next

    def connect1(self, root: Node):
        if not root:
            return None
        if root.left:
            root.left.next = root.right
        if root.right and root.next:
            root.right.next = root.next.left
        self.connect1(root.left)
        self.connect1(root.right)
        return root

    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        t = root.next
        while t and not t.left and not t.right:
            t = t.next
        t = t.left or t.right if t else None
        if root.left:
            root.left.next = root.right if root.right else t
        if root.right:
            root.right.next = t
        self.connect(root.right)
        # remember to connect right first before left cuz right part needs to be connected first
        self.connect(root.left)
        return root

    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        if not nums:
            return res
        marker = [False] * len(nums)
        self.permute_helper(nums, 0, [], marker, res)
        return res

    def permute_helper(self, nums, i, combi, marker, res):
        if i == len(nums):
            res.append(combi[:])
            return
        for j in range(len(nums)):
            if marker[j]:
                continue
            combi.append(nums[j])
            marker[j] = True
            self.permute_helper(nums, i + 1, combi, marker, res)
            marker[j] = False
            combi.pop()

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        if not nums:
            return res
        nums.sort()
        self.dfs(nums, [], [False] * len(nums), res)
        return res

    def dfs(self, nums, combi, used, res):
        if len(combi) == len(nums):
            res.append(combi[:])
            return
        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            combi.append(nums[i])
            used[i] = True
            self.dfs(nums, combi, used, res)
            used[i] = False
            combi.pop()

    def nextPermutation(self, nums: List[int]) -> None:
        if not nums or len(nums) == 1:
            return
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i < 0:
            nums.reverse()
        j = len(nums) - 1
        while j > i and nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
        nums[i+1:] = nums[:i:-1]

    def islandPerimeter(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        sum = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == 0:
                    continue
                if j == 0 or grid[i][j - 1] == 0:
                    sum += 1
                if j == len(grid[i]) - 1 or grid[i][j + 1] == 0:
                    sum += 1
                if i == 0 or grid[i - 1][j] == 0:
                    sum += 1
                if i == len(grid) - 1 or grid[i + 1][j] == 0:
                    sum += 1
        return sum

    def letterCombinations(self, digits: str) -> List[str]:
        res = []
        if not digits:
            return res
        phone = {'2': ['a', 'b', 'c'],
                 '3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],
                 '5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],
                 '7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],
                 '9': ['w', 'x', 'y', 'z']}

        def dfs(combi):
            if len(combi) == len(digits):
                res.append(''.join(combi))
                return
            c = digits[len(combi)]
            for x in phone[c]:
                combi.append(x)
                dfs(combi)
                combi.pop()

        dfs([])
        return res

    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        if n <= 0:
            return res

        def dfs(left, right, combi):
            if right == n:
                res.append(''.join(combi))
                return
            if left < n:
                combi.append('(')
                dfs(left + 1, right, combi)
                combi.pop()
            if right < left:
                combi.append(')')
                dfs(left, right + 1, combi)
                combi.pop()
        dfs(0, 0, [])
        return res

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        if not candidates:
            return res
        candidates.sort()

        def dfs(i, sum, combi):
            if sum == target:
                res.append(combi[:])
                return
            for j in range(i, len(candidates)):
                if j > i and candidates[j] == candidates[j-1]:
                    continue
                if sum + candidates[j] <= target:
                    combi.append(candidates[j])
                    dfs(j, sum + candidates[j], combi)
                    combi.pop()
                else:
                    break
        dfs(0, 0, [])
        return res

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        if not candidates:
            return res
        candidates.sort()

        def dfs(i, sum, combi):
            if sum == target:
                res.append(combi[:])
                return
            for j in range(i, len(candidates)):
                if j > i and candidates[j] == candidates[j-1]:
                    continue
                if sum + candidates[j] <= target:
                    combi.append(candidates[j])
                    dfs(j + 1, sum + candidates[j], combi)
                    combi.pop()

        dfs(0, 0, [])
        return res

    def combine(self, n: int, k: int) -> List[List[int]]:

        def dfs(i, combi):
            if len(combi) == k:
                res.append(combi[:])
                return
            for j in range(i, n + 1):
                combi.append(j)
                dfs(j + 1, combi)
                combi.pop()

        res = []
        if n < 1 or k < 1 or k > n:
            return res
        dfs(1, [])
        return res

    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        if not nums:
            return res
        res.append([])
        for x in nums:
            n = len(res)
            for j in range(n):
                l = res[j][:]
                l.append(x)
                res.append(l)
        return res

    def exist(self, board: List[List[str]], word: str) -> bool:

        def dfs(i, j, k):
            if k == len(word):
                return True
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
                return False
            t = board[i][j]
            board[i][j] = '#'
            exist = dfs(i - 1, j, k + 1) or dfs(i + 1, j, k + 1) or dfs(i, j - 1, k + 1) or dfs(i, j + 1, k + 1)
            board[i][j] = t
            return exist

        if not board or not board[0]:
            return word == ''
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == word[0] and dfs(i, j, 0):
                    return True
        return False

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        if not nums:
            return res
        res.append([])
        nums.sort()
        l = 0
        for i in range(len(nums)):
            start = l if i > 0 and nums[i] == nums[i - 1] else 0
            l = len(res)
            for j in range(start, l):
                t = res[j][:]
                t.append(nums[i])
                res.append(t)
        return res

    def restoreIpAddresses(self, s: str) -> List[str]:

        def is_valid(s):
            return 0 < len(s) <= 3 and (s[0] != '0' if len(s) > 1 else True) and int(s) <= 255

        def dfs(i, combi):
            if len(combi) == 3:
                if not is_valid(s[i:]):
                    return
                t = '.'.join(combi) + '.' + s[i:]
                res.append(t)
                return res
            for j in range(1, 4):
                str = s[i: i + j]
                if is_valid(str):
                    combi.append(str)
                    dfs(i + j, combi)
                    combi.pop()

        res = []
        if len(s) < 4:
            return res
        dfs(0, [])
        return res

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:

        def dfs(i, sum, combi, k):
            if k == 0:
                if sum == n:
                    res.append(combi[:])
                return
            for x in range(i, 10):
                if sum + x <= n:
                    combi.append(x)
                    dfs(x + 1, sum + x, combi, k - 1)
                    combi.pop()
                else:
                    break

        res = []
        if k < 1:
            return res
        dfs(1, 0, [], k)
        return res

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        if len(nums) < 3:
            return res
        nums.sort()

        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            j, k = i + 1, len(nums) - 1
            while j < k:
                sum = nums[i] + nums[j] + nums[k]
                if sum == 0:
                    res.append([nums[i], nums[j], nums[k]])
                    j, k = j + 1, k - 1
                    while j < k and nums[j] == nums[j-1]:
                        j += 1
                    while j < k and nums[k] == nums[k+1]:
                        k -= 1
                elif sum < 0:
                    j += 1
                else:
                    k -= 1
        return res

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        if len(nums) < 4:
            return []
        nums.sort()
        res = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            for j in range(i + 1, len(nums)):
                if j > i + 1 and nums[j] == nums[j-1]:
                    continue
                k, z = j + 1, len(nums) - 1
                while k < z:
                    sum = nums[i] + nums[j] + nums[k] + nums[z]
                    if sum < target:
                        k += 1
                    elif sum > target:
                        z -= 1
                    else:
                        res.append([nums[i], nums[j], nums[k], nums[z]])
                        k, z = k + 1, z - 1
                        while k < z and nums[k] == nums[k-1]:
                            k += 1
                        while k < z and nums[z] == nums[z+1]:
                            z -= 1
        return res

    def removeElement(self, nums: List[int], val: int) -> int:
        if not nums:
            return 0
        i, j = 0, 0
        while j < len(nums):
            if nums[j] != val:
                nums[i] = nums[j]
                i += 1
            j += 1
        return i

    def trap(self, height: List[int]) -> int:
        pass

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        min = float('inf')
        res = 0
        for i in range(len(nums)):
            j, k = i + 1, len(nums) - 1
            while j < k:
                sum = nums[i] + nums[j] + nums[k]
                if abs(target - sum) < min:
                    min = abs(target - sum)
                    res = sum
                if sum < target:
                    j += 1
                elif sum > target:
                    k -= 1
                else:
                    break
        return res

    def maxArea(self, height: List[int]) -> int:
        if len(height) < 2:
            return 0
        i, j, max = 0, len(height) - 1, 0
        while i < j:
            area = (j - i) * min(height[i], height[j])
            if area > max:
                max = area
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return max

    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        i = 0
        for j in range(1, len(nums)):
            if nums[i] != nums[j]:
                i += 1
                nums[i] = nums[j]
        return i + 1

    def sortColors(self, nums: List[int]) -> None:
        if not nums:
            return
        i, j, k = 0, 0, len(nums) - 1
        while j <= k:
            if nums[j] == 0:
                nums[i], nums[j] = nums[j], nums[i]
                i, j = i + 1, j + 1
            elif nums[j] == 2:
                nums[j], nums[k] = nums[k], 2
                k -= 1
            else:
                j += 1

    def moveZeroes(self, nums: List[int]) -> None:
        i = 0
        for j in range(len(nums)):
            if nums[j] != 0:
                nums[i] = nums[j]
                i += 1
        while i < len(nums):
            nums[i] = 0
            i += 1

    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        hm = {}
        for x in s:
            hm[x] = 1 if x not in hm else hm[x] + 1
        for y in t:
            if y not in hm:
                return False
            hm[y] -= 1
            if hm[y] < 0:
                return False
        return all(v == 0 for v in hm.values())

    def containsDuplicate(self, nums: List[int]) -> bool:
        hm = set()
        for x in nums:
            if x in hm:
                return True
            hm.add(x)
        return False

    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        if k < 1:
            return False
        hm = {}
        for i in range(len(nums)):
            if nums[i] in hm and i - hm[nums[i]] <= k:
                return True
            hm[nums[i]] = i
        return False

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = []
        if not strs:
            return res
        dict = collections.defaultdict(list)
        for str in strs:
            dict[tuple(sorted(str))].append(str)
        return list(dict.values())

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        if len(board) != 9 or len(board[0]) != 9:
            return False
        for i in range(9):
            s = set()
            for j in range(9):
                if board[i][j] != '.':
                    if board[i][j] in s:
                        return False
                    s.add(board[i][j])

        for j in range(9):
            s = set()
            for i in range(9):
                if board[i][j] != '.':
                    if board[i][j] in s:
                        return False
                    s.add(board[i][j])

        for k in range(9):
            s = set()
            for i in range(k//3*3, k//3*3 + 3):
                for j in range(k%3*3, k%3*3 + 3):
                    if board[i][j] != '.':
                        if board[i][j] in s:
                            return False
                        s.add(board[i][j])
        return True

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        if len(nums) < k:
            return []
        c = collections.Counter(nums)
        return [x for x, _ in c.most_common(k)]

    def frequencySort(self, s: str) -> str:
        if not s:
            return s
        c = collections.Counter(s)
        res = []
        for k, v in c.most_common():
            res.extend([k] * v)
        return ''.join(res)

    def firstUniqChar(self, s: str) -> int:
        if not s:
            return -1
        c = collections.Counter(s)
        for i in range(len(s)):
            if c[s[i]] == 1:
                return i
        return -1

    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        for c in "!?',;.":
            paragraph = paragraph.replace(c, ' ')
        words = paragraph.lower().split()
        c = collections.Counter(words)
        banset = set(banned)
        res, max = '', 0
        for k, v in c.items():
            if v > max and k not in banset:
                res, max = k, v
        return res

    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        hm = {}
        start, i, res = 0, 0, 0
        while i < len(s):
            if s[i] in hm and hm[s[i]] >= start:
                res = max(res, i - start)
                start = hm[s[i]] + 1
            hm[s[i]] = i
            i += 1
        return max(res, i - start)

    def simplifyPath(self, path: str) -> str:
        if not path:
            return ''
        s = []
        words = path.split('/')
        for w in words:
            if w == '' or w == '.':
                continue
            elif w == '..':
                if s:
                    s.pop()
            else:
                s.append(w)
        return '/' + '/'.join(s)

    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(i, j):
            grid[i][j] = '0'
            offset = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for o in offset:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1':
                    dfs(x, y)

        if not grid or not grid[0]:
            return 0
        res = 0
        for i in range(len(grid)):
            for j in range((len(grid[0]))):
                if grid[i][j] == '1':
                    res += 1
                    dfs(i, j)
        return res

    def majorityElement(self, nums: List[int]) -> List[int]:
        res = []
        if not nums:
            return res
        i, x, j, y = None, 0, None, 0
        for z in nums:
            if z == i:
                x += 1
            elif z == j:
                y += 1
            elif x == 0:
                i, x = z, 1
            elif y == 0:
                j, y = z, 1
            else:
                x, y = x - 1, y - 1
        x, y = 0, 0
        for z in nums:
            if z == i:
                x += 1
            if z == j:
                y += 1
        if x > len(nums) // 3:
            res.append(i)
        if y > len(nums) // 3:
            res.append(j)
        return res

    def reverseWords(self, s: str) -> str:
        if not s:
            return s
        return ' '.join(s.split()[::-1])

    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return -1
        lmax = res = nums[0]
        for i in range(1, len(nums)):
            lmax = max(lmax + nums[i], nums[i])
            res = max(lmax, res)
        return res

    def compareVersion(self, version1: str, version2: str) -> int:
        x, y = version1.split('.'), version2.split('.')
        for i in range(max(len(x), len(y))):
            m, n = int(x[i]) if i < len(x) else 0, int(y[i]) if i < len(y) else 0
            if m != n:
                return 1 if m > n else -1
        return 0

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        if numCourses <= 0:
            return True
        # construct graph using adjacency list and maintain an indegree array
        indegrees = [0] * numCourses
        graph = [[] for _ in range(numCourses)]
        for i, j in prerequisites:
            indegrees[i] += 1
            graph[j].append(i)

        # BFS and everytime enqueue vertices with indegree = 0
        q = collections.deque()
        for i, v in enumerate(indegrees):
            if v == 0:
                q.append(i)
        cnt = len(q)
        while q:
            x = q.popleft()
            for c in graph[x]:
                indegrees[c] -= 1
                if indegrees[c] == 0:
                    q.append(c)
                    cnt += 1
        return cnt == numCourses

        cnt = 0
        while q:
            x = q.popleft()
            cnt += 1
            for c in graph[x]:
                indegrees[c] -= 1
                if indegrees[c] == 0:
                    q.append(c)
        return cnt == numCourses

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        res = []
        if numCourses <= 0:
            return res
        indegrees = [0] * numCourses
        graph = collections.defaultdict(list)
        for i, j in prerequisites:
            indegrees[i] += 1
            graph[j].append(i)
        q = collections.deque()
        for i, n in enumerate(indegrees):
            if n == 0:
                q.append(i)

        while q:
            x = q.popleft()
            res.append(x)
            for c in graph[x]:
                indegrees[c] -= 1
                if indegrees[c] == 0:
                    q.append(c)
        return res if len(res) == numCourses else []

    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0 or n % 3 != 0:
            return False
        while n % 3 == 0:
            n //= 3
        return n == 1

    def countAndSay(self, n: int) -> str:
        if n < 1:
            return ''
        s = ['1']
        for i in range(1, n):
            t = []
            c = s[0]
            cnt = 1
            for j in range(1, len(s) + 1):
                if j < len(s) and s[j] == c:
                    cnt += 1
                else:
                    t.extend([str(cnt), c])
                    c, cnt = s[j], 1
            s = t
        return ''.join(s)

    def countNodes(self, root: TreeNode) -> int:
        tr = root
        nl = nr = 0
        while tr:
            tr = tr.left
            nl += 1
        tr = root
        while tr:
            tr = tr.right
            nr += 1
        if nl == nr:
            return 2 ** nl - 1
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)

    def summaryRanges(self, nums: List[int]) -> List[str]:
        def range_helper(i: int, j: int) -> str:
            if i == j:
                return str(i)
            return str(i) + '->' + str(j)

        res = []
        if not nums:
            return res
        i = nums[0]
        for j in range(1, len(nums)):
            if nums[j] != nums[j - 1] + 1:
                res.append(range_helper(i, nums[j - 1]))
                i = nums[j]
        res.append(range_helper(i, nums[-1]))
        return res

    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        res = []
        if not s:
            return res
        i = 0
        hs = set()
        while i + 10 <= len(s):
            sub = s[i:i + 10]
            if sub not in hs:
                hs.add(sub)
            else:
                res.append(sub)
            i += 1
        return list(set(res))

    class Codec:

        def serialize(self, root):
            """Encodes a tree to a single string.

            :type root: TreeNode
            :rtype: str
            """
            if not root:
                return ''
            res = []
            self.serialize_helper(root, res)
            return ','.join(res)

        def serialize_helper(self, root, res):
            if not root:
                res.append('#')
                return
            res.append(str(root.val))
            self.serialize_helper(root.left, res)
            self.serialize_helper(root.right, res)

        def deserialize(self, data):
            """Decodes your encoded data to tree.

            :type data: str
            :rtype: TreeNode
            """
            if not data:
                return None
            deque = collections.deque(data.split(','))
            return self.deserialize_helper(deque)

        def deserialize_helper(self, deque):
            x = deque.popleft()
            if x == '#':
                return None
            root = TreeNode(int(x))
            root.left = self.deserialize_helper(deque)
            root.right = self.deserialize_helper(deque)
            return root

        def findPeakElement(self, nums: List[int]) -> int:
            if not nums:
                return -1
            i, j = 0, len(nums) - 1
            while i <= j:
                m = i + (j - i) // 2
                if (m == 0 or nums[m] > nums[m - 1]) and (m == len(nums) - 1 or nums[m] > nums[m + 1]):
                    return m
                elif m != len(nums) - 1 and nums[m] < nums[m + 1]:
                    i = m + 1
                elif m != 0 and nums[m] < nums[m - 1]:
                    j = m - 1
            return -1

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
            t = cur = None
            while st1 or st2 or carry:
                x = st1.pop() if st1 else 0
                y = st2.pop() if st2 else 0
                sum = x + y + carry
                cur = ListNode(sum % 10)
                cur.next = t
                t = cur
                carry = sum // 10
            return cur

        def titleToNumber(self, s: str) -> int:
            if not s:
                return 0
            res = 0
            for c in s:
                res = res * 26 + ord(c) - ord('A') + 1
            return res

        def isValidBST(self, root: TreeNode) -> bool:
            def dfs(root, lmin, lmax):
                if not root:
                    return True
                return lmin < root.val < lmax and dfs(root.left, lmin, root.val) and dfs(root.right, root.val, lmax)
            return dfs(root, float('-inf'), float('inf'))

        class MyStack:

            def __init__(self):
                """
                Initialize your data structure here.
                """
                self.q1, self.q2 = collections.deque(), collections.deque()

            def push(self, x: int) -> None:
                """
                Push element x onto stack.
                """
                self.q1.append(x)

            def pop(self) -> int:
                """
                Removes the element on top of the stack and returns that element.
                """
                while len(self.q1) > 1:
                    self.q2.append(self.q1.popleft())
                x = self.q1.popleft()
                self.q1, self.q2 = self.q2, self.q1
                return x

            def top(self) -> int:
                """
                Get the top element.
                """
                while len(self.q1) > 1:
                    self.q2.append(self.q1.popleft())
                return self.q1[0]

            def empty(self) -> bool:
                """
                Returns whether the stack is empty.
                """
                return len(self.q1) == 0

        def myAtoi(self, str: str) -> int:
            str = str.strip()
            if not str:
                return 0
            is_neg = None
            if str[0] in '+-':
                is_neg = True if str[0] == '-' else False
            res, start = 0, 0 if is_neg is None else 1
            for i in range(start, len(str)):
                s = str[i]
                if not s.isdigit():
                    break
                res = res * 10 + ord(s) - ord('0')
            res = res if not is_neg else -res
            if res > 2 ** 31 - 1:
                res = 2 ** 31 - 1
            elif res < -(2 ** 31) - 1:
                res = -(2 ** 31) - 1
            return res

        def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
            if not head:
                return head
            dummy = ln = ListNode(0)
            dummy.next = head
            cur = head
            for i in range(m - 1):
                ln = ln.next
                cur = cur.next
            pre = nn = None
            for i in range(m, n + 1):
                nn = cur.next
                cur.next = pre
                pre = cur
                cur = nn
            ln.next.next = nn
            ln.next = cur
            return dummy.next

        def oddEvenList(self, head: ListNode) -> ListNode:
            if not head:
                return head
            n = 1
            dummyo = preo = ListNode(0)
            dummye = pree = ListNode(0)
            while head:
                if n % 2 == 1:
                    preo.next = head
                    preo = preo.next
                else:
                    pree.next = head
                    pree = pree.next
                n += 1

            preo.next = dummye.next
            return dummyo.next

        def coinChange(self, coins: List[int], amount: int) -> int:
            if not coins or amount <= 0:
                return -1
            coins.sort()
            dp = [amount + 1] * (amount + 1)
            dp[0] = 0
            for i in range(1, amount + 1):
                for j in coins:
                    if i < j:
                        break
                    dp[i] = min(dp[i], dp[i - j] + 1)
            return -1 if dp[amount] > amount else dp[amount]

        def uniquePaths(self, m: int, n: int) -> int:
            if m < 1 or n < 1:
                return 0
            dp = [[0] * n] * m
            for j in range(n):
                dp[0][j] = 1
            for i in range(m):
                dp[i][0] = 1
            for i in range(1, m):
                for j in range(1, n):
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
            return dp[m-1][n-1]

        def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
            if not obstacleGrid or not obstacleGrid[0]:
                return 0
            for i in range(len(obstacleGrid)):
                for j in range(len(obstacleGrid[0])):
                    if obstacleGrid[i][j] == 1:
                        obstacleGrid[i][j] = 0
                    elif i == 0 and j == 0:
                        obstacleGrid[i][j] = 1
                    else:
                        obstacleGrid[i][j] = (obstacleGrid[i-1][j] if i > 0 else 0) + (obstacleGrid[i][j-1] if j > 0
                                                                                       else 0)
            return obstacleGrid[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[i][j] += grid[i][j-1]
                elif j == 0:
                    grid[i][j] += grid[i-1][j]
                else:
                    grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        return grid[-1][-1]

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle or not triangle[0]:
            return 0
        for i in range(len(triangle) - 2, 0, -1):
            for j in range(len(triangle[i])):
                triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
        return triangle[0][0]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        if not s or not wordDict:
            return False
        ws = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
        for i in range(len(s)):
            if dp[i]:
                for j in range(i + 1, len(s)):
                    if s[i:j] in ws:
                        dp[i + 1] = True
        return dp[-1]

    def maxProduct(self, nums: List[int]) -> int:
        if not nums:
            return 0
        lmax = lmin = res = nums[0]
        for i in range(1, len(nums)):
            tmax = lmax
            lmax = max(nums[i], lmax * nums[i], lmin * nums[i])
            lmin = min(nums[i], tmax * nums[i], lmin * nums[i])
            res = max(res, lmax)
        return res

    def maximumProduct(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return 0
        nums.sort()
        return max(nums[0] * nums[1] * nums[-1], nums[-3] * nums[-2] * nums[-1])

    def maxProfit1(self, prices: List[int]) -> int:
        if not prices:
            return 0
        lmin = prices[0]
        res = 0
        for i in range(1, len(prices)):
            res = max(res, prices[i] - lmin)
            lmin = min(lmin, prices[i])
        return res

    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        res = 0
        for i in range(1, len(prices)):
            res += (prices[i] - prices[i-1] if prices[i] > prices[i-1] else 0)
        return res

    def detectCapitalUse(self, word: str) -> bool:
        return word.isupper() or word.islower() or word.istitle()

    def change(self, amount: int, coins: List[int]) -> int:
        # 组合问题。 假设一个coin必须取为一个子问题
        if amount <= 0 or not coins:
            return 0
        dp = [0] * (amount + 1)
        dp[0] = 1
        for i in coins:
            for j in range(i, amount + 1):
                dp[j] += dp[j - i]
        return dp[-1]

    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1
        # 排列问题。每一个中间值为一个子问题
        nums.sort()
        for i in range(1, target + 1):
            j = 0
            while j < len(nums) and nums[j] <= i:
                dp[i] += dp[i - nums[j]]
                j += 1
        return dp[-1]

    def closestValue(self, root, target):
        if not root:
            return -1
        minv = float('inf')
        res = None
        while root:
            if root.val == target or abs(root.val - target) < minv:
                minv = abs(root.val - target)
                res = root.val
            if root.val < target:
                root = root.right
            else:
                root = root.left
        return res

    def minSteps(self, n: int) -> int:
        if n < 1:
            return n
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = i
            for j in range(1, i // 2 + 1):
                if i % j == 0:
                    dp[i] = min(dp[i], dp[j] + i // j)
        return dp[-1]

    def inorderSuccessor(self, root, p):
        if not root or not p:
            return None
        res = None
        while root:
            if root.val > p.val:
                res = root
                root = root.left
            else:
                root = root.right
        return res

    def isValidBST(self, root):
        def dfs(root, minv, maxv):
            if not root:
                return True
            return minv < root.val < maxv and dfs(root.left, minv, root.val) and dfs(root.right, root.val, maxv)
        return dfs(root, float('-inf'), float('inf'))

    ListNode.__eq__ = lambda self, other: self.val == other.val
    ListNode.__lt__ = lambda self, other: self.val < other.val
    # python3 needs to provide comparison function for object before can used in heap comparison

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None
        heap = []
        heapq.heapify(heap)
        for l in lists:
            heapq.heappush(heap, (l.val, l)) if l else None
        dummy = cur = ListNode(0)
        while heap:
            _, node = heapq.heappop(heap)
            cur.next = node
            cur = cur.next
            if node.next:
                heapq.heappush(heap, (node.next.val, node.next))
        return dummy.next

    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp = [1] * len(nums)
        res = 1  # minimum is 1
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
                    res = max(res, dp[i])
        return res

    # def findNumberOfLIS(self, nums: List[int]) -> int:
    # need more thinking
    #     if not nums:
    #         return 0
    #     dp = [1] * len(nums)
    #     maxv, cnt = 1, 1
    #     for i in range(1, len(nums)):
    #         for j in range(i):
    #             if nums[j] < nums[i]:
    #                 dp[i] = max(dp[i], dp[j] + 1)
    #                 if dp[i] > maxv:
    #                     maxv = dp[i]
    #                     cnt = 1
    #                 elif dp[i] == maxv:
    #                     cnt += 1
    #     return cnt

    def findLongestChain(self, pairs: List[List[int]]) -> int:
        # sort first to make problem convert to a LIS problem because array can be taken out of order
        if not pairs:
            return 0
        pairs.sort()
        dp = [1] * len(pairs)
        res = 1
        for i in range(1, len(pairs)):
            for j in range(i):
                if pairs[i][0] > pairs[j][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
            res = max(res, dp[i])
        return res

    def findSubsequences(self, nums: List[int]) -> List[List[int]]:

        def dfs(i, combi, res):
            if len(combi) >= 2:
                res.add(tuple(combi))
                if i == len(nums):
                    return
            for j in range(i, len(nums)):
                # if j > i and nums[j] == nums[j-1]:
                #     continue
                # this dedup does not work because if [1,2,1,1] still produce 2 (1,1). so have to use a set to dedup
                if not combi or nums[j] >= combi[-1]:
                    combi.append(nums[j])
                    dfs(j+1, combi, res)
                    combi.pop()

        res = set()
        if not nums:
            return []
        dfs(0, [], res)
        return [list(x) for x in res]

    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:

        def dfs(i, j, original_color):
            image[i][j] = newColor
            offset = [[-1, 0], [1, 0], [0, -1], [0, 1]]
            for o in offset:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(image) and 0 <= y < len(image[0]) and image[x][y] == original_color:
                    dfs(x, y, original_color)

        if not image or not image[0] or image[sr][sc] == newColor:
            return image
        dfs(sr, sc, image[sr][sc])
        return image

    def increasingTriplet(self, nums: List[int]) -> bool:
        if len(nums) < 3:
            return False
        # always update m1, m2 to smaller values, but as long as m2 exist, there is a increasing len=2 subsequence.
        # so anywhere exists x > m2 is a triplit
        m1 = m2 = float('inf')
        for x in nums:
            if x <= m1:
                m1 = x
            elif x <= m2:
                m2 = x
            else:
                return True
        return False

    def findLHS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        c = collections.Counter(nums)
        #max cannot take empty [], but can take a default value
        return max((c[k] + c[k+1] for k in c if c[k+1]), default=0)

    def productExceptSelf(self, nums):
        if not nums:
            return []
        res = [1] * len(nums)
        for i in range(1, len(nums)):
            res[i] = nums[i - 1] * res[i - 1]
        t = 1
        # end index is -1 because we need to go to 0
        for i in range(len(nums) - 2, -1, -1):
            t *= nums[i + 1]
            res[i] *= t
        return res

    def permute2(self, nums: List[int]) -> List[List[int]]:

        def dfs(marker: List[bool], combi: List[int], res: List[List[int]]):
            if len(combi) == len(nums):
                res.append(combi[:])
                return
            for i in range(len(nums)):
                if not marker[i]:
                    marker[i] = True
                    combi.append(nums[i])
                    dfs(marker, combi, res)
                    combi.pop()
                    marker[i] = False
        res = []
        if not nums:
            return res
        dfs([False] * len(nums), [], res)
        return res

    def mergeKLists2(self, lists):
        ListNode.__eq__ = lambda x, y: x.val == y.val
        ListNode.__lt__ = lambda x, y: x.val < y.val
        if not lists:
            return None
        heap = [x for x in lists if x]
        heapq.heapify(heap)
        dummy = cur = ListNode(0)

        while heap:
            x = heapq.heappop(heap)
            print(x.val)
            cur.next = x
            cur = cur.next
            if x.next:
                heapq.heappush(heap, x.next)
        return dummy.next

    def maxPathSum(self, root):
        def dfs(root, res):
            if not root:
                return 0
            lv = max(dfs(root.left, res), 0)
            rv = max(dfs(root.right, res), 0)
            res[0] = max(res[0], root.val + lv + rv)
            return root.val + max(lv, rv)
        res = [float('-inf')]
        dfs(root, res)
        return res[0]

    def reverse(self, head):
        pre = None
        cur = head
        while cur:
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
        return pre

    def deleteNode(self, node):
        node.val = node.next.val
        node.next = node.next.next

    def isSymmetric(self, root):
        def dfs(p, q):
            if not p:
                return not q
            if not q:
                return False
            return p.val == q.val and dfs(p.left, q.right) and dfs(p.right, q.left)

        if not root:
            return True
        return dfs(root.left, root.right)

    def zigzagLevelOrder(self, root):
        res = []
        if not root:
            return res
        cntc, cntn = 1, 0
        combi = []
        should_reverse = False
        q = collections.deque([root])
        while q:
            x = q.popleft()
            cntc -= 1
            combi.append(x.val)
            if x.left:
                q.append(x.left)
                cntn += 1
            if x.right:
                q.append(x.right)
                cntn += 1
            if cntc == 0:
                res.append(combi if not should_reverse else combi[::-1])
                should_reverse = not should_reverse
                cntc, cntn = cntn, 0
                combi = []
        return res

    def longestIncreasingSubsequence(self, nums):
        if not nums:
            return 0
        dp = [1] * len(nums)
        res = 1
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
            res = max(res, dp[i])
        return res

    def threeSum(self, numbers):
        res = []
        if len(numbers) < 3:
            return res
        numbers.sort()
        for i in range(len(numbers) - 2):
            if i > 0 and numbers[i] == numbers[i-1]:
                continue
            j, k = i + 1, len(numbers) - 1
            while j < k:
                sum = numbers[i] + numbers[j] + numbers[k]
                if sum < 0:
                    j += 1
                elif sum > 0:
                    k -= 1
                else:
                    res.append([numbers[i], numbers[j], numbers[k]])
                    j, k = j+ 1, k - 1
                    while j < k and numbers[j] == numbers[j-1]:
                        j += 1
                    while j < k and numbers[k] == numbers[k+1]:
                        k -= 1
        return res

    def exist2(self, board, word):
        def dfs(i, j, k):
            if k == len(word) - 1:
                return True
            t = board[i][j]
            board[i][j] = '#'
            offset = [[-1, 0], [1, 0], [0, 1], [0, -1]]
            for o in offset:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] == word[k + 1]:
                    if dfs(x, y, k + 1):
                        return True
            board[i][j] = t
            return False
        if not board or not board[0]:
            return False
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    if dfs(i, j, 0):
                        return True
        return False

    def addLists2(self, l1, l2):
        if not l1 or not l2:
            return l1 or l2
        s1, s2 = [], []
        while l1:
            s1.append(l1.val)
            l1 = l1.next
        while l2:
            s2.append(l2.val)
            l2 = l2.next
        carry = 0
        pre = None
        while s1 or s2 or carry:
            sum = carry
            if s1:
                sum += s1.pop()
            if s2:
                sum += s2.pop()
            node = ListNode(sum % 10)
            node.next = pre
            pre = node
            carry = sum // 10
        return pre

    def compareVersion2(self, version1, version2):
        va1 = version1.split('.')
        va2 = version2.split('.')
        for i in range(max(len(va1), len(va2))):
            x1 = int(va1[i]) if i < len(va1) else 0
            x2 = int(va2[i]) if i < len(va2) else 0
            if x1 != x2:
                return 1 if x1 > x2 else -1
        return 0

    def compress(self, originalString):
        if not originalString:
            return originalString
        cnt = 1
        c = originalString[0]
        res = []
        i = 1
        while i < len(originalString):
            if originalString[i] != originalString[i - 1]:
                res.extend([c, str(cnt)])
                c = originalString[i]
                cnt = 1
            else:
                cnt += 1
            i += 1
        res.extend([c, str(cnt)])
        new_str = ''.join(res)
        return new_str if len(new_str) < len(originalString) else originalString

    def permuteUnique2(self, nums):
        def dfs(marker, combi, res):
            if len(combi) == len(nums):
                res.append(combi[:])
                return
            for i in range(len(nums)):
                if marker[i] or (i > 0 and nums[i] == nums[i - 1] and not marker[i - 1]):
                    continue
                marker[i] = True
                combi.append(nums[i])
                dfs(marker, combi, res)
                combi.pop()
                marker[i] = False

        res = []
        if not nums:
            return res
        nums.sort()
        dfs([False] * len(nums), [], res)
        return res

    def numIslands2(self, grid):
        def dfs(i, j):
            grid[i][j] = 0
            offset = [[-1, 0], [1, 0], [0, -1], [0, 1]]
            for o in offset:
                x, y = i + o[0], j + o[1]
                if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 1:
                    dfs(x, y)

        if not grid or not grid[0]:
            return 0
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    res += 1
                    dfs(i, j)
        return res

    def getIntersectionNode(self, headA, headB):
        if not headA or not headB:
            return None
        m = n = 0
        ca, cb = headA, headB
        while ca:
            m += 1
            ca = ca.next
        while cb:
            n += 1
            cb = cb.next
        ca, cb = headA, headB
        while m - n > 0: # no =
            ca = ca.next
            m -= 1
        while n - m > 0: # no =
            cb = cb.next
            n -= 1
        while ca != cb:
            ca = ca.next
            cb = cb.next
        return ca

    def convertToTitle(self, n):
        if n < 1:
            return ''
        res = []
        while n:
            n -= 1 # use 26 as test case
            x = n % 26 + ord('A')
            res.append(chr(x))
            n //= 26
        return ''.join(reversed(res))

    def buildTreePreIn(self, preorder, inorder):
        def build(pl, pr, il, ir, im):
            if il > ir:
                return None
            root = TreeNode(preorder[pl])
            k = im[preorder[pl]]
            root.left = build(pl + 1, pl + k - il, il, k - 1, im)
            root.right = build(pl + k - il + 1, pr, k + 1, ir, im)
            return root

        if not preorder or len(preorder) != len(inorder):
            return None
        im = {}
        for i, x in enumerate(inorder):
            im[x] = i
        return build(0, len(preorder) - 1, 0, len(inorder) - 1, im)

    def buildTree(self, inorder, postorder):
        def build(il, ir, pl, pr, im):
            if il > ir:
                return None
            root = TreeNode(postorder[pr])
            k = im[postorder[pr]]
            root.left = build(il, k - 1, pl, pl + k - il - 1, im)
            root.right = build(k + 1, ir, pl + k - il, pr - 1, im)
            return root

        if not inorder or len(inorder) != len(postorder):
            return None
        im = {}
        for i, x in enumerate(inorder):
            im[x] = i
        return build(0, len(inorder) - 1, 0, len(postorder) - 1, im)

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # postorder. either meet one and all left or right, then the one is. or one left one right, then current root is
        if not root or root == p or root == q:
            return root
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)
        if l and r:
            return root
        return l or r

    def evalRPN(self, tokens: List[str]) -> int:
        if not tokens:
            return 0
        st = []
        for t in tokens:
            if t not in '+-*/': # '-11'.isdigit() return False. Need to be pure number
                st.append(int(t))
            else:
                y = st.pop()
                x = st.pop()
                r = 0
                if t == '+':
                    r = x + y
                elif t == '-':
                    r = x - y
                elif t == '*':
                    r = x * y
                else:
                    # -10 // 3 = -4. Python when negative divide, go to the smaller value. This is different than all
                    # other languages
                    r = x // y if x * y > 0 else -((-x) // y)
                st.append(r)
        return st.pop()

    def sortedArrayToBST(self, A):
        def dfs(i, j):
            if i > j:
                return None
            m = i + (j - i) // 2
            root = TreeNode(A[m])
            root.left = dfs(i, m - 1)
            root.right = dfs(m + 1, j)
            return root
        if not A:
            return None
        return dfs(0, len(A) - 1)

    def combine2(self, n, k):
        def dfs(i, combi, res):
            if len(combi) == k:
                res.append(combi[:])
                return
            for j in range(i, n + 1):
                combi.append(j)
                dfs(j + 1, combi, res)
                combi.pop()

        res = []
        if n <= 0 or k <= 0 or n < k:
            return res
        dfs(1, [], res)
        return res






































































h1 = ListNode(2)
h1.next = ListNode(4)

h2 = ListNode(-1)

s2 = Solution2()
s2.mergeKLists2([h1, h2])


# print('Final=', s2.restoreIpAddresses("25525511135"))


tt = "Bob hit a ball, the hit BALL flew far after it was hit."

# print(s2.mostCommonWord(tt, []))





s = Solution()

head = ListNode(1)
head.next = ListNode(2)

res = s.swapPairs(head)

# print(s.convertToTitle(1))
# print(s.convertToTitle(26))
# print(s.convertToTitle(25))
# print(s.convertToTitle(27))

l = [1,2,2,1]
n = [2,2]
# print(s.containsDuplicate(l))

# print(s.intersect(l, n))


t1 = (1,2,3)
t2 = (1,2,4,6)

ll1 = TestLL(t1).create_ll()
ll2 = TestLL(t2).create_ll()


# res = s.mergeTwoLists(ll1, ll2)

# print_ll(res)

l = [9,9,9,9]
# print(s.removeElement(l, 3))

# print(s.plusOne(l))

# print(s.addBinary('11', '1'))

# print(s.climbStairs(3))

l1 = [6, 0, 0, 0, 0]
l2 = [1, 2,3]
# s.merge(l1, 1, l2, 3)
# print(l1)

root = TreeNode(10)
root.left = TreeNode(5)
root.right = TreeNode(15)
root.left.left = TreeNode(3)
root.left.right = TreeNode(8)

# print(s.levelOrder(root))




