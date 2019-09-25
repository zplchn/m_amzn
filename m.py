from typing import Optional, List
import collections
import heapq


class Solution:

    # 1. 只有大写和小写字母组成，要求返回一个大写字母，which is the largest alphabetically order and occurs both in lower and upper cases in
    # the string。 比如"aAbxeEX"，A，E，X都符合题意，但X的字母序比较大，所以返回X。
    def largestLetter(self, word: str) -> Optional[str]:
        small, large = [0] * 26, [0] * 26

        for c in word:
            if c.isupper():
                large[ord(c) - ord('A')] = 1
            else:
                small[ord(c) - ord('a')] = 1
        for x in range(25, 0, -1):
            if large[x] and small[x]:
                return chr(ord('A') + x)
        return None

    # 4. 给一个String S，求其中最长的没有连续3个相同字符的子字符串。比如uuuttxaaa要返回uuttxaa
    def remove_three_consecutive(self, str: str) -> str:
        if len(str) < 3:
            return str
        w = 0
        cnt = 1
        s = list(str)
        for r in range(1, len(s)):
            if s[r] == s[w]:
                if cnt < 2:
                    w += 1
                    s[w] = s[r]
                    cnt = 2
                else:
                    continue
            else:
                w += 1
                s[w] = s[r]
                cnt = 1
        return ''.join(s[:w+1])

    def longest_substring_no_3_same_letters(self, str: str) -> str:
        if len(str) < 3:
            return str
        res = ''
        maxv = 0
        cnt = 1
        pre = 0
        for i in range(1, len(str)):
            if str[i] != str[i - 1]:
                cnt = 1
                if i - pre + 1 > maxv:
                    maxv = i - pre + 1
                    res = str[pre: i + 1]
            elif cnt < 2:
                cnt = 2
                if i - pre + 1 > maxv:
                    maxv = i - pre + 1
                    res = str[pre: i + 1]
            else:
                pre = i - 1
        return res

    # 3.一个string求删掉一个字母后获得的lexicographical最小的string. 比如abcdz ->abcd
    # Lexicographically smallest string formed by removing at most one character
    def remove_one_char(self, s: str) -> str:
        if not s:
            return s
        i = 1
        while i < len(s) and s[i] >= s[i-1]:
            i += 1
        return s[:i-1] + s[i:]

    # largest alphabetically order and occurs both in lower and upper cases in the string
    def largest_both(self, s: str) -> str:
        if not s:
            return s
        res = ''
        hs = set()
        for c in s:
            if (c.islower() and c.upper() in hs) or (c.isupper() and c.lower() in hs):
                res = max(res, c.upper())
            hs.add(c)
        return res

    # replace ?
    def replace_question_mark(self, s: str) -> str:
        def replace_helper(i):
            for x in range(26):
                c = chr(x + ord('a'))
                if (i == 0 or sl[i - 1] != c) and (i == len(s) - 1 or sl[i + 1] != c):
                    return c

        if not s:
            return s
        sl = list(s)
        for j in range(len(sl)):
            if sl[j] == '?':
                sl[j] = replace_helper(j)
        return ''.join(sl)

    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        if not points or K < 1:
            return []
        return heapq.nsmallest(K, points, lambda l: l[0] ** 2 + l[1] ** 2)

    def subarraySum(self, nums: List[int], k: int) -> int:
        # use a hashmap to denote how many times a sum has been achived. so accumulate sum - k is a subproblem. sum
        # of these possibilities adds up to the total ways
        if not nums:
            return 0
        counter = collections.Counter()
        counter[0] = 1
        res = 0
        sum = 0
        for x in nums:
            sum += x
            res += counter[sum - k]
            counter[sum] += 1
        return res

    def remove_str_and_make_occurance_unique(self, s: str) -> int:
        if not s:
            return 0
        c = collections.Counter(s)
        heap = [-v for _, v in c.items()]
        heapq.heapify(heap)
        res = pre = 0
        while heap:
            x = heapq.heappop(heap)
            if x == pre:
                res += 1
                heapq.heappush(heap, x + 1) if x != -1 else None
            pre = x
        return res




TESTSTR = 'aAbxeEX'

s = Solution()
# print(s.largestLetter(TESTSTR))

# print(s.remove_three_consecutive('uuu'))
# print(s.remove_one_char('b'))

# print(s.largest_both('aAbxeEX'))
# print(s.replace_question_mark('????'))
for x in ['abc', 'aab','aabbc', 'aaabce']:
    # print(s.longest_substring_no_3_same_letters(x))
    print(s.remove_str_and_make_occurance_unique(x))


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
