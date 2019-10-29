from typing import List, Optional
import collections

class Solution:
    def anagram(self, s, t):
        if len(s) != len(t):
            return False
        hm = {}
        # hm size is bound to 26 + 26 and is still considered as constant space complexity
        # 0 ^ 1 ^ 2^ 3 == 0
        for x in s:
            hm[x] = hm.get(x, 0) + 1
        for y in t:
            hm[y] = hm.get(y, 0) - 1
            if hm[y] < 0:
                return False
        return True

    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        if not paragraph:
            return paragraph
        p = paragraph
        for c in "!?',;.":
            p = p.replace(c, ' ')
        c = collections.Counter([w for w in p.lower().split()])
        res, maxv = '', 0
        bs = set(banned)
        for k, v in c:
            if v > maxv and k not in bs:
                maxv = v
                res = k
        return res

    def isValid(self, s: str) -> bool:
        if not s:
            return True
        st = []
        for c in s:
            if c in '([{':
                st.append(c)
            else:
                if not st:
                    return False
                x = st.pop()
                if c == ')' and x != '(' or c == '}' and x != '{' or c == ']' and x != '[':
                    return False
        return len(st) == 0

    def compareVersion(self, version1: str, version2: str) -> int:
        va1, va2 = version1.split('.'), version2.split('.')
        for i in range(max(len(va1), len(va2))):
            x1 = int(va1[i]) if i < len(va1) else 0
            x2 = int(va2[i]) if i < len(va2) else 0
            if x1 != x2:
                return 1 if x1 > x2 else -1
        return 0

    def myAtoi(self, str: str) -> int:
        s = str.strip()
        if not s:
            return 0
        is_neg = False
        start = 0
        if s[0] in '+-':
            is_neg = s[0] == '-'
            start = 1
        res = 0
        for i in range(start, len(s)):
            if s[i].isdigit():
                res = res * 10 + ord(s[i]) - ord('0')
            else:
                break
        res = -res if is_neg else res
        if res > 2 ** 31 - 1:
            res = 2 ** 31 - 1
        if res < -(2 ** 31):
            res = -(2 ** 31)
        return res

    def isPalindrome(self, s: str) -> bool:
        if not s:
            return True
        i, j = 0, len(s) - 1
        while i < j:
            if not s[i].isalnum():
                i += 1
            elif not s[j].isalnum():
                j -= 1
            elif s[i].lower() != s[j].lower():
                return False
            else:
                i, j = i + 1, j - 1
        return True

    def firstUniqChar(self, s: str) -> int:
        if not s:
            return -1
        c = collections.Counter(s)
        for i in range(len(s)):
            if c[s[i]] == 1:
                return i
        return -1

    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ''
        prefix = strs[0]
        if len(strs) == 1:
            return prefix
        r = 0
        while r < len(prefix):
            if not all(r < len(strs[i]) and prefix[r] == strs[i][r] for i in range(1, len(strs))):
                break
            r += 1
        return prefix[:r]

    def simplifyPath(self, path: str) -> str:
        if not path:
            return path
        st = []
        for p in path.split('/'):
            if not p or p == '.':
                continue
            elif p != '..':
                st.append(p)
            elif st:
                st.pop()
        return '/' + '/'.join(st)

    def largestNumber(self, nums: List[int]) -> str:
        class CustomLargest(str):
            def __lt__(self, other):
                return self + other > other + self
        if not nums:
            return ''

        iter_str = map(str, nums) # map(func, iterable) return a iterator with func(x)
        return ''.join(sorted(iter_str, key=CustomLargest)).lstrip('0') or '0'

    def validPalindrome(self, s: str) -> bool:
        def is_palindrome(l, r):
            while l < r:
                if s[l] != s[r]:
                    return False
                l, r = l + 1, r - 1
            return True

        if not s:
            return True
        l, r = 0, len(s) - 1
        while l < r:
            if s[l] != s[r]:
                return is_palindrome(l + 1, r) or is_palindrome(l, r - 1)
            l, r = l + 1, r - 1
        return True

    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        hm = {}
        for i, c in enumerate(s):
            if c in hm:
                if hm[c] != t[i]:
                    return False
            elif t[i] in hm.values():
                return False
            else:
                hm[c] = t[i]
        return True

    def convertToTitle(self, n: int) -> str:
        if n < 1:
            return ''
        res = []
        while n:
            n -= 1 # must minus 1 first otherwise 26 will be AZ not Z
            res.append(chr(n % 26 + ord('A')))
            n //= 26
        return ''.join(reversed(res))

    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        start = maxv = 0
        hm = {}

        for i in range(len(s)):
            if s[i] not in hm or hm[s[i]] < start:
                maxv = max(maxv, i - start + 1)
            else:
                start = hm[s[i]] + 1
            hm[s[i]] = i
        return maxv

    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        if not s:
            return 0
        hm = {}
        left = maxv = 0

        for i in range(len(s)):
            hm[s[i]] = hm.get(s[i], 0) + 1
            while len(hm) > 2:
                hm[s[left]] -= 1
                if hm[s[left]] == 0:
                    del hm[s[left]]
                left += 1  # this needs to happen after the deletion
            maxv = max(maxv, i - left + 1)
        return maxv

    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        if not s or k < 1:
            return 0
        hm = {}
        left = maxv = 0
        for i in range(len(s)):
            hm[s[i]] = hm.get(s[i], 0) + 1
            while len(hm) > k:
                hm[s[left]] -= 1
                if hm[s[left]] == 0:
                    del hm[s[left]]
                left += 1
            maxv = max(maxv, i - left + 1)
        return maxv

    def lengthOfLastWord(self, s: str) -> int:
        # if not s:                not ' ' is False. An string with space is Truthy
        #     return 0
        sa = s.split()
        return len(sa[-1] if sa else 0)

    def frequencySort(self, s: str) -> str:
        if not s:
            return s
        c = collections.Counter(s)
        return ''.join(k * v for k, v in c.most_common())

    def partitionLabels(self, S: str) -> List[int]:
        # hashmap for a letter -> its last index of appearance. Start loop and all letters between current and last
        # will have to be in one partition and keep the last updated. once index reach a local max, it can form one partition.
        res = []
        if not S:
            return res
        hm = {}
        for i, x in enumerate(S):
            hm[x] = i
        start, lmax = 0, 0
        for i in range(len(S)):
            lmax = max(hm[S[i]], lmax)
            if i == lmax:
                res.append(i - start + 1)
                start = i + 1
        return res

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = []
        if not strs:
            return res
        hm = collections.defaultdict(list)
        for s in strs:
            t = tuple(sorted(list(s)))
            hm[t].append(s)
        return list(hm.values()) #hashmap.values() return a ValuesView iterable

    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        res = []
        if not strings:
            return res
        hm = collections.defaultdict(list)
        for s in strings:
            k = []
            for i in range(1, len(s)):
                k.append((ord(s[i]) - ord(s[0])) % 26) # 'za' and 'ab' are considered an anagram.  -1 % 26 = 25.
            hm[tuple(k)].append(s)
        return list(hm.values())

    def areSentencesSimilar(self, words1: List[str], words2: List[str], pairs: List[List[str]]) -> bool:
        if len(words1) != len(words2):
            return False

        hm = collections.defaultdict(set)
        for x, y in pairs:
            hm[x].add(y)
            hm[y].add(x)
        return all(words2[i] == words1[i] or words2[i] in hm[words1[i]] for i in range(len(words1)))

    def lastSubstring(self, s: str) -> str:
        if not s:
            return s
        i, j, k = 0, 1, 0
        while j + k < len(s):
            if s[j + k] > s[i + k]:
                i, j = j, j + 1 #cacacb need to output cb not cacb. j need to be right next to the new i
                k = 0
            elif s[j + k] < s[i + k]:
                j = j + k + 1
                k = 0
            else:
                k += 1
        return s[i:]

    def reverseWords3(self, s: str) -> str:
        return ' '.join([x[::-1] for x in s.split()])

    def reverseWords(self, s: List[str]) -> None:
        def reverse(i, j):
            while i < j:
                s[i], s[j] = s[j], s[i]
                i, j = i + 1, j - 1
        if not s:
            return
        reverse(0, len(s) - 1)
        i = 0
        for j in range(len(s) + 1):
            if j == len(s) or s[j] == ' ':
                reverse(i, j - 1)
                i = j + 1

    def addStrings(self, num1: str, num2: str) -> str:
        if not num1 or not num2:
            return num1 or num2
        carry = 0
        i, j = len(num1) - 1, len(num2) - 1
        res = []
        while i >= 0 or j >= 0 or carry:
            sum = carry
            if i >= 0:
                sum += ord(num1[i]) - ord('0')
                i -= 1
            if j >= 0:
                sum += ord(num2[j]) - ord('0')
                j -= 1
            res.append(str(sum % 10)) # use str() not chr(), because it's for ascii value to char
            carry = sum // 10
        return ''.join(reversed(res))

    def reverseOnlyLetters(self, S: str) -> str:
        if not S:
            return S
        i, j = 0, len(S) - 1
        sa = list(S)
        while i < j:
            while i < j and not sa[i].isalpha():
                i += 1
            while i < j and not sa[j].isalpha():
                j -= 1
            sa[i], sa[j] = sa[j], sa[i]
            i, j = i + 1, j - 1
        return ''.join(sa)































































