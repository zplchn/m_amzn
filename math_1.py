from typing import List, Tuple
import collections

class Solution:
    def isPowerOfFour(self, num: int) -> bool:
        # bit 1 in odd position. 0xff is 1 byte / 8 bits
        # for power of 2 that is only 1 bit of 1 exist, use x & (x - 1) == 0
        return num > 0 and (num & (num - 1) == 0) and (num & 0x55555555 == num)

    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False
        while n % 3 == 0:
            n //= 3
        return n == 1

    def calculate224(self, s: str) -> int:
        '''
        different cases:
        100 - (2 + 3 - 4)
        1. if digit, accumulate the number as int and use a local number to record it
        2. if + or -. use a sign varible and treat + as 1 and - as -1
        3. use a stack, when (, push in stack with res and sign, reset local number to 0
        4. when ), calculate, then the entire () become a local num, and pop sign, and pop res. the next sign will
        trigger the calculation

        Every case, think 3 varibles to check, [num, res, and sign]
        :param s:
        :return:
        '''
        if not s:
            return 0
        num = res = 0
        sign = 1
        st = []
        # condition to trigger calculation -> meet a sign or meet a )
        s += '+'

        for c in s:
            if c.isdigit():
                num = num * 10 + ord(c) - ord('0')
            elif c in '+-':
                res += sign * num
                num = 0
                sign = 1 if c == '+' else -1
            elif c == '(':
                st.append(res)
                st.append(sign)
                res = 0
                sign = 1
            elif c == ')':
                res += sign * num
                num = res  # the entire () become a num and only need to calculate after it
                sign = st.pop()
                res = st.pop()
        return res

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

    def mySqrt(self, x: int) -> int:
        if x <= 1:
            return x
        l, r = 1, x // 2 # sqrt cannot be larger than N / 2
        while l <= r:
            m = l + ((r - l) >> 1)
            t = m * m
            if t == x:
                return m
            elif t < x:
                l = m + 1
            else:
                r = m - 1
        return r

    def myPow(self, x: float, n: int) -> float:
        # 50
        if n < 0:
            return 1 / self.myPow(x, -n)
        if n == 0:
            return 1
        t = self.myPow(x, n // 2)
        if n % 2 == 0:
            return t * t
        else:
            return t * t * x


    def prisonAfterNDays957(self, cells: List[int], N: int) -> List[int]:
        # as there are 6 digit 0 or 1, so total is 2 ^ 6 = 64 stats. so when N is large, it will repeat. and we need
        # to find the loop length, record in a hm
        res = []
        if not cells or N <= 0:
            return res
        hm = {}
        hm[tuple(cells)] = N
        while N:
            N -= 1
            c2 = [int(0 < i < 7 and cells[i - 1] == cells[i + 1]) for i in range(8)]
            t = tuple(c2)
            if t in hm:
                N %= hm[t] - N # after N % len,N will only small than the len and % will only be N.
            hm[t] = N
            cells = c2
        return cells

    def intToRoman(self, num: int) -> str:
        '''
        Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
        :param num:
        :return:
        '''
        rint = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        rstr = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        res = []
        i = 0
        while num > 0:
            if num >= rint[i]:
                num -= rint[i]
                res.append(rstr[i])
            else:
                i += 1
        return ''.join(res)

    def divide29(self, dividend: int, divisor: int) -> int:
        # x divide y means check how many N of y exist in x. and a number can expressed as a series of
        # N = a * 2 ** 31 + b * 2 ** 30 + ... + x * 2 + y. so we loop through 32 bit and check it's greater than divisor
        if dividend == - 2 ** 31 and divisor == -1:
            return 2 ** 31 - 1
        res = 0
        x, y = abs(dividend), abs(divisor)
        for i in reversed(range(32)):
            if x >= (y << i):
                res += (1 << i)
                x -= (y << i)
        return res if (dividend >= 0) == (divisor >= 0) else -res

    def numberToWords273(self, num: int) -> str:
        thousands = ['', 'Thousand', 'Million', 'Billion']
        till19 = ['', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve',
                  'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
        tens = ['', '', "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

        def read(n: int) -> str:
            if n < 20:
                return till19[n]
            if n < 100:
                return tens[n // 10] + ' ' + read(n % 10)
            else:
                return read(n // 100) + ' Hundred ' + read(n % 100)

        if num == 0:
            return 'Zero'
        res = ''
        for t in thousands:
            if num % 1000:  # 1000000. we dont want to output 1 million thousand
                res = read(num % 1000) + ' ' + t + ' ' + res  # for 328, will add two space after. need strip
            num //= 1000
        return ' '.join(res.split()) # split will ignore leading or trailing empty space

    def minAreaRect939(self, points: List[List[int]]) -> int:
        if not points:
            return 0
        hs = set()
        minv = float('inf')
        for x, y in points:
            hs.add((x, y))
        for x1, y1 in points:
            for x2, y2 in points:
                if x2 > x1 and y2 > y1 and (x2, y1) in hs and (x1, y2) in hs:
                    # if there is a rectangle, there must exist top right and bottom left
                    minv = min(minv, (x2 - x1) * (y2 - y1))
        return 0 if minv == float('inf') else minv

    def calculate772(self, s: str) -> int:
        def cal(i: int) -> Tuple[int, int]:
            st = []
            num = 0
            sign = '+'

            while i < len(s):
                if s[i].isdigit():
                    num = num * 10 + ord(s[i]) - ord('0')
                elif s[i] == '(':
                    num, i = cal(i + 1)
                elif s[i] in '+-*/)':
                    if sign == '+':
                        st.append(num)
                    elif sign == '-':
                        st.append(-num)
                    elif sign == '*':
                        st.append(st.pop() * num)
                    elif sign == '/':
                        st.append(int(st.pop() / num))

                    if s[i] == ')':
                        return sum(st), i # recursion but pass back the end ) index
                    sign = s[i]
                    num = 0
                i += 1
            return sum(st), i

        if not s:
            return 0
        s += '+'
        return cal(0)[0]

    def canMeasureWater365(self, x: int, y: int, z: int) -> bool:
        # Think the problem like an inf large jug, use two jug size = x, y
        # and + is pour in, - is pour out. at last z = ax + by. To have a result on this equation. z must be a
        # multiple of the gcd(x, y)
        def gcd(x, y):
            return x if y == 0 else gcd(y, x % y)
        return z == 0 or (z <= x + y and z % gcd(x, y) == 0)

    def countPrimes204(self, n: int) -> int:
        # if we know 2 is prime, then any multiple of 2, 2 * m <= n will not be prime, we can mark them off in a table
        # evertime find a number not marked, then inner loop to mark not prime. Outer loop starts from 2, stop sqrt(n).
        # inner loop start from i * 2.
        if n < 2: return 0
        marker = [True] * (n + 1)
        i = 2
        while i * i <= n: # 12 = 2 * 6, 3 * 4, no need to calculate 4 * 3, 6 * 2
            if marker[i]:
                for j in range(i * i, n + 1, i):
                    marker[j] = False
            i += 1
        res = 0
        for i in range(2, n):
            res += 1 if marker[i] else 0
        return res

    def isRectangleCover391(self, rectangles: List[List[int]]) -> bool:
        # Think of 2 sub rectangle if they can form one. 1. if they overlap, sum of area != final rec. We can check
        # global min and max so we know supposedly the final rec 4 cornors. 2. if they apart from each other, area will equal
        # but there will be more corners, how do we correctly count corner? we can use a set to record 4 corners for every
        # rectangle, we a point appear again, we cancel it and remove from the hs. so the inner corners should all cancel in
        # pairs and the final set should only contain the final 4 corners.
        hs = set()
        area = 0
        minx, miny, maxx, maxy = float('inf'), float('inf'), float('-inf'), float('-inf')
        for x1, y1, x2, y2 in rectangles:
            minx = min(minx, x1)
            miny = min(miny, y1)
            maxx, maxy = max(maxx, x2), max(maxy, y2)
            area += (x2 - x1) * (y2 - y1)

            for p in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
                if p in hs:
                    hs.remove(p)
                else:
                    hs.add(p)
        return hs == {(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)} and area == (maxx - minx) * (maxy - miny)

    def subtractProductAndSum1281(self, n: int) -> int:
        sumv, product = 0, 1
        while n:
            sumv += n % 10
            product *= n % 10
            n //= 10
        return product - sumv

    def isPalindrome9(self, x: int) -> bool:
        if x < 0:
            return False
        div = 1
        while x // div >= 10:
            div *= 10
        while x: # must compare till x, not >= 10. 10000021. we need 2 to fail
            if x // div != x % 10:
                return False
            x = x % div // 10
            div //= 100
        return True

    def maxPoints149(self, points: List[List[int]]) -> int:
        # Basic idea is to calculate slope between two every pairs, each match with one after it. Some special cases:
        # 1. same points appear again, need to count seperately 2. slope = 90 degree/inf
        def gcd(x, y):
            return x if y == 0 else gcd(y, x % y)

        if len(points) <= 2:
            return len(points)
        res = 0
        for i in range(len(points) - 1):
            hm = {}
            same = 0
            slope = None
            for j in range(i + 1, len(points)):
                dx, dy = points[i][0] - points[j][0], points[i][1] - points[j][1]
                if dx == dy == 0:
                    same += 1
                    continue
                if dx == 0:
                    slope = float('inf')
                else:
                    t = gcd(dx, dy)
                    slope = (dy // t, dx // t)
                hm[slope] = 2 if slope not in hm else hm[slope] + 1
            res = max(res, max(hm.values(), default=1) + same) # default=1 to handle all same points case!
        return res











def gcd(x, y):
    return x if y == 0 else gcd(y, x % y)

print(gcd(94911151,94911150))
print(gcd(94911152,94911151))




s = Solution()
res = []
# for x in [8, 28, 328, 5328, 25328]:
# for x in [123]:
#     res += [s.numberToWords273(x)]


print(s.isRectangleCover391([[1,1,3,3],[3,1,4,2],[3,2,4,4],[1,3,2,4],[2,3,3,4]]))

for y in res:
    print(y)











