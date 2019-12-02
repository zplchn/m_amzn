from typing import List


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
        if n < 0:
            return 1 / self.myPow(x, -n)
        res = 1
        while n:
            if n % 2 == 1:
                res *= x
            x *= x
            n //= 2
        return res

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
        def cal(i: int):
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
            return sum(st)

        if not s:
            return 0
        s += '+'
        return cal(0)




s = Solution()
res = []
# for x in [8, 28, 328, 5328, 25328]:
for x in [123]:
    res += [s.numberToWords273(x)]

for y in res:
    print(y)











