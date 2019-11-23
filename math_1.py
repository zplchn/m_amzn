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
        4. when ), pop stack, use sign to multiple local number and then add to res
        dont forget the last number need to be added to res at the end

        Every case, think 3 varibles to check, [num, res, and sign]
        :param s:
        :return:
        '''
        if not s:
            return 0
        res, num, sign = 0, 0, 1
        st = []
        for x in s:
            if x.isdigit():
                num = num * 10 + ord(x) - ord('0')
            elif x in '+-':
                res += sign * num
                num = 0
                sign = 1 if x == '+' else -1
            elif x == '(':
                st.append(res)
                st.append(sign)
                sign = 1 # need to reset everything as beginning
                num = res = 0
            elif x == ')':
                res += sign * num
                sign = st.pop()
                res = sign * res + st.pop()
                num = 0
        # if the last is a num, it's need to add to res
        res += sign * num
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






