from queue import Queue

# Set1:
# 1. Polish + Delete subtree
# 2. Two queues + Web slow
# 3. Wildcard matching


class Set1:
    '''

    Two queues get elements with distance <= 1


    '''
    def two_queues(self):
        # cur_ts = blocking_queue1.getNext()
        # q1.append(cur_ts)
        #
        # while q2 and cur_ts - q2[0] > 1:
        #     q2.popleft()  # deque, for performance
        #
        # for ts in q2:
        #     if abs(ts - cur_ts) <= 1:
        #         print(ts, cur_ts)
        #     else:
        #         break
        pass



    def isMatch(self, s: str, p: str) -> bool:
        # use dp[i][j] means i chars in s and j chars in p match
        # subproblem: 1. when p[j] == *, a - a*. * could mean 0 chars dp[i][j - 1]
        # 2. when p[j] == *, ab - a*.  * means some chars, dp[i - 1][j]
        # 3. when p[j] != *, compare s[i] == p[j] or p[j] == ?. dp[i - 1][j - 1]

        ls, lp = len(s), len(p)
        dp = [[False] * (lp + 1) for _ in range(len(s) + 1)]
        dp[0][0] = True
        for j in range(lp):
            dp[0][j + 1] = p[j] == '*' and dp[0][j]
        for i in range(ls):
            for j in range(lp):
                if p[j] == '*':
                    dp[i + 1][j + 1] = dp[i + 1][j] or dp[i][j + 1]
                else:
                    dp[i + 1][j + 1] = (s[i] == p[j] or p[j] == '?') and dp[i][j]
        return dp[-1][-1]

