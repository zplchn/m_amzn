import heapq
from typing import List, Tuple


class Solution:
    '''

    * Order Book


    Take in a stream of orders (as lists of limit price, quantity, and side, like ["155", "3", "buy"]) and return the total number of executed shares.

    Rules
    - An incoming buy is compatible with a standing sell if the buy's price is >= the sell's price. Similarly, an incoming sell is compatible with a standing buy if the sell's price <= the buy's price.
    - An incoming order will execute as many shares as possible against the best compatible order, if there is one (by "best" we mean most favorable to the incoming order).
    - Any remaining shares will continue executing against the best compatible order, if there is one. If there are shares of the incoming order remaining and no compatible standing orders, the incoming order becomes a standing order.

    Example input (more details are in the slide deck):
    Stream of orders
    ("150", "10", "buy"), ("165", "7", "sell"), ("168", "3", "buy"), ("155", "5", "sell"), ("166", "8", "buy")

    Example output:
    11

    '''

    def total_executed_shares(self, orders: List[Tuple[str, str, str]]) -> int:
        buy, sell = [], []
        res = 0

        for price, cnt, op in orders:
            price, cnt = int(price), int(cnt)
            if op == 'buy':
                heapq.heappush(buy, [-price, cnt])
            else:
                heapq.heappush(sell, [price, cnt])

            while buy and sell and -buy[0][0] >= sell[0][0]:
                if buy[0][1] >= sell[0][1]:
                    res += sell[0][1]
                    buy[0][1] -= sell[0][1]
                    sell[0][1] -= sell[0][1]
                else:
                    res += buy[0][1]
                    sell[0][1] -= buy[0][1]
                    buy[0][1] -= buy[0][1]

                if sell[0][1] == 0:
                    heapq.heappop(sell)
                if buy[0][1] == 0:
                    heapq.heappop(buy)
        return res

    '''
    You have N securities available to buy that each has a price Pi.
    Your friend predicts for each security the stock price will be Si at some future date.
    But based on volatility of each share, you only want to buy up to Ai shares of each security i.
    Given M dollars to spend, calculate the maximum value (revenue not profit) you could potentially
    achieve based on the predicted prices Si (and including any cash you have remaining).
    
    Assume fractional shares are possible. 
    
    * N = List of securities
    * Pi = Current Price
    * Si = Expected Future Price
    * Ai = Maximum units you are willing to purchase
    * M = Dollars available to invest
    
    Example Input:
    M = $140 available
    N = 4 Securities
    - P1=15, S1=45, A1=3 (AAPL)
    - P2=25, S2=35, A2=3 (SNAP
    - P3=40, S3=50, A3=3 (BYND
    - P4=30, S4=25, A4=4 (TSLA
    
    Output: $265
    3 shares of apple -> 45(15 *3), 135(45 *3)
    3 shares of snap -> 75, 105
    0.5 share of bynd -> 20, 25

    '''
    def max_value(self, options: List[List[int]], m: int) -> int:
        options.sort(key=lambda l: l[1] / l[0], reverse=True)
        res = 0
        i = 0
        for p, s, a in options:
            if s <= p:
                break
            shares = m / p
            if shares >= a:
                res += s * a
                m -= p * a
            else:
                res += shares * s
                break
        return res


class InsufficientFunds(Exception):
    pass


def bad_transfer(src_account, dst_account, amount):
    src_cash = src_account.cash # DB read
    dst_cash = dst_account.cash # DB read
    if src_cash < amount:
        raise InsufficientFunds
    src_account.cash = src_cash - amount # DB write
    src_account.send_src_transfer_email()

    dst_account.cash = dst_cash + amount # DB write
    dst_account.send_dst_transfer_email()


s = Solution()
input = [("150", "10", "buy"), ("165", "7", "sell"), ("168", "3", "buy"), ("155", "5", "sell"), ("166", "8", "buy")]

# res = s.total_executed_shares(input)
# input = [[15, 45, 3], [25, 35, 3], [40, 50, 3], [30, 25, 4]]

# input = [[10, 5, 3], [8, 5, 2]]
# res = s.max_value(input, 140)
# print(res)


def t() -> Tuple[int, int, int]:
    return 1



