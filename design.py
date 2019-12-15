import collections
import heapq
import itertools
from typing import List


class Twitter:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.followees = collections.defaultdict(set)
        self.tweets = collections.defaultdict(collections.deque) # use a deque, enqueue use appendLeft. deque by
        # default is popleft
        self.timer = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        """
        Compose a new tweet.
        """
        self.tweets[userId].appendleft((self.timer, tweetId))
        self.timer -= 1 # order by time max

    def getNewsFeed(self, userId: int) -> List[int]:
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        """
        it = heapq.merge(*(self.tweets[i] for i in (self.followees[userId] | {userId})))
        return [t for _, t in itertools.islice(it, 10)]

    def follow(self, followerId: int, followeeId: int) -> None:
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        """
        self.followees[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        """
        self.followees[followerId].discard(followeeId)