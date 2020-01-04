import collections
from typing import List


class Solution:
    def find_children(self, nums: List[int], index: int) -> List[int]:
        # Time O(n). each node is visited once
        def find(i: int) -> int:
            if nums[i] not in [index, -1]:
                nums[i] = find(nums[i])
            return nums[i]
        res = []
        nums[index] = index # make this one root
        for i in range(len(nums)):
            if find(i) == index:
                res.append(i)
        return res

    '''
    hm = {index -> roots(either the subtree to be deleted or the root of the tree)}
    def find(node):
        if node in hm:
            return hm[node]
        if node.val != val and node.parent != -1:
            node.parent = find(node.parent)
        hm[val] = node.parent if node.val != val else node.val
        return hm[val]
    
    for i in nums:
        if find(i) == target:
            i.is_valid = false
            
    O(n) every node put in the hm(or modified parent) then, it's o(1） per operation. it's the find in union find
    
    '''

class Read4:
    def __init__(self):
        self.queue = collections.deque()
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Number of characters to read (int)
        :rtype: The number of actual characters read (int)
        """
        i = 0
        while i < n:
            buf4 = [''] * 4
            _ = read4(buf4)
            self.queue.extend(buf4)
            count = min(len(self.queue), n-i)
            if not count:
                break
            buf[i:] = [self.queue.popleft() for _ in range(count)]
            i += count
        return i


s = Solution()
input = [2,2,-1,0,1]
print(s.find_children(input, 0))