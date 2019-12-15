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


s = Solution()
input = [2,2,-1,0,1]
print(s.find_children(input, 0))