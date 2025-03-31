from typing import List
#80 
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2: 
            return len(nums)
        
        memory = None
        count = 0
        for i in range(len(nums)): 
            if nums[i] != memory: 
                memory = nums[i]
                count = 1
            else: 
                count += 1
            
            if count > 2: 
                nums[i] = 10**4 + 1

        i = 0
        for num in nums: 
            if num < 10**4 + 1: 
                nums[i] = num
                i += 1

        return i
    
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) <= 2: 
            return len(nums)
        
        write = 2
        for read in range(2, len(nums)): 
            if nums[write - 2] != nums[read]: 
                nums[write] = nums[read]
                write += 1
                
        return write 
    
# 189
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(k): 
            nums[:] = nums[-1:] + nums[:-1]

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n
        nums[:] = nums[-k:] + nums[:-k]

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n

        nums.reverse()
        nums[:k] = reversed(nums[:k])
        nums[k:] = reversed(nums[k:])

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        total = 0 
        for i in range(len(prices) - 1): 
            if prices[i+1] > prices[i]: 
                total += prices[i+1] - prices[i]

        return total 

#55 
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) == 1: 
            return True
        
        # construct a power map
        power_dict = {}
        for i in range(len(nums)): 
            power_dict[i] = i + nums[i]

        # back tracking
        pr = len(nums) - 1
        for i in range(len(nums)-1, -1, -1): 
            if power_dict[i] >= pr: 
                pr = i
            
        return pr == 0
    
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        # back tracking
        pr = len(nums) - 1
        for i in range(len(nums)-1, -1, -1): 
            if nums[i] + i >= pr: 
                pr = i
            
        return pr == 0