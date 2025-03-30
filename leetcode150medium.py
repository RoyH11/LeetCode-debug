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