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
    
# 45 
class Solution:
    def jump(self, nums: List[int]) -> int:
        level_index_list = [{0}]

        while (len(nums) - 1) not in level_index_list[-1]:
            new_level = set()
            
            for i in level_index_list[-1]: 
                for j in range(1, nums[i] + 1): 
                    new_level.add(i+j)
            
            level_index_list.append(new_level)

        return len(level_index_list) - 1

class Solution:
    def jump(self, nums: List[int]) -> int:
        jumps = 0
        cur_end = 0
        farthest = 0

        for i in range(len(nums) - 1): 
            farthest = max(farthest, nums[i] + i)

            if i == cur_end: 
                jumps += 1
                cur_end = farthest

        return jumps
    
#150    
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations = sorted(citations, reverse=True)
        for i in range(len(citations)): 
            if citations[i] <= i: 
                return i
        return len(citations)
    
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)
        bucket = [0] * (n+1)

        for c in citations: 
            if c >= n: 
                bucket[n] += 1
            else:
                bucket[c] += 1

        total = 0
        for i in range(n , -1, -1): 
            total += bucket[i]
            if total >= i: 
                return i
            
        return 0
    

# 380
import random
class RandomizedSet:

    def __init__(self):
        self.vals = []
        self.pos = {}


    def insert(self, val: int) -> bool:
        if val in self.pos: 
            return False
        self.vals.append(val)
        self.pos[val] = len(self.vals) - 1
        return True
        

    def remove(self, val: int) -> bool:
        if val not in self.pos: 
            return False
        last_val = self.vals[-1]

        val_index = self.pos[val]
        
        # swap in list 
        self.vals[val_index] = last_val

        # swap in dictionary
        self.pos[last_val] = val_index

        # remove list 
        self.vals.pop()

        # remove dictionary  
        del self.pos[val]

        return True

    def getRandom(self) -> int:
        return random.choice(self.vals)


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()


# 238
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        left_list = [1]
        right_list = [1]
        
        for i in range(n - 1): 
            left_list.append(left_list[-1] * nums[i])
            right_list.append(right_list[-1] * nums[n - 1 - i])
        
        right_list.reverse()

        result = []
        for i in range(n): 
            result.append(left_list[i] * right_list[i])

        return result 
    
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [1] * n
        left_mul = 1
        right_mul = 1

        for i in range(1, n): 
            left_mul *= nums[i-1]
            result[i] *= left_mul

            right_mul *= nums[n-i]
            result[n-i-1] *= right_mul

        return result
    

# 134
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total_gas = 0
        tank = 0 
        start = 0 
        for i in range(len(gas)): 
            gain = gas[i] - cost[i]
            total_gas += gain
            tank += gain 

            if tank < 0: 
                start = i + 1
                tank = 0 

        return start if total_gas >= 0 else -1
    
# 12
class Solution:
    def intToRoman(self, num: int) -> str:
        table = {
            1000:"M", 
            900:"CM", 
            500:"D", 
            400:"CD", 
            100:"C", 
            90:"XC",
            50:"L", 
            40:"XL",
            10:"X", 
            9:"IX",
            5:"V", 
            4:"IV",
            1:"I"
            }
        

        roman = ""

        for d in table: 
            # add 
            count = num // d
            for i in range(count): 
                roman += table[d]

            num %= d

        return roman 
    
class Solution:
    def intToRoman(self, num: int) -> str:
        table = [
            (1000, "M"), 
            (900, "CM"), 
            (500, "D"), 
            (400, "CD"), 
            (100, "C"), 
            (90, "XC"),
            (50, "L"), 
            (40, "XL"),
            (10, "X"), 
            (9, "IX"),
            (5, "V"), 
            (4, "IV"),
            (1, "I")
        ]
        

        roman = ""

        for val, roman_val in table:
            # add 
            count = num // val
            
            roman += roman_val * count

            # update 
            num %= val

        return roman 
    
# 151
class Solution:
    def reverseWords(self, s: str) -> str:
        words = s.split()
        words.reverse()
        return ' '.join(words)

# 6 
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if len(s) <= numRows or numRows == 1: 
            return s
        
        characters = [c for c in s]
        characters.reverse()
        
        columns = []
        # start 
        while characters: 
            counter = 0
            # start the zigzag
            while counter < numRows-1 and characters: 
                column = [' '] * numRows
                if counter == 0: 
                    # populate the complete row
                    inner_counter = 0
                    while inner_counter < numRows and characters: 
                        column[inner_counter] = characters.pop()
                        inner_counter += 1
                else: 
                    column[numRows - 1 - counter] = characters.pop()

                # add column 
                columns.append(column)
                counter += 1
        
        result = ''
        for i in range(numRows): 
            for j in range(len(columns)): 
                if columns[j][i] != ' ': 
                    result += columns[j][i]

        return result

class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows >= len(s) or numRows == 1: 
            return s
        
        rows = [''] * numRows
        cur_row = 0
        going_down = False

        for c in s: 
            rows[cur_row] += c

            if cur_row == 0 or cur_row == numRows - 1: 
                going_down = not going_down

            cur_row += 1 if going_down else -1

        return ''.join(rows)
    
# 167
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        small_index = 0
        large_index = len(numbers) -1 
        satisfied = False

        while small_index < large_index and not satisfied: 
            if numbers[small_index] + numbers[large_index] == target: 
                satisfied = True
            elif numbers[small_index] + numbers[large_index] < target: 
                small_index += 1
            else: 
                large_index -= 1
            
        return [small_index+1, large_index+1]
    
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        small_index = 0
        large_index = len(numbers) -1 

        while small_index < large_index: 
            if numbers[small_index] + numbers[large_index] == target: 
                return [small_index+1, large_index+1]
            elif numbers[small_index] + numbers[large_index] < target: 
                small_index += 1
            else: 
                large_index -= 1
            

# 11
class Solution:
    def maxArea(self, height: List[int]) -> int:
        max_area = 0 
        left = 0
        right = len(height) - 1

        while left < right: 
            cur_area = min(height[left], height[right]) * (right - left)
            max_area = max(max_area, cur_area)

            if height[left] <= height[right]: 
                left += 1
            else: 
                right -= 1
        
        return max_area

# 15
# too slow 
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        results_set = []
        results = []
        for i in range(len(nums)-2): 
            for j in range(i+1, len(nums)-1): 
                for k in range(j+1, len(nums)): 
                    if nums[i] + nums[j] + nums[k] == 0 and {nums[i], nums[j], nums[k]} not in results_set: 
                        results_set.append({nums[i], nums[j], nums[k]})
                        results.append([nums[i], nums[j], nums[k]])
        return results
    
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        results = []

        for i in range(len(nums)-2): 
            if i == 0 or nums[i] != nums[i-1]:
                # 2 pointers
                j = i+1
                k = len(nums) - 1 
                while j < k: 
                    three_sum = nums[i] + nums[j] + nums[k]
                    if three_sum < 0: 
                        j += 1
                    elif three_sum > 0: 
                        k -= 1
                    else: 
                        if (j == i+1 or nums[j] != nums[j-1]) and (k == len(nums)-1 or nums[k] != nums[k+1]): 
                            results.append([nums[i], nums[j], nums[k]])
                            j += 1
                            k -= 1
                        else:   
                            if j != i+1 and nums[j] == nums[j-1]: 
                                j += 1
                            if k != len(nums)-1 and nums[k] == nums[k+1]: 
                                k -= 1
                        
        return results

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        results = []

        for i in range(len(nums)-2): 
            if i == 0 or nums[i] != nums[i-1]:
                # 2 pointers
                j = i+1
                k = len(nums) - 1 
                while j < k: 
                    three_sum = nums[i] + nums[j] + nums[k]
                    if three_sum < 0: 
                        j += 1
                    elif three_sum > 0: 
                        k -= 1
                    else: 
                        results.append([nums[i], nums[j], nums[k]])
                        # skip if next is the same
                        while j<k and nums[j] == nums[j+1]: 
                            j += 1
                        while j<k and nums[k] == nums[k-1]: 
                            k -= 1
                        # move pointers 
                        j += 1
                        k -= 1
                                                
        return results
    
# 209 
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        min_length = len(nums) + 1
        total = 0
        start = 0

        for end in range(len(nums)): 
            total += nums[end]

            while start < end and (total - nums[start]) >= target:
                total -= nums[start]
                start += 1

            if total >= target: 
                min_length = min(min_length, (end - start + 1))

        return 0 if min_length == len(nums) + 1 else min_length 
    
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        min_length = n + 1
        total = 0
        start = 0

        for end in range(n): 
            total += nums[end]

            while total >= target:
                min_length = min(min_length, (end - start + 1))
                total -= nums[start]
                start += 1                

        return 0 if min_length == n + 1 else min_length 
    
from bisect import bisect_right
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        prefix = [0] * (n+1)

        for i in range(1, n+1): 
            prefix[i] = prefix[i-1] + nums[i-1]
        
        min_len = n+1

        for i in range(1, n+1): 
            required = prefix[i] - target
            j = bisect_right(prefix, required) - 1
            if j >= 0: 
                min_len = min(min_len, i-j)

        return 0 if min_len == n+1 else min_len

# 3
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) < 2: 
            return len(s)
        
        cur_set = set()
        longest_len = 0
        start = 0
        for c in s: 
            if c not in cur_set: 
                cur_set.add(c)
            else: 
                while s[start] != c: 
                    cur_set.discard(s[start])
                    start += 1
                start += 1
            longest_len = max(longest_len, len(cur_set))

        return longest_len

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        cur_set = set()
        longest_len = 0
        start = 0

        for end in range(len(s)):
            while s[end] in cur_set:
                cur_set.discard(s[start])
                start += 1
            cur_set.add(s[end])
            longest_len = max(longest_len, end-start+1)

        return longest_len
            