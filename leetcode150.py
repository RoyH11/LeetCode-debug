class Solution(object):
    def wordPattern(self, pattern, s):
        """
        :type pattern: str
        :type s: str
        :rtype: bool
        """
        words = s.split()

        if len(pattern)!=len(words): 
            return False

        match_dict = {}
        for index, c in enumerate(pattern):
            word = words[index]
            if c not in match_dict: 
                match_dict[c] = word
            elif match_dict[c] != word: 
                return False
            
        # check if values are unique
        unique_values = set()
        for value in match_dict.values(): 
            if value not in unique_values: 
                unique_values.add(value)
            else: 
                return False
        return True


from collections import Counter
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s)!=len(t): 
            return False
        
        char_dict = Counter(s)

        for char in t: 
            if char not in char_dict: 
                return False
            else: 
                char_dict[char] -= 1

        for value in char_dict.values(): 
            if value != 0: 
                return False
            
        return True


class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        num_index_dict = {}

        for index, num in enumerate(nums):
            dif = target-num
            if dif in num_index_dict: 
                diff_index = num_index_dict[dif]
                return [diff_index, index]
            else: 
                num_index_dict[num] = index


class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        visited = set()
        while n!=1 and n not in visited:
            visited.add(n)

            n_str = str(n)
            n = 0
            for c in n_str: 
                c_int = int(c)
                n += c_int**2
            

        return n==1
    
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        for i in range(len(nums)-1): 
            min_val = min(len(nums), i+1+k)
            for j in range(i+1, min_val): 
                if nums[i]==nums[j]:
                    return True
        return False
    

class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        # convert nums to a dictionary
        nums_dict = {}
        for index, num in enumerate(nums): 
            if (num in nums_dict) and abs(index-nums_dict[num])<=k:
                return True
            else: 
                nums_dict[num] = index
                
        return False
    



class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        result = []
        i = 0
        while i < len(nums): 
            start = nums[i]
            while i+1 < len(nums) and nums[i+1] == nums[i]+1: 
                i += 1
            
            if start == nums[i]: 
                result.append(str(start))
            else: 
                result.append(str(start) + "->" + str(nums[i]))
            i += 1       

        return result
    

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """

        stack = []
        bracket_map = {')':'(', ']':'[', '}':'{'}

        for char in s: 
            if char in bracket_map: 
                if stack and stack[-1] == bracket_map[char]: 
                    stack.pop()
                else: 
                    return False
            else: 
                stack.append(char)

        return not stack
            
        