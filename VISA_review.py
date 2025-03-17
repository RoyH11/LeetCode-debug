class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        roman_int_dict = {
            "I":1, 
            "V":5,
            "X":10, 
            "L":50, 
            "C":100, 
            "D":500, 
            "M":1000
        }

        s = list(s)
        result = 0
        max_value = 0
        for char in reversed(s):
            current_value = roman_int_dict[char]
            if current_value >= max_value: 
                result += current_value
                max_value = current_value
            else: 
                result -= current_value

        return result
    

class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = list(reversed(s))
        check = False
        count = 0
        for char in s: 
            # ending scinario 
            if char == " " and check == True: 
                return count
            # starting scinario
            elif char != " " and check == False: 
                check = True
            
            # in progress
            if check: 
                count += 1

        # if no ending
        return count
    
    
class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs: 
            return ""
        
        # find the shortest string
        min_str = min(strs, key=len)

        # now construct common_string
        common_string = ""

        # for each char
        for i, char in enumerate(min_str):
            for other in strs: 
                if other[i] != char: 
                    return min_str[:i]

        return min_str


class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if len(haystack)<len(needle): 
            return -1
        
        # create a dict 
        for index in range(len(haystack)-len(needle)+1): 
            current_case = haystack[index : (index+len(needle))]
            if current_case == needle: 
                return index
            
        return -1
    

class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # remove all non alphabet 
        forward = ""
        for char in s: 
            if char.isalnum():
                forward = forward + char

        forward = forward.lower()
        backward = forward[::-1]

        return forward == backward


class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        left = 0
        right = len(s)-1

        while left<right: 
            while left<right and not s[left].isalnum(): 
                left+=1
            while left<right and not s[right].isalnum(): 
                right-=1

            if s[left].lower() != s[right].lower(): 
                return False
            
            left+=1
            right-=1

        return True
    
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # Two-pointer approach (O(n) time, O(1) space)
        left, right = 0, len(s) - 1

        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1

            if s[left].lower() != s[right].lower():
                return False

            left += 1
            right -= 1

        return True



class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        p_s = 0
        for char in t: 
            if p_s < len(s) and s[p_s] == char: 
                p_s += 1

        return p_s == len(s)
    

class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        if len(ransomNote)>len(magazine): 
            return False
        
        # construct dict_m
        dict_m = {}
        for c in magazine: 
            if c in dict_m: 
                dict_m[c] += 1
            else: 
                dict_m[c] = 1

        # construct dict_r
        dict_r = {}
        for c in ransomNote: 
            if c in dict_r: 
                dict_r[c] += 1
            else: 
                dict_r[c] = 1

        #test each key in dict_r
        for key in dict_r.keys(): 
            if key not in dict_m: 
                return False
            elif dict_r[key]>dict_m[key]: 
                return False
            
        return True
    
from collections import Counter

class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        if len(ransomNote)>len(magazine): 
            return False
        
        count_r = Counter(ransomNote)
        count_m = Counter(magazine)

        for key, value in count_r.items(): 
            if value>count_m[key]: 
                return False
            
        return True
    

class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s)!=len(t): 
            return False
        
        match_dict = {}
        for index, s_char in enumerate(s): 
            t_char = t[index]
            if s_char not in match_dict: 
                match_dict[s_char] = t_char
            else: 
                if match_dict[s_char]!=t_char: 
                    return False
                
        # check uniqueness
        unique = set()
        for value in match_dict.values(): 
            if value in unique: 
                return False
            else: 
                unique.add(value)
        return True
    

