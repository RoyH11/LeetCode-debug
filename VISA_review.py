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
        