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
            
#36 
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:

        # check whole line
        for i in range(9):

            # check horizontally 
            nums_1 = set()
            # check vertically
            nums_2 = set()

            for j in range(9): 

                # left to right
                if board[i][j] != '.': 
                    if board[i][j] not in nums_1:
                        nums_1.add(board[i][j]) 
                    else: 
                        return False
                    
                # top to down 
                if board[j][i] != '.': 
                    if board[j][i] not in nums_2:
                        nums_2.add(board[j][i]) 
                    else: 
                        return False
                
        # check 3 by 3
        for i in range(0, 9, 3):
            for j in range(0, 9, 3): 

                nums = set()

                # check cube
                for x in range(i, i+3): 
                    for y in range(j, j+3): 

                        if board[x][y] != '.':
                            if board[x][y] not in nums: 
                                nums.add(board[x][y])
                            else: 
                                return False
        
        return True


class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:

        # check whole line
        for i in range(9):

            # check horizontally 
            nums_1 = set()
            # check vertically
            nums_2 = set()

            for j in range(9): 

                # left to right
                if board[i][j] != '.': 
                    if board[i][j] not in nums_1:
                        nums_1.add(board[i][j]) 
                    else: 
                        return False
                    
                # top to down 
                if board[j][i] != '.': 
                    if board[j][i] not in nums_2:
                        nums_2.add(board[j][i]) 
                    else: 
                        return False

                # check 3 by 3  
                if i % 3 == 0 and j % 3 == 0: 
                    print("hit")
                    nums = set()

                    # check cube
                    for x in range(i, i+3): 
                        for y in range(j, j+3): 

                            if board[x][y] != '.':
                                if board[x][y] not in nums: 
                                    nums.add(board[x][y])
                                else: 
                                    return False
        
        return True
    
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]

        for i in range(9): 
            for j in range(9): 
                val = board[i][j]

                if val == '.': 
                    continue

                if val in rows[i] or val in cols[j] or val in boxes[(i//3) * 3 + (j//3)]: 
                    return False
                
                rows[i].add(val)
                cols[j].add(val)
                boxes[(i//3) * 3 + (j//3)].add(val)

        return True
    
# 334
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        seq = [float('inf')] * 2
        for num in nums: 
            if num < seq[0]: 
                seq[0] = num
            elif seq[0] < num < seq[1]: 
                seq[1] = num
            elif num > seq[1]: 
                return True
            
        return False

class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = second = float('inf')
        for num in nums: 
            if num <= first: 
                first = num
            elif num <= second: 
                second = num
            else: 
                return True
            
        return False

# 443
class Solution:
    def compress(self, chars: List[str]) -> int:
        pr = 0
        element = None
        counter = None

        while pr < len(chars): 
            element = chars[pr]
            counter = 1
            pr += 1

            while pr < len(chars) and chars[pr] == element: 
                chars.pop(pr)
                counter += 1

            if counter > 1: 
                counter_list = list(str(counter))
                for digit in counter_list: 
                    chars.insert(pr, digit)
                    pr += 1

        return len(chars)
    
class Solution:
    def compress(self, chars: List[str]) -> int:
        write = 0
        read = 0 

        while read < len(chars): 
            char = chars[read]
            count = 0
            while read < len(chars) and chars[read] == char: 
                read += 1
                count += 1
            
            chars[write] = char
            write += 1

            if count > 1: 
                for digit in str(count): 
                    chars[write] = digit
                    write += 1
        
        return write

# 1679
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        right = len(nums) - 1
        output = 0

        while left < right: 
            if nums[left] + nums[right] == k: 
                output += 1
                left += 1
                right -= 1
            elif nums[left] + nums[right] < k: 
                left += 1
            else: 
                right -= 1

        return output
    
# 1456
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        output = 0
        count = 0
        start = 0
        vowels = set('aeiou')
        for end in range(len(s)): 
            if s[end] in vowels: 
                count += 1
            
            if end >= k: 
                if s[start] in vowels: 
                    count -= 1
                start += 1
            
            output = max(output, count)

        return output

class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        count = sum(1 for c in s[:k] if c in vowels)
        output = count

        for i in range(k, len(s)): 
            if s[i] in vowels: 
                count += 1
            
            if s[i-k] in vowels: 
                count -= 1
            
            output = max(output, count)

        return output
    
# 1004 
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        credits = k 
        start = 0
        maximum = 0
        count = 0

        for i in range(len(nums)): 

            if nums[i] == 1: # normal just increase
                count += 1
                if count == 1: 
                    start = i

            else: # if current is 0
                
                if credits != 0: # use credits if still available
                    count += 1
                    credits -= 1

                else: # ran out of credits already
                    
                    if nums[start] == 0: # beginning was 0
                        start += 1
                        
                    else: # beginning was 1, restart
                        if k == 0:  
                            start = i
                            count = 0
                            credits = k
                        else: # move start until nums[start - 1] == 0 
                            while start < i and nums[start] != 0: 
                                count -= 1
                                start += 1
                            start += 1

            maximum = max(count, maximum)
        
        return maximum
    

class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        for right in range(len(nums)):
            if nums[right] == 0:
                k -= 1

            if k < 0:
                if nums[left] == 0:
                    k += 1
                left += 1

        return right - left + 1

# 1439 
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        if 0 not in nums: 
            return len(nums) - 1
        
        left = 0
        k = 1
        for right in range(len(nums)):
            if nums[right] == 0:
                k -= 1

            if k < 0:
                if nums[left] == 0:
                    k += 1
                left += 1

        return right - left

# 1657
from collections import Counter
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool: 
        map1 = Counter(word1)
        map2 = Counter(word2)
        
        return set(word1) == set(word2) and sorted(map1.values()) == sorted(map2.values())
    
from collections import Counter
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool: 
        map1 = Counter(word1)
        map2 = Counter(word2)
        
        return map1.keys() == map2.keys() and sorted(map1.values()) == sorted(map2.values())

# 2352    
class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        n = len(grid)
        map_rows = {}
        map_cols = {}

        for i in range(n): 
            row = tuple(grid[i])
            col = tuple(r[i] for r in grid)

            if row in map_rows: 
                map_rows[row] += 1
            else: 
                map_rows[row] = 1
            
            if col in map_cols: 
                map_cols[col] += 1
            else: 
                map_cols[col] = 1

        count = 0
        for key in map_rows: 
            if key in map_cols: 
                count += map_rows[key] * map_cols[key]
        return count
    
class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        n = len(grid)
        
        row_counts = Counter(tuple(row) for row in grid)
        col_counts = Counter(tuple(row[i] for row in grid) for i in range(n))

        return sum(row_counts[key] * col_counts[key] for key in row_counts if key in col_counts)
    
# 2390
class Solution:
    def removeStars(self, s: str) -> str:
        stack = []
        for c in s: 
            if c != '*': 
                stack.append(c)
            else: 
                stack.pop()
        return ''.join(stack)

# 735 
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for a in asteroids: 
            if a > 0: 
                stack.append(a)
            else: 
                while stack and 0 < stack[-1] < -a: 
                    stack.pop()

                if not stack or stack[-1] < 0: 
                    stack.append(a)
                elif stack[-1] == -a: 
                    stack.pop()

        return stack
                
# 394 
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        current_string = ''
        current_num = 0

        for c in s: 
            if c.isdigit(): 
                current_num = current_num * 10 + int(c)

            elif c == '[': 
                stack.append((current_string, current_num))
                current_string = ''
                current_num = 0

            elif c == ']': 
                prev_string, repeat = stack.pop()
                current_string = prev_string + current_string * repeat

            else: 
                current_string += c

        return current_string

# 649
from collections import deque
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        queue = deque(senate)
        while queue: 
            current_senate = queue.popleft()
            if current_senate == 'D': 
                if 'R' in queue: 
                    queue.remove('R')
                else:
                    return 'Dire'
            else: 
                if 'D' in queue:
                    queue.remove('D')
                else: 
                    return 'Radiant'
            queue.append(current_senate)

from collections import deque
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        n = len(senate)
        radiant = deque()
        dire = deque()
        for i, s in enumerate(senate): 
            if s == 'R': 
                radiant.append(i)
            else: 
                dire.append(i)
        
        while radiant and dire: 
            r_idx = radiant.popleft()
            d_idx = dire.popleft()

            if r_idx < d_idx: 
                radiant.append(r_idx + n)
            else: 
                dire.append(d_idx + n)

        return 'Radiant' if radiant else 'Dire'
    
# 2095 
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head.next: 
            return None
        
        n = 0
        org_head = head 
        while head: 
            n += 1
            head = head.next
        
        head = org_head
        prev = None
        pr = 0 
        while pr < n // 2: 

            if pr == (n//2 - 1): 
                prev = head

            head = head.next
            pr += 1

        prev.next = head.next 

        return org_head
    
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head.next: 
            return None
        
        slow = head
        fast = head
        prev = None

        while fast and fast.next: 
            fast = fast.next.next
            prev = slow 
            slow = slow.next 

        prev.next = slow.next
        return head  
    
# 328
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None or head.next == None or head.next.next == None: 
            return head 
        
        even = head 
        odd = head.next 
        odd_head = head.next
        while odd and odd.next: 
            even.next = even.next.next
            odd.next = odd.next.next
            even = even.next
            odd = odd.next 

        even.next = odd_head
        return head

class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head: 
            return head 
        
        even = head 
        odd = head.next 
        odd_head = head.next
        while odd and odd.next: 
            even.next = even.next.next
            odd.next = odd.next.next
            even = even.next
            odd = odd.next 

        even.next = odd_head
        return head
    
# 2130 
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        nums = []
        while head: 
            nums.append(head.val)
            head = head.next 

        maximum = 0
        n = len(nums)
        for i in range(n // 2): 
            maximum = max(maximum, nums[i] + nums[n - 1 - i])
        
        return maximum
    
class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        slow = head
        fast = head 
        while fast and fast.next: 
            slow = slow.next 
            fast = fast.next.next 
        
        # start reversing 
        prev = None
        curr = slow 
        next = slow.next 
        while next: 
            curr.next = prev 
            prev = curr 
            curr = next 
            next = next.next 
        curr.next = prev

        result = 0
        while curr and head: 
            result = max(result, curr.val + head.val)
            curr = curr.next 
            head = head.next 

        return result

class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        slow = head
        fast = head 
        while fast and fast.next: 
            slow = slow.next 
            fast = fast.next.next 
        
        # start reversing 
        prev = None
        curr = slow 
        while curr: 
            temp = curr.next 
            curr.next = prev 
            prev = curr 
            curr = temp 

        max_sum = 0
        first = head 
        second = prev
        while second:
            max_sum = max(max_sum, first.val + second.val)
            first = first.next 
            second = second.next 

        return max_sum

# 1448
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        count = 0
        max_val = root.val

        def dfs(node, local_max):
            if not node: 
                return 
            
            if node.val >= local_max: 
                nonlocal count 
                count += 1
                local_max = node.val
            
            dfs(node.left, local_max)
            dfs(node.right, local_max)
        
        dfs(root, max_val)

        return count
    
from collections import deque
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node, max_val): 
            if not node: 
                return 0
            
            good = 1 if node.val >= max_val else 0
            max_val = max(max_val, node.val)

            left = dfs(node.left, max_val)
            right = dfs(node.right, max_val)

            return good + left + right
        
        return dfs(root, root.val)

# 437 
from collections import defaultdict

class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        prefix_sum_count = defaultdict(int)
        prefix_sum_count[0] = 1 

        def dfs(node, curr_sum): 
            if not node: 
                return 0 
            
            curr_sum += node.val
            count = prefix_sum_count[curr_sum - targetSum]

            prefix_sum_count[curr_sum] += 1
            count += dfs(node.left, curr_sum)
            count += dfs(node.right, curr_sum) 
            prefix_sum_count[curr_sum] -= 1

            return count
        
        return dfs(root, 0)
    
# 1372 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        def dfs(node, dir, count): 
            if not node: 
                return 0
                       
            left = dfs(node.left, 'L', 1 if dir == 'L' else count + 1)
            right = dfs(node.right, 'R', count + 1 if dir == 'L' else 1)
            
            return max(count, left, right)
        
        return dfs(root, None, 0)
    
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        self.max_len = 0

        def dfs(node, dir, length): 
            if not node: 
                return 
            
            self.max_len = max(self.max_len, length)

            if dir == 'L': 
                dfs(node.left, 'L', 1)
                dfs(node.right, 'R', length + 1)
            else: 
                dfs(node.left, 'L', length + 1)
                dfs(node.right, 'R', 1)
            
            return 
        
        dfs(root, None, 0)

        return self.max_len
    
# 236 
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        self.result = None
        def dfs(node, p, q): 
            if not node: 
                return False
            
            curr = False
            if node.val == p.val or node.val == q.val: 
                curr = True
            
            left = dfs(node.left, p, q)
            right = dfs(node.right, p, q)

            if (left and right) or (curr and (left or right)): 
                self.result = node

            return left or right or curr
        
        dfs(root, p, q)
        return self.result
    
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q: 
            return root 
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right: 
            return root
        
        return left if left else right
    
# 199
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root: 
            return []
        
        curr_level = [root]
        next_level = []
        result = []
        while curr_level: 
            right_most = curr_level[0]
            result.append(right_most.val)
            for node in curr_level: 
                if node.right: 
                    next_level.append(node.right)
                if node.left: 
                    next_level.append(node.left)
            
            curr_level = next_level
            next_level = []

        return result
    
# 1161
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        sums = []

        current_level = [root]
        next_level = []

        while current_level: 
            level_sum = 0
            for node in current_level: 
                level_sum += node.val
                if node.left: 
                    next_level.append(node.left)
                if node.right: 
                    next_level.append(node.right)
                
            sums.append(level_sum)
            current_level = next_level
            next_level = []

        return sums.index(max(sums)) + 1

# 450 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root: 
            return None
        
        if key < root.val: 
            root.left = self.deleteNode(root.left, key)
        elif key > root.val: 
            root.right = self.deleteNode(root.right, key)
        else: 
            if not root.left: 
                return root.right
            elif not root.right: 
                return root.left
            
            successor = self.getMin(root.right)
            root.val = successor.val
            root.right = self.deleteNode(root.right, successor.val)

        return root
    
    def getMin(self, node): 
        while node.left: 
            node = node.left
        return node

# 841 
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        keys = set(rooms[0])
        unvisited = set(range(1, len(rooms)))

        # only keep the useful keys
        keys &= unvisited

        while keys: 
            # remove the rooms 
            unvisited -= keys

            new_keys = []

            for key in keys: 
                n_k = rooms[key]
                for k in n_k: 
                    new_keys.append(k)

            new_keys = set(new_keys)
            new_keys &= unvisited
            keys = new_keys

        return True if not unvisited else False

class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        unlocked = set([0])
        stack = [0]

        while stack: 
            room = stack.pop()
            for key in rooms[room]: 
                if key not in unlocked: 
                    unlocked.add(key)
                    stack.append(key)

        return len(unlocked) == len(rooms)

# 547 
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        if len(isConnected) == 1: 
            return 1

        visited = set()
        stack = []
        provinces = 0

        def dfs(visited, matrix, stack): 
            while stack: 
                city = stack.pop()
                for index, value in enumerate(matrix[city]): 
                    if (value == 1) and (index not in visited): 
                        visited.add(index)
                        stack.append(index)

        while len(visited) != len(isConnected): 
            # add the lowest index who is not in the visited set to the stack 
            for idx in range(len(isConnected)): 
                if idx not in visited: 
                    stack.append(idx)
                    break
            
            dfs(visited, isConnected, stack)
            provinces += 1

        return provinces

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        visited = set()
        provinces = 0

        def dfs(city): 
            stack = [city]
            while stack: 
                curr = stack.pop()
                for neighbor, is_connected in enumerate(isConnected[curr]): 
                    if (is_connected == 1) and (neighbor not in visited): 
                        visited.add(neighbor)
                        stack.append(neighbor)

        for i in range(len(isConnected)): 
            if i not in visited: 
                visited.add(i)
                dfs(i)
                provinces += 1
        
        return provinces

# 1466 
from collections import defaultdict
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        graph = defaultdict(list)

        for a, b in connections: 
            graph[a].append((b, 1))
            graph[b].append((a, 0))

        def dfs(node, parent): 
            for neighbor, needs_reversal in graph[node]: 
                if neighbor == parent: 
                    continue
                res[0] += needs_reversal
                dfs(neighbor, node)

        res = [0]
        dfs(0, -1)
        return res[0]
    
from collections import defaultdict
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        graph = defaultdict(list)

        for a, b in connections: 
            graph[a].append((b, 1))
            graph[b].append((a, 0))

        res = 0

        def dfs(node, parent): 
            nonlocal res
            for neighbor, needs_reversal in graph[node]: 
                if neighbor == parent: 
                    continue
                res += needs_reversal
                dfs(neighbor, node)

        dfs(0, -1)
        return res
    
from collections import defaultdict, deque
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        graph = defaultdict(list)

        for a, b in connections: 
            graph[a].append((b, 1))
            graph[b].append((a, 0))

        res = 0
        visited = set()
        queue = deque([0])

        while queue: 
            node = queue.popleft()
            visited.add(node)

            for neighbor, needs_reversal in graph[node]: 
                if neighbor in visited: 
                    continue
                res += needs_reversal
                queue.append(neighbor)

        return res
    
# 399
from collections import defaultdict

class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        graph = defaultdict(list)

        # build graph
        for (a, b), val in zip(equations, values): 
            graph[a].append((b, val))
            graph[b].append((a, 1/val))

        # dfs
        def dfs(start, end, visited): 
            if start == end: 
                return 1.0
            
            visited.add(start)

            for neighbor, value in graph[start]: 
                if neighbor in visited:
                    continue
                
                res = dfs(neighbor, end, visited)
                if res != -1.0: 
                    return res * value
            
            return -1.0

        # now construct results 
        results = []
        for a, b in queries: 
            if a not in graph or b not in graph: 
                results.append(-1.0)
            elif a == b: 
                results.append(1.0)
            else: 
                results.append(dfs(a, b, set()))
        return results 


# 1926 
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        from collections import deque

        m, n = len(maze), len(maze[0])
        queue = deque()
        queue.append((entrance[0], entrance[1], 0))
        maze[entrance[0]][entrance[1]] = '+'

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue: 
            r, c, steps = queue.popleft()

            for dr, dc in directions: 
                nr, nc = r + dr, c + dc

                if 0 <= nr < m and 0 <= nc < n and maze[nr][nc] == '.': 
                    if nr == 0 or nr == m-1 or nc == 0 or nc == n-1: 
                        return steps + 1
                    
                    maze[nr][nc] = '+'
                    queue.append((nr, nc, steps + 1))

        return -1
    


# 994 
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        from collections import deque
        
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        
        # find the rotten orange
        for r in range(m): 
            for c in range(n): 
                if grid[r][c] == 2: 
                    queue.append((r, c, 0))
                elif grid[r][c] == 1: 
                    fresh += 1
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        steps = 0

        while queue: 
            r, c, steps = queue.popleft()

            for dr, dc in directions: 
                nr, nc = r+dr, c+dc

                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1: 
                    grid[nr][nc] = 2
                    fresh -= 1
                    queue.append((nr, nc, steps + 1))

        return steps if fresh == 0 else -1
    
# 215
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        sorted_num = sorted(nums)
        return sorted_num[-k]
    
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap = nums[:k]
        heapq.heapify(min_heap)

        for num in nums[k:]: 
            if num > min_heap[0]: 
                heapq.heappushpop(min_heap, num)

        return min_heap[0]
    
# 2336 
import heapq
class SmallestInfiniteSet:

    def __init__(self):
        self.current = 1
        self.heap = []
        self.in_heap = set()

    def popSmallest(self) -> int:
        if self.heap:
            smallest = heapq.heappop(self.heap)
            self.in_heap.remove(smallest)
            return smallest
        else: 
            smallest = self.current
            self.current += 1
            return smallest

    def addBack(self, num: int) -> None:
        if num < self.current and num not in self.in_heap: 
            heapq.heappush(self.heap, num)
            self.in_heap.add(num)
        


# Your SmallestInfiniteSet object will be instantiated and called as such:
# obj = SmallestInfiniteSet()
# param_1 = obj.popSmallest()
# obj.addBack(num)

# 2542 
import heapq
class Solution:
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        # pair and sort by nums2 in descending order
        pairs = sorted(zip(nums2, nums1), reverse=True)

        min_heap = []
        total_sum = 0
        max_score = 0

        for n2, n1 in pairs:
            heapq.heappush(min_heap, n1)
            total_sum += n1 

            if len(min_heap) > k: 
                total_sum -= heapq.heappop(min_heap)

            if len(min_heap) == k: 
                max_score = max(max_score, total_sum * n2)

        return max_score
    
# 2462
from collections import deque
import heapq
class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        n = len(costs) 
        total = 0

        dq = deque(costs)

        left_heap = []
        right_heap = []

        for _ in range(candidates): 
            if dq: 
                heapq.heappush(left_heap, dq.popleft())
            if dq: 
                heapq.heappush(right_heap, dq.pop())

        for _ in range(k):
            if left_heap and right_heap: 
                if left_heap[0] <= right_heap[0]: 
                    total += heapq.heappop(left_heap)
                    if dq: 
                        heapq.heappush(left_heap, dq.popleft())

                else: 
                    total += heapq.heappop(right_heap)
                    if dq: 
                        heapq.heappush(right_heap, dq.pop())
            
            elif left_heap: 
                total += heapq.heappop(left_heap)
                if dq: 
                    heapq.heappush(left_heap, dq.popleft())

            else: 
                total += heapq.heappop(right_heap)
                if dq: 
                    heapq.heappush(right_heap, dq.pop())

        return total
    
from collections import deque
import heapq
class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        n = len(costs) 
        total = 0

        left = 0
        right = n-1

        left_heap = []
        right_heap = []

        for _ in range(candidates): 
            if left <= right: 
                heapq.heappush(left_heap, costs[left])
                left += 1
            if left <= right: 
                heapq.heappush(right_heap, costs[right])
                right -= 1

        for _ in range(k):
            if left_heap and right_heap: 
                if left_heap[0] <= right_heap[0]: 
                    total += heapq.heappop(left_heap)
                    if left <= right: 
                        heapq.heappush(left_heap, costs[left])
                        left += 1

                else: 
                    total += heapq.heappop(right_heap)
                    if left <= right: 
                        heapq.heappush(right_heap, costs[right])
                        right -= 1
            
            elif left_heap: 
                total += heapq.heappop(left_heap)
                if left <= right: 
                    heapq.heappush(left_heap, costs[left])
                    left += 1

            else: 
                total += heapq.heappop(right_heap)
                if left <= right: 
                    heapq.heappush(right_heap, costs[right])
                    right -= 1

        return total

# 2300
import math
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        potions.sort()
        m = len(potions)
        results = []

        for spell in spells: 
            min_potion = math.ceil(success/spell)

            left = 0
            right = m - 1
            result = -1

            while left <= right: 
                mid = (left + right)//2

                if potions[mid] >= min_potion: 
                    result = mid
                    right = mid -1 
                else: 
                    left = mid + 1

            if result != -1: 
                results.append(m - result)
            else: 
                results.append(0)

        return results

import bisect
import math
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        potions.sort()
        m = len(potions)
        results = []

        for spell in spells: 
            min_potion = math.ceil(success/spell)
            idx = bisect.bisect_left(potions, min_potion)
            results.append(m - idx)

        return results
    
#162 
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if len(nums) == 1: 
            return 0
        
        for i in range(len(nums)):
            if i == 0: 
                if nums[i] > nums[i + 1]: 
                    return i
            elif i == len(nums) - 1: 
                return i
            elif nums[i - 1] < nums[i] > nums[i + 1]:
                return i
            
        return -1

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1

        while left < right: 
            mid = (left + right) // 2

            if nums[mid] > nums[mid + 1]: 
                right = mid 
            else: 
                left = mid + 1

        return left 

# 875 
import math
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left = 1
        right = max(piles)
        result = right

        while left <= right: 
            mid = (left + right) // 2
            t = 0 
            for pile in piles: 
                t += math.ceil(pile / mid)

            if t <= h: 
                # should eat slower
                result = mid 
                right = mid - 1
            else: 
                # should eat faster
                left = mid + 1

        return result 
    
import math
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left = 1
        right = max(piles)
        result = right

        while left <= right: 
            mid = (left + right) // 2
            t = sum(math.ceil(pile / mid) for pile in piles)

            if t <= h: 
                # should eat slower
                result = mid 
                right = mid - 1
            else: 
                # should eat faster
                left = mid + 1

        return result 
    
# 17 
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits: 
            return []
        
        phone_map = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }

        res = []

        def backtrack(index, path): 
            if index == len(digits): 
                res.append(''.join(path))
                return 
            
            for letter in phone_map[digits[index]]: 
                path.append(letter)
                backtrack(index + 1, path)
                path.pop()

        backtrack(0, [])
        return res

# 198 
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums: 
            return 0
        
        n = len(nums) 
        if n == 1: 
            return nums[0]
        
        dp = [0] * n
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for i in range(2, n): 
            dp[i] = max(dp[i-1], dp[i-2] + nums[i])
        
        return dp[-1]
    
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums: 
            return 0
        
        n = len(nums) 
        if n == 1: 
            return nums[0]
        
        prev2 = nums[0]
        prev1 = max(nums[0], nums[1])

        for i in range(2, n): 
            curr = max(prev1, prev2 + nums[i])
            prev2, prev1 = prev1, curr
        
        return prev1
    
