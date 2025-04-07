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
            
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        visited = set()
        while head: 
            if head not in visited: 
                visited.add(head)
                head = head.next
            else: 
                return True
            
        return False
    

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow, fast = head, head
        while fast and fast.next: 
            slow = slow.next
            fast = fast.next.next
            if slow == fast: 
                return True
            
        return False
    

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy = ListNode()
        tail = dummy 
        while list1 and list2: 
            if list1.val < list2.val: 
                tail.next = list1
                list1 = list1.next
            else: 
                tail.next = list2
                list2 = list2.next

            tail = tail.next

        tail.next = list1 if list1 else list2

        return dummy.next 
    



# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if root is None: 
            return 0
        if root.left is None and root.right is None: 
            return 1
        
        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        return max(left_depth, right_depth) +1
    
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if root is None: 
            return 0

        return max(self.maxDepth(root.left), self.maxDepth(root.right)) +1
    


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: Optional[TreeNode]
        :type q: Optional[TreeNode]
        :rtype: bool
        """
        if p is None and q is None: 
            return True
        elif p is None or q is None: 
            return False
        else: 
            if p.val != q.val: 
                return False
            else: 
                return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
                # swap = self.isSameTree(p.right, q.left) and self.isSameTree(p.left, q.right)
                return normal


class Solution(object):
    def invertTree(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: Optional[TreeNode]
        """
        if root is None: 
            return root
        elif root.left is None and root.right is None: 
            return root
        else: 
            self.invertTree(root.left)
            self.invertTree(root.right)

            temp = root.left
            root.left = root.right
            root.right = temp
            return root 


# BFS            
from collections import deque
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        if not root: 
            return False

        queue = deque([(root.left, root.right)])

        while queue: 
            left, right = queue.popleft()

            if not left and not right: 
                continue
            if not left or not right or left.val != right.val: 
                return False
            
            queue.append((left.right, right.left))
            queue.append((left.left, right.right))

        return True


# DFS
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: bool
        """
        if not root: 
            return False
        
        stack = [(root.left, root.right)]

        while stack: 
            left, right = stack.pop()

            if not left and not right: 
                continue
            if not left or not right or left.val != right.val: 
                return False
            
            stack.append((left.left, right.right))
            stack.append((left.right, right.left))

        return True


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: Optional[TreeNode]
        :type targetSum: int
        :rtype: bool
        """
        if not root: 
            return False
                    
        if root.val == targetSum and not root.left and not root.right: 
            return True
            
        targetSum -= root.val 
        return self.hasPathSum(root.left, targetSum) or self.hasPathSum(root.right, targetSum)
    
from collections import deque
class Solution(object):
    def countNodes(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        if not root: 
            return 0
        
        counter = 0
        queue = deque()
        queue.append(root)
        while queue: 
            node = queue.popleft()
            counter += 1
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)

        return counter
    
class Solution(object):
    def countNodes(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: int
        """
        def get_left_height(node):
            h = 0
            while node: 
                h+=1
                node = node.left
            return h
        
        def get_right_height(node): 
            h = 0
            while node: 
                h+=1
                node = node.right
            return h 
        
        if not root: 
            return 0
        
        left_height = get_left_height(root)
        right_height = get_right_height(root)

        if left_height == right_height: 
            return (1<<left_height) -1
        else: 
            return 1+self.countNodes(root.left)+self.countNodes(root.right)
        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: Optional[TreeNode]
        :rtype: List[float]
        """
        avg_list = []
        current_l = [root]
        while current_l: 
            next_l = []
            level_count = len(current_l)
            total = 0

            for node in current_l: 
                total += node.val
                if node.left: next_l.append(node.left)
                if node.right: next_l.append(node.right)

            avg_list.append(total/level_count)
            current_l = next_l

        return avg_list


from typing import List, Optional
from collections import deque

class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        result = []
        queue = deque([root])

        while queue: 
            level_total = 0 
            level_count = len(queue)

            for _ in range(level_count): 
                node = queue.popleft()
                level_total += node.val

                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)

            result.append(level_total/level_count)

        return result
    

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        ordered_list = []

        def dfs(node): 
            if not node: 
                return 
            dfs(node.left)
            ordered_list.append(node.val)
            dfs(node.right)

        dfs(root)

        min_val = float('inf')
        for i in range(len(ordered_list)-1): 
            current = ordered_list[i]
            next = ordered_list[i+1]
            min_val = min(min_val, next-current)
        
        return int(min_val)



class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        prev = None
        min_diff = float('inf')

        def inorder(node): 
            nonlocal prev, min_diff
            if not node: 
                return 
            
            inorder(node.left)
            if prev is not None: 
                min_diff = min(min_diff, node.val - prev)
            prev = node.val
            inorder(node.right)

        inorder(root)
        
        return int(min_diff)


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums: 
            return None
        
        mid_index = len(nums)//2
        node = TreeNode(nums[mid_index])

        node.left = self.sortedArrayToBST(nums[:mid_index])
        node.right = self.sortedArrayToBST(nums[mid_index+1:])

        return node

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def build(left, right): 
            if left>right: 
                return None
            
            mid = (left + right)//2
            node = TreeNode(nums[mid])
            node.left = build(left, mid-1)
            node.right = build(mid+1, right)

            return node

        return build(0, len(nums)-1)


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left<=right: 
            mid = (left + right) // 2
            if nums[mid] == target: 
                return mid
            elif nums[mid] < target: 
                left = mid + 1
            else: 
                right = mid - 1 
        return left
    
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        import bisect
        return bisect.bisect_left(nums, target)
    

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        i, j = len(a)-1, len(b)-1
        carry = 0
        result = []

        while i>=0 or j>=0 or carry: 
            bit_a = int(a[i]) if i >= 0 else 0
            bit_b = int(b[j]) if j >= 0 else 0
            total = bit_a + bit_b + carry
            carry = total // 2
            result.append(str(total % 2))
            i -= 1
            j -= 1

        return "".join(reversed(result))
    

class Solution:
    def reverseBits(self, n: int) -> int:
        result = 0 
        for i in range(32): 
            # shift 1 bit to the left, grab last bit in n
            result = (result << 1) | n & 1
            # remove last bit in n 
            n >>= 1
        return result 
    

# 191 
class Solution:
    def hammingWeight(self, n: int) -> int:
        set_bits = 0 
        while n: 
            set_bits += n & 1
            n >>= 1
        return set_bits

class Solution: 
    def hammingWeight(self, n: int) -> int:
        set_bits = 0 
        while n: 
            n &= (n-1)
            set_bits += 1
        return set_bits
    
# 136 
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0 
        for num in nums: 
            result ^= num
        return result
    
# 9 
class Solution:
    def isPalindrome(self, x: int) -> bool:
        forward_x = str(x)
        backward_x = str(x)[::-1]
        if x<0: 
            forward_x = '-' + forward_x
            backward_x = backward_x + '-'

        i = 0
        for i in range(len(forward_x)): 
            if forward_x[i] != backward_x[i]: 
                return False
        return True

class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0: 
            return False
        return str(x) == str(x)[::-1]
    
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        x_temp = x
        x_backward = 0
        while x_temp: 
            x_backward = x_backward * 10 + x_temp % 10 
            x_temp //= 10
        return x == x_backward
        
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        x_backward = 0
        while x > x_backward: 
            x_backward = x_backward * 10 + x % 10 
            x //= 10
        return x == x_backward or x == x_backward // 10 
        
# 66 
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if digits[-1] != 9: 
            digits[-1] += 1
            return digits
        
        carry = 1
        for i in range(len(digits)-1, -1, -1): 
            if digits[i] == 9: 
                digits[i] = 0
            else: 
                digits[i] += 1
                carry = 0
                break

        if carry: 
            digits = [1] + digits

        return digits
        

class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        
        for i in range(len(digits)-1, -1, -1): 
            if digits[i] < 9: 
                digits[i] += 1
                return digits
            digits[i] = 0

        return [1] + digits
    
# 69
class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 2: 
            return x
        
        low, high = 0, x
        res = 0
        while low <= high: 
            mid = (low + high)//2
            if mid * mid == x: 
                return mid
            elif mid * mid < x: 
                low = mid + 1
                res = mid
            else: 
                high = mid - 1 

        return res 
    
#70 
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2: 
            return n 
        a, b = 1, 2
        for i in range(3, n+1): 
            a, b = b, a + b
        return b
    
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2: 
            return n 
        
        dp = [0] * (n+1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n+1): 
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
    
# 1768
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        result = ""
        while word1 and word2: 
            result += word1[0] + word2[0]
            word1 = word1[1:]
            word2 = word2[1:]

        result += word1
        result += word2
        return result
    
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        result = ""
        index = 0
        min_length = min(len(word1), len(word2))
        while index < min_length:
            result += word1[index] + word2[index]
            index += 1

        result += word1[index:]
        result += word2[index:]
        return result
    
# 1071
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        # find the divisor for 1
        divisors = []
        for i in range(len(str1)): 
            # check 
            if len(str1) % (i+1) == 0 and str1 == str1[:i+1] * (len(str1)//(i+1)): 
                divisor = str1[:i+1]
                divisors.append(divisor)
        
        # check if it is divisor for 2
        for divisor in reversed(divisors):
            if len(str2) % len(divisor) == 0 and str2 == divisor * (len(str2)//len(divisor)): 
                return divisor
        
        return ""

import math
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        if str1 + str2 != str2 + str1: 
            return ""
        
        gcd_len = math.gcd(len(str1), len(str2))
        return str1[:gcd_len]

# 1431 
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        max_kid = max(candies)
        result = []
        for kid in candies: 
            if kid + extraCandies >= max_kid: 
                result.append(True)
            else: 
                result.append(False)

        return result
    
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        max_kid = max(candies)
        return [extraCandies + c >= max_kid for c in candies]
    

# 605 
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        bed_len = len(flowerbed)
        max_flower = 0
        flowerbed = [0] + flowerbed + [0]

        for i in range(1, bed_len+1): 
            if flowerbed[i-1] == 0 and flowerbed[i] == 0 and flowerbed[i+1] == 0: 
                flowerbed[i] = 1
                max_flower += 1
        
        return max_flower >= n 
    
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        count = 0
        bed = [0] + flowerbed + [0]

        for i in range(1, len(bed) - 1): 
            if bed[i-1] == bed[i] == bed[i+1] == 0: 
                bed[i] = 1
                count += 1
                if count >= n: 
                    return True

        return count >= n

# 345 
class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = 'aeiouAEIOU'
        stack = []
        result = ''
        for c in s: 
            if c in vowels: 
                stack.append(c)
        
        for c in s: 
            if c in vowels: 
                result += stack.pop()
            else: 
                result += c
        
        return result
    

class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = set('aeiouAEIOU')
        stack = []
        result = ''
        for c in s: 
            if c in vowels: 
                stack.append(c)
        
        for c in s: 
            if c in vowels: 
                result += stack.pop()
            else: 
                result += c
        
        return result
    
# two pointer implementation
class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = set('aeiouAEIOU')
        s_list = list(s)
        left = 0
        right = len(s)-1

        while left < right: 
            if s_list[left] not in vowels: 
                left += 1
            elif s_list[right] not in vowels: 
                right -= 1
            else: 
                s_list[left], s_list[right] = s_list[right], s_list[left]
                left += 1
                right -= 1
        
        return ''.join(s_list)
    
# 283 
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        pr = 0
        for i in range(len(nums)): 
            if nums[i] != 0: 
                nums[i], nums[pr] = nums[pr], nums[i]
                pr += 1

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        pr = 0
        for i in range(len(nums)): 
            if nums[i] != 0: 
                if i != pr:
                    nums[i], nums[pr] = nums[pr], nums[i]
                pr += 1

# 643 
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        max_avg = float('-inf')
        total = 0
        for i in range(len(nums)): 
            total += nums[i]
            if i >= k: 
                total -= nums[i-k]
            if i >= k-1 or i == len(nums)-1:
                max_avg = max(max_avg, total)
        return max_avg/k
    
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        window_sum = sum(nums[:k])
        max_sum = window_sum
        for i in range(k, len(nums)): 
            window_sum += nums[i] - nums[i-k]
            max_sum = max(max_sum, window_sum)
        return max_sum/k


# 1732
class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        highest = 0 
        current = 0
        for i in gain: 
            current += i
            highest = max(current, highest)
        
        return highest

from itertools import accumulate    
class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        return max(0, *accumulate(gain))