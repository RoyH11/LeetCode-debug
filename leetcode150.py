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