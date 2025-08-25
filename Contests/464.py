class Solution:
    def gcdOfOddEvenSums(self, n: int) -> int:
        # first find the sums 
        odd_sum = (1 + 2*n-1) * n / 2
        even_sum = (2 + 2*n) * n / 2

        while even_sum:
            odd_sum, even_sum = even_sum, odd_sum % even_sum
        
        return int(odd_sum)