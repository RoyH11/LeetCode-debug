from typing import List

#135 
class Solution:
    def candy(self, ratings: List[int]) -> int:
        # find lowest
        worst_kid = 0
        for i in range(len(ratings)): 
            if ratings[i] <= ratings[worst_kid]: 
                worst_kid = i 

        total = 1
        if worst_kid == 0: 
            # move right 
            prev = 1
            for i in range(1, len(ratings)): 
                if ratings[i] > ratings[i-1]: 
                    prev += 1
                if ratings[i] <= ratings[i-1]: 
                    prev = 1
                total += prev
                

        elif worst_kid == len(ratings) - 1: 
            # move left 
            prev = 1 
            for i in range(len(ratings) - 2, -1, -1): 
                if ratings[i] > ratings[i+1]: 
                    prev += 1
                if ratings[i] <= ratings[i+1]: 
                    prev = 1
                total += prev
                
        else: 
            # move both way 
            pass