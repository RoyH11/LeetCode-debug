def calcSubset(A, res, subset, index):
    # Add the current subset to the result list
    res.append(subset[:])
    #print('*')
    print("rec start",index)
    #if index == 3: print('*')
    #print(subset[:])
 
    # Generate subsets by recursively including and excluding elements
    for i in range(index, len(A)):
        print("in loop", index, "-", i)
        # Include the current element in the subset
        subset.append(A[i])
        #print(subset)
 
        # Recursively generate subsets with the current element included
        calcSubset(A, res, subset, i + 1)
        #print("rec back")
 
        # Exclude the current element from the subset (backtracking)
        subset.pop()
    print("out loop", index)
 
 
def subsets(A):
    subset = []
    res = []
    index = 0
    calcSubset(A, res, subset, index)
    return res
 
 
# Driver code
if __name__ == "__main__":
    array = [0, 1, 2]
    res = subsets(array)
 
    # Print the generated subsets
    for subset in res:
        print(*subset)