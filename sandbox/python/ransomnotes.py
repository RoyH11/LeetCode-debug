def ransomnote(mag, ransom): 
    # create dictionary
    my_dict = {}

    # populate dictionary from mag
    for char in mag: 
        if char in my_dict:
            my_dict[char] += 1
        else: 
            my_dict[char] = 1

    # check each letter in ransom
    for char in ransom: 
        if char in my_dict:
            # if the letter exists in my_dict: value --
            # if value is negative: return false
            my_dict[char] -= 1
            
            if my_dict[char] < 0: 
                return False
        else: 
            # else (letter doesn't exist in my_dict): return false
            return False


    return True


# 2 inputs: 
# mag: available letters
# ransom: target 
# 