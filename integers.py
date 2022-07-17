lst = [10,100,1200]



def get_zero_count(nums):
    """
    Parameters:
        nums : Integer type
    Returns:
        Integer - returns the number of zeros in the list
    
    """
    c = 0
    for num in nums:
        for c in str(num):
            if int(c) == 0:
                c+=1
    return c






