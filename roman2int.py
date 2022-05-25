def romanToInt(self, s: str) -> int:
    roman2int = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}
    n = 0
    prev = None
    for char in s:
        if prev is None:
            prev = char
        else:
            if roman2int[char] > roman2int[prev]:
                n = n - roman2int[prev]
            else:
                n = n + roman2int[prev]
        prev = char
    n = n + roman2int[prev]
    return n