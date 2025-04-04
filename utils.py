def decimal_to_pentanary(decimal_number):
    
    pentanary_number = ""
    for _ in range(4):
        remainder = decimal_number % 5
        pentanary_number = str(remainder) + pentanary_number
        decimal_number //= 5
    
    return pentanary_number