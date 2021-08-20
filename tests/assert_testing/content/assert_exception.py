#test for Timon




def divide(denominator):
    try:
        outcome = 10 / denominator


    except ZeroDivisionError as inst:
        print("denominator cannot be 0")
        outcome = inst
    except ValueError as inst:
        print("denominator must be a number")
        outcome = inst

    return outcome





