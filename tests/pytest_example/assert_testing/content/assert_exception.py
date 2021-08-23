#test for Timon




def divide(denominator):
    try:
        outcome = 10 / denominator

    except ZeroDivisionError as inst:
        print("denominator cannot be 0")
        outcome = type(inst).__name__
    except TypeError as inst:
        print("denominator must be a number")
        outcome = type(inst).__name__

    return outcome





