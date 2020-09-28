import math 



def perfect_sqr(num):
    """
    checks whether a number is perfect square.
    """

    sqr = math.sqrt(num)

    return (num - sqr == 0)