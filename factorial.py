import math


def factorial(n):
    if n == 0 or n == 1:
        return 1
    elif n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    else:
        return n * factorial(n-1)

try:
    print(f"The factorial of 5 is {factorial(5)}")
except ValueError as e:
    print(e)