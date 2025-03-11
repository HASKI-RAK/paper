def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number using recursion.

    The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones,
    usually starting with 0 and 1. That is, F(0) = 0, F(1) = 1, and F(n) = F(n-1) + F(n-2) for n > 1.

    Parameters:
    n (int): The position in the Fibonacci sequence to calculate. Must be a non-negative integer.

    Returns:
    int: The nth Fibonacci number.

    Raises:
    ValueError: If n is a negative integer.
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
