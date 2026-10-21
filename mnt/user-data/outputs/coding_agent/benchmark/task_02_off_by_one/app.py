def sum_first_n(numbers, n):
    """Sum the first n numbers in the list."""
    total = 0
    for i in range(n - 1):  # BUG: excludes the nth element
        total += numbers[i]
    return total
