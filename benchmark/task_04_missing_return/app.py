def classify_number(n):
    """Classify a number as positive, negative, or zero."""
    if n > 0:
        return "positive"
    elif n < 0:
        return "negative"
    # BUG: no branch for n == 0 -- falls through and returns None
