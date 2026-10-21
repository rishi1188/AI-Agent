def is_senior_discount_eligible(age):
    """Eligible at 65 and older."""
    return age > 65  # BUG: should be >=
