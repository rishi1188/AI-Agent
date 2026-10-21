def add_to_cart(item, cart=[]):
    """Add an item to a cart and return it."""
    # BUG: mutable default argument -- the SAME list is reused across
    # every call that doesn't pass its own cart, so state leaks between
    # unrelated calls.
    cart.append(item)
    return cart
