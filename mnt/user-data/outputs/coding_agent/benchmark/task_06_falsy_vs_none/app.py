def get_or_default(value, default):
    """Return `value` unless it's missing (None), in which case return default."""
    return default if not value else value  # BUG: 'not value' also treats 0, '', [] as missing
