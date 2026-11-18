"""A cache that is not keyed by its argument (stale-cache bug)."""

_computed = {}


def compute(key):
    return len(key)


def get_config(key):
    # BUG: the cache is not keyed by `key`, so every call returns the first
    # computed value regardless of the argument.
    if not _computed:
        _computed["value"] = compute(key)
    return _computed["value"]
