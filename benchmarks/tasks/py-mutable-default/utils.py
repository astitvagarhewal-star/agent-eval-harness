"""A function with a mutable default-argument bug."""


def append_item(item, items=[]):
    items.append(item)
    return items
