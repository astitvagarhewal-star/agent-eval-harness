"""A loader that swallows conversion errors."""

def load_number(text):
    try:
        return int(text)
    except Exception:
        return 0
