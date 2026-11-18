"""Hidden evaluation test for the exception-swallowing bug."""

from loader import load_number


def test_load_number_invalid_raises():
    try:
        load_number("abc")
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-numeric input")


def test_load_number_valid():
    assert load_number("42") == 42


if __name__ == "__main__":
    test_load_number_invalid_raises()
    test_load_number_valid()
    print("ok")
