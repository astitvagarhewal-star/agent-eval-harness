"""Hidden evaluation test for the circular import bug."""

import app


def test_no_circular_import():
    assert app.APP_NAME == "myapp"


if __name__ == "__main__":
    test_no_circular_import()
    print("ok")
