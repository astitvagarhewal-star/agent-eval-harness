"""Hidden evaluation test for the mutable default argument bug."""

from utils import append_item


def test_mutable_default_is_fresh():
    assert append_item(1) == [1]
    assert append_item(2) == [2]
    assert append_item(3) == [3]


if __name__ == "__main__":
    test_mutable_default_is_fresh()
    print("ok")
