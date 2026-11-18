"""Hidden evaluation test for the dataclass mutable default bug."""

from model import Task


def test_dataclass_default_isolated():
    t1 = Task()
    t1.tags.append("a")
    t2 = Task()
    assert t2.tags == []


if __name__ == "__main__":
    test_dataclass_default_isolated()
    print("ok")
