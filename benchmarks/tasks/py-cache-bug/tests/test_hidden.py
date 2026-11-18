"""Hidden evaluation test for the caching bug."""

from cache import get_config


def test_cache_keyed_by_argument():
    assert get_config("aaa") == 3
    assert get_config("bbbb") == 4
    assert get_config("aaa") == 3


if __name__ == "__main__":
    test_cache_keyed_by_argument()
    print("ok")
