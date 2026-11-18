"""Hidden evaluation test for the async race condition."""

import asyncio

from counter import Counter, main


def test_counter_value():
    assert asyncio.run(main()) == 100


def test_increment_is_atomic():
    c = Counter()
    asyncio.run(c.increment())
    assert c.value == 1


if __name__ == "__main__":
    test_counter_value()
    test_increment_is_atomic()
    print("ok")
