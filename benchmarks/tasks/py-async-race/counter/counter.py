"""An async counter with a read-modify-write race condition."""

import asyncio


class Counter:
    def __init__(self):
        self.value = 0

    async def increment(self):
        # BUG: non-atomic read-modify-write with a yield in between.
        current = self.value
        await asyncio.sleep(0)  # yield control, widening the race window
        self.value = current + 1


async def main():
    c = Counter()
    await asyncio.gather(*[c.increment() for _ in range(100)])
    return c.value
