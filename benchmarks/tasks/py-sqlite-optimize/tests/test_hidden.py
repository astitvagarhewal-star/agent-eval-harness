"""Hidden evaluation test for SQLite query optimization."""

import sqlite3

from reports import count_high_value


def test_count_high_value():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE orders (id INTEGER, amount INTEGER)")
    conn.execute("INSERT INTO orders VALUES (1, 50)")
    conn.execute("INSERT INTO orders VALUES (2, 100)")
    conn.execute("INSERT INTO orders VALUES (3, 200)")
    assert count_high_value(conn) == 2


if __name__ == "__main__":
    test_count_high_value()
    print("ok")
