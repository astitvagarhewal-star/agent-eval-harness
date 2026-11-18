"""Reports over an SQLite orders table (query bug to fix)."""

import sqlite3


def count_high_value(conn):
    # BUG: off-by-one excludes orders worth exactly 100.
    cur = conn.execute("SELECT COUNT(*) FROM orders WHERE amount > 100")
    return cur.fetchone()[0]
