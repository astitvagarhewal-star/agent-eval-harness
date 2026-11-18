package main

import "sync"

type Counter struct {
	mu sync.Mutex
	n  int
}

func (c *Counter) Add(v int) {
	c.mu.Lock()
	// BUG: Unlock is forgotten, so subsequent Add/Value calls deadlock.
	c.n += v
}

func (c *Counter) Value() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.n
}
