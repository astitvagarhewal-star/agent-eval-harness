package main

import (
	"testing"
	"time"
)

func TestCounter(t *testing.T) {
	c := &Counter{}
	done := make(chan struct{})
	go func() {
		c.Add(1)
		c.Add(1)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("Add deadlocked (missing Unlock)")
	}
	if c.Value() != 2 {
		t.Fatalf("expected 2, got %d", c.Value())
	}
}
