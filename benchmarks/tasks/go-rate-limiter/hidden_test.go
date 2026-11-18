package main

import "testing"

func TestRateLimiter(t *testing.T) {
	l := NewLimiter()
	results := make([]bool, 4)
	for i := range results {
		results[i] = l.Allow()
	}
	want := []bool{true, true, true, false}
	for i := range results {
		if results[i] != want[i] {
			t.Fatalf("expected %v, got %v", want, results)
		}
	}
}
