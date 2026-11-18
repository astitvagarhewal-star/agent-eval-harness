package main

import "testing"

func TestSumChannel(t *testing.T) {
	got := SumChannel([]int{1, 2, 3, 4})
	if got != 10 {
		t.Fatalf("expected 10, got %d", got)
	}
}
