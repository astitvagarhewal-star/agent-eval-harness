package main

import (
	"reflect"
	"testing"
)

func TestCopyInts(t *testing.T) {
	orig := []int{1, 2, 3}
	cp := CopyInts(orig)
	orig[0] = 99
	if cp[0] != 1 {
		t.Fatalf("expected copy to be independent, got %d", cp[0])
	}
	if !reflect.DeepEqual(cp, []int{1, 2, 3}) {
		t.Fatalf("expected [1 2 3], got %v", cp)
	}
}
