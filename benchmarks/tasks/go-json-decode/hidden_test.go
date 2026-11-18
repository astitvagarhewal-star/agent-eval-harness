package main

import "testing"

func TestDecode(t *testing.T) {
	c, err := Decode([]byte(`{"port": 8080}`))
	if err != nil {
		t.Fatal(err)
	}
	if c.Port != 8080 {
		t.Fatalf("expected port 8080, got %d", c.Port)
	}
}
