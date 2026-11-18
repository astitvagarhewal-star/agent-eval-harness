package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func targetHandler(w http.ResponseWriter, r *http.Request) {
	io.WriteString(w, "target-ok")
}

func TestMiddleware(t *testing.T) {
	h := Middleware(http.HandlerFunc(targetHandler))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	h.ServeHTTP(rec, req)
	body, _ := io.ReadAll(rec.Body)
	if string(body) != "target-ok" {
		t.Fatalf("expected target-ok, got %q", string(body))
	}
}
