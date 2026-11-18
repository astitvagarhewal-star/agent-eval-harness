package main

import "net/http"

// Middleware wraps an http.Handler.
func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// BUG: the wrapped handler is never invoked.
		w.WriteHeader(http.StatusOK)
	})
}
