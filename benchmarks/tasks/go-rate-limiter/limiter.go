package main

type Limiter struct {
	tokens int
}

func NewLimiter() *Limiter {
	return &Limiter{tokens: 3}
}

func (l *Limiter) Allow() bool {
	// BUG: off-by-one allows one request too many (4 instead of 3).
	if l.tokens < 0 {
		return false
	}
	l.tokens--
	return true
}
