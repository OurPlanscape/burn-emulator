package ratelimit

import (
	"sync"
	"time"

	"golang.org/x/time/rate"
)

type Store struct {
	mu       sync.Mutex
	limiters map[string]*rate.Limiter
	rps      rate.Limit
	burst    int
}

func NewStore(rps float64, burst int) *Store {
	return &Store{
		limiters: make(map[string]*rate.Limiter),
		rps:      rate.Limit(rps),
		burst:    burst,
	}
}

func (s *Store) Allow(key string) bool {
	return s.get(key).Allow()
}

func (s *Store) get(key string) *rate.Limiter {
	s.mu.Lock()
	defer s.mu.Unlock()
	l, ok := s.limiters[key]
	if !ok {
		l = rate.NewLimiter(s.rps, s.burst)
		s.limiters[key] = l
	}
	return l
}

func (s *Store) Cleanup(interval time.Duration, stop <-chan struct{}) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			s.mu.Lock()
			for k, l := range s.limiters {
				if l.Tokens() >= float64(s.burst) {
					delete(s.limiters, k)
				}
			}
			s.mu.Unlock()
		case <-stop:
			return
		}
	}
}