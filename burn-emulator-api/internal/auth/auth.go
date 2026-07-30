package auth

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"google.golang.org/api/idtoken"
)

type Verifier struct {
	validator *idtoken.Validator
	audience  string
	allowed   map[string]bool
}

func NewVerifier(ctx context.Context, audience string, allowedServiceAccounts []string) (*Verifier, error) {
	validator, err := idtoken.NewValidator(ctx)
	if err != nil {
		return nil, fmt.Errorf("creating idtoken validator: %w", err)
	}

	allowed := make(map[string]bool, len(allowedServiceAccounts))
	for _, sa := range allowedServiceAccounts {
		if sa = strings.TrimSpace(sa); sa != "" {
			allowed[sa] = true
		}
	}

	return &Verifier{validator: validator, audience: audience, allowed: allowed}, nil
}

func (v *Verifier) Allow(r *http.Request) bool {
	if len(v.allowed) == 0 {
		return false
	}

	tok := bearerToken(r)
	if tok == "" {
		return false
	}

	payload, err := v.validator.Validate(r.Context(), tok, v.audience)
	if err != nil {
		return false
	}

	verified, _ := payload.Claims["email_verified"].(bool)
	if !verified {
		return false
	}
	email, _ := payload.Claims["email"].(string)
	if email == "" {
		return false
	}

	return v.allowed[email]
}

func bearerToken(r *http.Request) string {
	const prefix = "Bearer "
	h := r.Header.Get("Authorization")
	if !strings.HasPrefix(h, prefix) {
		return ""
	}
	return strings.TrimPrefix(h, prefix)
}

// only interal calls are made so this should be fine
func ClientIP(r *http.Request) string {
	if ip := r.Header.Get("X-Forwarded-For"); ip != "" {
		return ip
	}
	return r.RemoteAddr
}
