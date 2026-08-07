package auth

import (
	"context"
	"fmt"
	"log/slog"
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
		slog.Warn("auth: rejected request, no allowed service accounts configured")
		return false
	}

	tok := bearerToken(r)
	if tok == "" {
		slog.Warn("auth: rejected request, missing bearer token")
		return false
	}

	payload, err := v.validator.Validate(r.Context(), tok, v.audience)
	if err != nil {
		slog.Warn("auth: rejected request, token validation failed", "error", err)
		return false
	}

	verified, _ := payload.Claims["email_verified"].(bool)
	if !verified {
		slog.Warn("auth: rejected request, email not verified")
		return false
	}
	email, _ := payload.Claims["email"].(string)
	if email == "" {
		slog.Warn("auth: rejected request, missing email claim")
		return false
	}

	if !v.allowed[email] {
		slog.Warn("auth: rejected verified caller not on allowlist", "email", email)
		return false
	}

	return true
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
