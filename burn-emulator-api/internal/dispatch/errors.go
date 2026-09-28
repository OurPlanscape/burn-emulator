package dispatch

import (
	"errors"
	"net/http"

	"google.golang.org/api/googleapi"
)

func isStatusCode(err error, code int) bool {
	apiErr, ok := err.(*googleapi.Error)
	return ok && apiErr.Code == code
}

// timeouts, network errors, and 5xx are ambiguous. leaving that to timeout
func isRejected(err error) bool {
	var apiErr *googleapi.Error
	return errors.As(err, &apiErr) &&
		apiErr.Code >= 400 && apiErr.Code < 500 && apiErr.Code != http.StatusRequestTimeout
}
