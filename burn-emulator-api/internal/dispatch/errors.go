package dispatch

import "google.golang.org/api/googleapi"

// reports whether err is a *googleapi.Error with this HTTP code.
func isStatusCode(err error, code int) bool {
	apiErr, ok := err.(*googleapi.Error)
	return ok && apiErr.Code == code
}
