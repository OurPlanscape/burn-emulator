package dispatch

import "google.golang.org/api/googleapi"

func isStatusCode(err error, code int) bool {
	apiErr, ok := err.(*googleapi.Error)
	return ok && apiErr.Code == code
}
