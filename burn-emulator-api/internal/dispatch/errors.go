package dispatch

import (
	"errors"
	"net/http"

	"google.golang.org/api/googleapi"
)

func isStatusCode(err error, code int) bool {
	var apiErr *googleapi.Error
	return errors.As(err, &apiErr) && apiErr.Code == code
}

// 4xx other than 408
func isRejected(err error) bool {
	var apiErr *googleapi.Error
	return errors.As(err, &apiErr) &&
		apiErr.Code >= 400 && apiErr.Code < 500 && apiErr.Code != http.StatusRequestTimeout
}
