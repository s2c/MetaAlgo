package helper

import (
	// "fmt"
	"github.com/buger/jsonparser"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	OUT_OF_DATE_RANGE  int           = 1
	UNKNOWN_HTTP_ERROR int           = 2
	UNKNOWN_ERROR      int           = 9
	ALL_GOOD           int           = 0
	INP_EXC_ERR_MSG    string        = "InputException"
	MAX_TIME           time.Duration = 14 * 24 * time.Hour
	VALID_TIME         time.Duration = time.Hour * 12
)

//Either expand or kill, stifling right now
func CheckError(e error, resp ...*http.Response) int {
	if resp != nil {

		data, _ := ioutil.ReadAll(resp[0].Body)
		msg, _ := jsonparser.GetString(data, "error_type")

		if msg == INP_EXC_ERR_MSG {
			return OUT_OF_DATE_RANGE
		} else if e != nil {
			log.Panic(e.Error())
			return UNKNOWN_HTTP_ERROR
		}

	} else if e != nil {
		log.Panic(e.Error())
		return UNKNOWN_ERROR
	}
	return ALL_GOOD
}

//
func AddCurr(curr time.Time, final time.Time) time.Time {
	remaining := final.Sub(curr)
	//if there are more than 28 days remaining, then just add 28 to curr and return it
	if remaining > MAX_TIME {
		return curr.Add(MAX_TIME)
	} else { //Otherwise just return the current time
		return curr
	}

}

// Formats the JSON message data appropriately for local storage
func FormatData(data string) []byte {
	dat := strings.Replace(string(data), "[", "", -1)
	dat = strings.Replace(dat, "]", "", -1)
	dat = strings.Replace(dat, "\"", "", -1)
	return ReplaceNth([]byte(dat), ',', '\n', 6)

}

// Returns true or false if a file that contains the access token CAN STILL BE VALID
// DOES NOT CHECK THE ACTUAL TOKEN, JUST THE FILE TO SEE IF IT CAN BE VALID
func AccessTokenValidity(AccesTokenFile string) bool {
	fi, err := os.Stat(AccesTokenFile)
	if os.IsNotExist(err) {
		return false
	} else if (time.Now().Sub(fi.ModTime())) > VALID_TIME {
		return false
	} else {
		return true
	}
}

func ReplaceNth(dat []byte, oldChar byte, newChar byte, nth int) []byte {
	count := 0
	for i := 0; i < len(dat); i++ {
		if dat[i] == oldChar {
			count += 1
		}
		if count%nth == 0 && dat[i] == oldChar {
			dat[i] = newChar
			count = 0
		}
	}
	return dat
}

//wrapper to ensure there is a default 10 second timeout
func HttpClient(timeout time.Duration) *http.Client {
	client := &http.Client{}
	client.Timeout = timeout * time.Second
	return client
}
