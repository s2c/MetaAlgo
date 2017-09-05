package helper

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

func CheckError(e error, resp ...*http.Response) {
	if resp != nil && resp[0].StatusCode != 200 {
		CheckHttpBody(resp[0])
		log.Printf("HTTP ERROR")

	} else if e != nil {
		log.Fatal("Unknown Error")
	}
}

func CheckHttpBody(resp *http.Response) {
	data, _ := ioutil.ReadAll(resp.Body)
	//CheckError(err)
	fmt.Println(string(data))
}

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
	} else if (time.Now().Sub(fi.ModTime())) > time.Hour*24 {
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
