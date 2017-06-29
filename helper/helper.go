package helper

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
)

func CheckError(e error, resp ...*http.Response) {
	if resp != nil && resp[0].StatusCode != 200 {
		CheckHttpBody(resp[0])
		log.Panic("HTTP ERROR")

	} else if e != nil {
		log.Panic("Unknown Error")
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
