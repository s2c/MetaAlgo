package helper

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
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
