package main

import (
	"fmt"
	// "io/ioutil"
	// "kite-go/helper"
	"kite-go/kiteGo"
	// "strings"
	//"flag"
	"os"
	// "sync"
	"bufio"
	"encoding/csv"
	"io"
)

const (
	API_KEY      string = "l3zela16irfa6rax"
	REQ_TOKEN    string = "4be382t2y1czjvpwia91pomqenxx1d6j" //Constant for this session
	API_SECRET   string = "qefc9t3ovposnzvvy94k3sckna7vwuxs"
	MINUTE       string = "minute"
	THREE_MIN    string = "3minute"
	FIVE_MIN     string = "5minute"
	TEN_MIN      string = "10minute"
	FIFTEEN_MIN  string = "15minute"
	THIRTY_MIN   string = "30minute"
	SIXTY_MIN    string = "60minute"
	DAY          string = "day"
	MAX_ROUTINES int    = 3
	CONFIG_FILE  string = "config.json"
)

func main() {
	os.RemoveAll("data/.csv")
	os.MkdirAll("data/", 0777)
	HistPool := make(chan bool, MAX_ROUTINES)
	// REGULATORY REQUIREMENT
	// Go to https://kite.trade/connect/login?api_key=l3zela16irfa6rax to get request token for the day
	// curl https://api.kite.trade/instruments to retreive master list
	fmt.Println("Starting Client")
	client := kiteGo.KiteClient(CONFIG_FILE)
	f, _ := os.Open("instruments.txt")
	FROM := "2012-01-01"
	TO := "2017-06-01"

	instruments := csv.NewReader(bufio.NewReader(f))
	for {
		record, err := instruments.Read()
		if err == io.EOF {
			break
		}
		exchangeToken := record[0]
		fname := record[2]
		// tickSize := record[7]
		segment := record[11]
		// fmt.Println(fname + " " + exchangeToken + " " + tickSize + " " + segment)
		//fmt.Println(tickSize == "1")
		if segment == "NSE" || segment == "BSE" {
			fmt.Printf("ADDED %s to QUEUE \n", (fname))
			go client.GetHistorical(MINUTE, exchangeToken, FROM, TO, fname+".csv", HistPool)
		}
	}
	//

	//
	// go client.GetHistorical(MINUTE, "2012673", "2005-01-01", "2017-06-01", "GePowerIndia.csv", HistPool)
	// //go client.GetHistorical(MINUTE, "136522756", "2017-01-01", "2017-06-01", "A2Z.csv", HistPool)
	// // go client.GetHistorical(MINUTE, "136385796", "2017-01-01", "2017-06-01", "Voltamp.csv", HistPool)
	// go client.GetHistorical(FIVE_MIN, "4296449", "2017-01-01", "2017-06-01", "GETD.csv", HistPool)
	// go client.GetHistorical(FIVE_MIN, "3884545", "2017-01-01", "2017-06-01", "TRIL.csv", HistPool)

	fmt.Println("DONE WITH ALL TASKS")
	var input string
	fmt.Println("Enter to end...")
	fmt.Scanln(&input)

}
