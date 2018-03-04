package main

import (
	// "encoding/hex"
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
	// "io"
	"log"
	// "time"
)

const (
	// API_KEY      string = "l3zela16irfa6rax"
	// REQ_TOKEN    string = "4be382t2y1czjvpwia91pomqenxx1d6j" //Constant for this session
	// API_SECRET   string = "qefc9t3ovposnzvvy94k3sckna7vwuxs"
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

func find(records [][]string, val string, col int) string {
	var blank string
	for _, row := range records {
		if row[col] == val && row[11] == "NSE" {
			// fmt.Println(i)
			return row[0]
		}
	}
	return blank
	// return empty []byte
}

func main() {
	// os.RemoveAll("data/.csv")
	// os.MkdirAll("data/", 0777)
	HistPool := make(chan bool, MAX_ROUTINES)
	// REGULATORY REQUIREMENT
	// Go to https://kite.trade/connect/login?api_key=l3zela16irfa6rax to get request token for the day
	// curl https://api.kite.trade/instruments to retreive master list
	fmt.Println("Starting Client")
	client := kiteGo.KiteClient(CONFIG_FILE)

	// f, _ := os.Open("instruments.txt")

	instList, _ := os.Open("instruments.csv")
	records, err := csv.NewReader(bufio.NewReader(instList)).ReadAll()
	if err != nil {
		panic(err)
	}
	// fmt.Printf("%q is at row %v\n", "two", )

	FROM := "2009-02-17" //currentDate.Format("2006-01-02") // Start of history
	TO := "2018-02-24"   //currentDate.Add(time.Hour * 24).Format("2006-01-02") //currentDate.Format("2006-01-02")   // End of history

	scripFile, err := os.Open("tradeList.txt") //instruments being considered
	if err != nil {
		log.Fatal(err)
	}
	scripList := bufio.NewScanner(scripFile) // instruments being considered

	for scripList.Scan() { // looped through
		fmt.Println("HERE")
		fname := string(scripList.Text())                                             // name of scrip
		exchangeToken := find(records, scripList.Text(), 2)                           // find the exchange token
		fmt.Printf("ADDED %s to QUEUE \n", (fname))                                   // print out info
		go client.GetHistorical(DAY, exchangeToken, FROM, TO, fname+".csv", HistPool) // get data

	}
	fmt.Println("Waiting")
	fmt.Println(string(scripList.Text()))

	// exchangeToken := "2933761"
	// fname := "JPASSOCIAT"
	// client.GetHistorical(MINUTE, exchangeToken, FROM, TO, fname+".csv", HistPool)

	fmt.Println("DONE WITH ALL TASKS")
	var input string
	fmt.Println("Enter to end...")
	fmt.Scanln(&input)

}
