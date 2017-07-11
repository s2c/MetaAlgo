package main

import (
	"fmt"
	// // "io/ioutil"
	// "kite-go/helper"
	"kite-go/kiteGo"
	// // "strings"
	//"flag"
	"os"
	// "sync"
)

const (
	API_KEY      string = "l3zela16irfa6rax"
	REQ_TOKEN    string = "hg9qdg22swgle73kvi8dlxty6mac01o4" //Constant for this session
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
)

func main() {
	os.RemoveAll("data/")
	os.MkdirAll("data/", 0777)
	// REGULATORY REQUIREMENT
	// Go to https://kite.trade/connect/login?api_key=l3zela16irfa6rax to get request token for the day
	// curl https://api.kite.trade/instruments to retreive master list
	fmt.Println("Starting Client")
	client := kiteGo.KiteClient(API_KEY, REQ_TOKEN, API_SECRET)
	client.Login()

	HistPool := make(chan bool, MAX_ROUTINES)

	go client.GetHistorical(MINUTE, "256265", "2017-01-01", "2017-06-01", "Nifty50.csv", HistPool)
	go client.GetHistorical(MINUTE, "2012673", "2017-01-01", "2017-06-01", "GePowerIndia.csv", HistPool)
	go client.GetHistorical(MINUTE, "136522756", "2017-01-01", "2017-06-01", "A2Z.csv", HistPool)
	go client.GetHistorical(MINUTE, "136385796", "2017-01-01", "2017-06-01", "Voltamp.csv", HistPool)
	go client.GetHistorical(MINUTE, "4296449", "2017-01-01", "2017-06-01", "GETD.csv", HistPool)
	go client.GetHistorical(MINUTE, "3884545", "2017-01-01", "2017-06-01", "TRIL.csv", HistPool)

	// helper.CheckPoolDone(HistPool)

	fmt.Println("DONE WITH ALL TASKS")
	var input string
	fmt.Println("Enter to end...")
	fmt.Scanln(&input)

}
