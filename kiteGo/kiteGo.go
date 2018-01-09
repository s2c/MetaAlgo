package kiteGo

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"github.com/buger/jsonparser"
	"io/ioutil"
	"kite-go/helper"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

const (
	API_ROOT      string = "https://api.kite.trade"
	TOKEN_URL     string = "/session/token"
	PARAMETERS    string = "/parameters"
	USER_MARGINS  string = "/user/margins/"
	ORDERS        string = "/orders"
	TRADES        string = "/trades"
	HISTORICAL    string = "/instruments/historical/"
	API_FIELD     string = "API_KEY"
	SECRET_FIELD  string = "API_SECRET"
	REQ_TOK_FIELD string = "REQ_TOKEN"
	MINUTE        string = "minute"
	THREE_MIN     string = "3minute"
	FIVE_MIN      string = "5minute"
	TEN_MIN       string = "10minute"
	FIFTEEN_MIN   string = "15minute"
	THIRTY_MIN    string = "30minute"
	SIXTY_MIN     string = "60minute"
	DAY           string = "day"
	EX_CDS        string = "CDS"
	EX_NFO        string = "NFO"
	EX_BSE        string = "BSE"
	EX_BFO        string = "BFO"
	EX_NSE        string = "NSE"
	EX_MCX        string = "MCX"
	POST          string = "POST"
	GET           string = "GET"
	KITE_VERSION  string = "3"
)
const (
	DEFAULT_TIMEOUT         time.Duration = 60
	DEFAULT_FULLTIME_LAYOUT string        = "2006-01-02T15:04:05-0700"
	DEFAULT_DATE_LAYOUT     string        = "2006-01-02"
	ACC_TOKEN_FILE          string        = "ACCTOKEN.txt"
)
const (
	OUT_OF_DATE_RANGE  int    = 1
	UNKNOWN_HTTP_ERROR int    = 2
	UNKNOWN_ERROR      int    = 9
	ALL_GOOD           int    = 0
	INP_EXC_ERR_MSG    string = "InputException"
)

//User Tokens
type kiteClient struct {
	Client_API_KEY    string
	Client_REQ_TOKEN  string
	Client_API_SECRET string
	Client_ACC_TOKEN  string
	Client_PUB_TOKEN  string
}

//Create a new Kite Client
func KiteClient(configFile string, ReqToken ...string) *kiteClient {

	k := &kiteClient{}
	k.Client_PUB_TOKEN = "" // Unset

	// jsonParser magic
	data, err := ioutil.ReadFile(configFile)

	if err != nil {
		log.Fatal(err)
	}

	k.Client_API_SECRET, _ = jsonparser.GetString(data, SECRET_FIELD)
	k.Client_API_KEY, _ = jsonparser.GetString(data, API_FIELD)
	k.Client_REQ_TOKEN, _ = jsonparser.GetString(data, REQ_TOK_FIELD)

	//FIle Read, now need to check if we still need to login today.

	// Check if AccessToken is still valid
	validAcc := helper.AccessTokenValidity(ACC_TOKEN_FILE)
	if validAcc {
		contByte, err := ioutil.ReadFile(ACC_TOKEN_FILE)

		if err != nil {
			log.Fatal(err)
		}

		contString := string(contByte)
		k.SetAccessToken(contString) // If yes then just read and set Access Token

	} else {
		k.login()
		ioutil.WriteFile(ACC_TOKEN_FILE, []byte(k.Client_ACC_TOKEN), 0644)
		// if error, then log and exit
		// Write k.Client_ACC_TOKEN to file if not valid then we need to login and generate the Access Token

	}

	return k
}

//Set the access token
func (k *kiteClient) SetAccessToken(acc_token string) {
	k.Client_ACC_TOKEN = acc_token
}

func (k *kiteClient) SetPublicToken(pub_token string) {
	k.Client_PUB_TOKEN = pub_token
}

//Login to retrieve access token. API_KEY,REQ_TOKEN, API_SECRET MUST BE SET
func (k *kiteClient) login() {
	//Compute Hash and store checksum
	hasher := sha256.New()
	checksum := k.Client_API_KEY + k.Client_REQ_TOKEN + k.Client_API_SECRET
	hasher.Write([]byte(checksum))
	checksum = hex.EncodeToString(hasher.Sum(nil))
	//Create Http client, add the required fields
	hc := helper.HttpClient(DEFAULT_TIMEOUT)
	form := url.Values{}
	form.Add("api_key", k.Client_API_KEY)
	form.Add("request_token", k.Client_REQ_TOKEN)
	form.Add("checksum", checksum)
	//Create the request
	req, err := http.NewRequest(POST, API_ROOT+TOKEN_URL, strings.NewReader(form.Encode()))
	if err != nil {
		log.Fatal(err)
	}

	//Do the request
	resp, err := hc.Do(req)
	if err != nil {
		log.Fatal(err)
	}

	//Read the response
	message, err := (ioutil.ReadAll(resp.Body))
	if err != nil {
		log.Fatal(err)
	}

	//parse accesstoken and store
	accToken, _, _, err := jsonparser.Get(message, "data", "access_token")
	if err != nil {
		log.Fatal(err)
	}
	k.SetAccessToken(string(accToken))
	if k.Client_ACC_TOKEN != "" {

		fmt.Println("Access Key set")
	}

	//parse publictoken and store
	pubToken, _, _, err := jsonparser.Get(message, "data", "public_token")
	if err != nil {
		log.Fatal(err)
	}
	k.SetPublicToken(string(pubToken))
	if k.Client_PUB_TOKEN != "" {
		fmt.Println("Public Key set")
	}

}

//Builds the form that sends the request.
// Returns pointer to http.request
func (k *kiteClient) histFormBuilder(FROM string, TO string, DURATION string, exchangeToken string) *http.Request {

	req, err := http.NewRequest(GET, API_ROOT+HISTORICAL+exchangeToken+"/"+DURATION, nil)
	if err != nil {
		log.Fatal(err)
	}

	form := req.URL.Query()
	form.Add("X-Kite-Version", KITE_VERSION)
	form.Add("api_key", k.Client_API_KEY)
	form.Add("access_token", k.Client_ACC_TOKEN)
	form.Add("from", FROM)
	form.Add("to", TO)
	req.URL.RawQuery = form.Encode()

	req, err = http.NewRequest(GET, req.URL.String(), nil)
	if err != nil {
		log.Fatal(err)
	}

	return req

}

//dates of format yyyy-mm-dd
//concurrent safe for now.
//Limited to 1 call per second per instance. Current max limit is 3 calls per second, which is ridiculously slow but whatever.
//TODO: Store ticker name as well as the data
func (k *kiteClient) GetHistorical(duration string, exchangeToken string, from string, to string, filename string, ch chan bool) {
	ch <- true
	//fmt.Println("WE HERE")
	//Create HTTP client
	fmt.Printf("Starting to acquire %s from %s to %s \n", filename[0:len(filename)-4], from, to)
	hc := helper.HttpClient(DEFAULT_TIMEOUT)
	curr, _ := time.Parse(DEFAULT_DATE_LAYOUT, from)
	final, _ := time.Parse(DEFAULT_DATE_LAYOUT, to)
	valid := false //used to find the right starting location for an invalid date range
	//Create basic request
	//If duration is day then we can just grab the entire interval at once
	// Assuming the interval exists
	if duration == DAY {
		var message []byte

		// while the message is invalid, repeat till either the message becomes valid or we run out of the possible range
		for !valid {
			//Build the request
			req := k.histFormBuilder(curr.Format(DEFAULT_DATE_LAYOUT), final.Format(DEFAULT_DATE_LAYOUT), duration, exchangeToken)
			//Send the request
			resp, err := hc.Do(req)
			//Make sure there isn't a generic error
			if err != nil {
				log.Fatal(err)
			}

			//Read it
			message, err = ioutil.ReadAll(resp.Body)
			if err != nil {
				log.Fatal(err)
			}
			response, _ := jsonparser.GetString(message, "message")
			// if message is empty, we need to do it again, but lets try 1 month ahead as there is no data for current month
			if string(response) == "No candles found based on token and time and candleType." {

				curr = curr.Add(helper.MAX_TIME)
				fmt.Printf("Invalid Date range for %s. Trying to acquire from %s to %s \n", filename[0:len(filename)-4],
					curr.Format(DEFAULT_DATE_LAYOUT), final.Format(DEFAULT_DATE_LAYOUT))
				time.Sleep(time.Second * 1) // Need a 1 second timer between new requests to not break things

				if final.Sub(curr) < 0 {
					fmt.Println("Start date greater than end date!")
					os.Remove(filename)
					fmt.Printf("Error acquring %s from %s to %s , SKIPPING \n", filename[0:len(filename)-4], from, to)
					<-ch   // Race condition here but hopefully not a big deal. Prolly should fix at some point
					return // TODO: Change this to not os.Exit
				}
			} else {
				valid = true
			}

		}
		//Parse to get Candles
		data, _, _, err := jsonparser.Get(message, "data", "candles")
		if err != nil {
			log.Fatal(err)
		}
		//Format correctly
		data = helper.FormatData(string(data))

		//Store
		err = ioutil.WriteFile(filename, data, 0644)
		if err != nil {
			log.Fatal(err)
		}

	} else { //if Duration is not day then we need to split it into multiple days
		dataFile, _ := os.OpenFile(filename,
			os.O_WRONLY|os.O_CREATE, 0666)

		// start is i, increment is i to i + 30 days, stop when the difference between final and now is less than 28
		for curr = curr; final.Sub(curr) > 0; curr = helper.AddCurr(curr, final) {
			var message []byte
			i := 0
			// while the message is invalid, repeat till either the message becomes valid or we run out of the possible range
			// find the valid date range by checking 1 month periods till valid message is received
			for ; !valid; i++ {
				//Build the request
				req := k.histFormBuilder(curr.Format(DEFAULT_DATE_LAYOUT), curr.Add(helper.MAX_TIME).Format(DEFAULT_DATE_LAYOUT), duration, exchangeToken)
				//Send the request
				//fmt.Println(req)
				resp, err := hc.Do(req)
				//Make sure there isn't a generic error
				if err != nil {
					log.Fatal(err)
				}

				//Read it
				message, err = ioutil.ReadAll(resp.Body)

				if err != nil {
					log.Fatal(err)
				}
				//fmt.Println(string(message))
				response, _ := jsonparser.GetString(message, "message")
				// if message is empty, we need to do it again, but lets try 1 month ahead as there is no data for current month
				if string(response) == "No candles found based on token and time and candleType." {

					curr = curr.Add(helper.MAX_TIME)
					fmt.Printf("Invalid Date range for %s. Trying to acquire from %s to %s \n", filename[0:len(filename)-4],
						curr.Format(DEFAULT_DATE_LAYOUT), final.Format(DEFAULT_DATE_LAYOUT))
					time.Sleep(time.Second * 1) // Need a 1 second timer between new requests to not break things

					if final.Sub(curr) < 0 || i > 24 {
						//fmt.Println("Start date greater than end date!")
						dataFile.Close()
						os.Remove(filename)
						fmt.Printf("Error acquring %s from %s to %s , SKIPPING \n", filename[0:len(filename)-4], from, to)
						<-ch   // Race condition here but hopefully not a big deal. Prolly should fix at some point
						return // TODO: Change this to not os.Exit
					}

				} else if string(response) == "Invalid segment in instrument token" {
					log.Println("Use instrument tokens, the first column, you dumbass")
					<-ch
					return
				} else {
					fmt.Printf("VALID RANGE FOUND for %s \n", filename)
					valid = true
				}
			}

			//need to build request now that we're in the valid region
			if valid == true {
				var req *http.Request
				if final.Sub(curr.Add(helper.MAX_TIME)) < 0 {
					req = k.histFormBuilder(curr.Format(DEFAULT_DATE_LAYOUT), final.Format(DEFAULT_DATE_LAYOUT), duration, exchangeToken)

				} else {
					req = k.histFormBuilder(curr.Format(DEFAULT_DATE_LAYOUT), curr.Add(helper.MAX_TIME).Format(DEFAULT_DATE_LAYOUT), duration, exchangeToken)
				} //Send the request
				resp, err := hc.Do(req)
				//Make sure there isn't a generic error
				if err != nil {
					log.Fatal(err)
				}
				//Read it
				message, err = ioutil.ReadAll(resp.Body)
				// fmt.Println((string(message)))
				if err != nil {
					log.Fatal(err)
				}
				time.Sleep(time.Second * 1)
				data, _, _, err := jsonparser.Get(message, "data", "candles")

				data = helper.FormatData(string(data))

				_, err = dataFile.Write(data)
				//_, err = dataFile.Write([]byte("\n"))
				if err != nil {
					log.Fatal(err)
				}
				//fmt.Println("Looped")
				fmt.Printf("INPROGRESS: Finished acquring %s from %s to %s \n", filename[0:len(filename)-4], curr, curr.Add(helper.MAX_TIME))
				//fmt.Println(final.Sub(curr))
				// Write new line so next day begins at newline
				_, err = dataFile.Write([]byte(string('\n')))
			}

			if final.Sub(curr.Add(helper.MAX_TIME)) < 0 {
				break
			}

		}
		dataFile.Close()
		fmt.Printf("FINISHED acquring %s from %s to %s \n", filename[0:len(filename)-4], from, to)
		<-ch
	}
}
