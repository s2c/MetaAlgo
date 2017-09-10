package kiteGo

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"github.com/buger/jsonparser"
	"io/ioutil"
	"kite-go/helper"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

const (
	API_ROOT                string        = "https://api.kite.trade"
	TOKEN_URL               string        = "/session/token"
	PARAMETERS              string        = "/parameters"
	USER_MARGINS            string        = "/user/margins/"
	ORDERS                  string        = "/orders"
	TRADES                  string        = "/trades"
	HISTORICAL              string        = "/instruments/historical/"
	API_FIELD               string        = "API_KEY"
	SECRET_FIELD            string        = "API_SECRET"
	REQ_TOK_FIELD           string        = "REQ_TOKEN"
	MINUTE                  string        = "minute"
	THREE_MIN               string        = "3minute"
	FIVE_MIN                string        = "5minute"
	TEN_MIN                 string        = "10minute"
	FIFTEEN_MIN             string        = "15minute"
	THIRTY_MIN              string        = "30minute"
	SIXTY_MIN               string        = "60minute"
	DAY                     string        = "day"
	EX_CDS                  string        = "CDS"
	EX_NFO                  string        = "NFO"
	EX_BSE                  string        = "BSE"
	EX_BFO                  string        = "BFO"
	EX_NSE                  string        = "NSE"
	EX_MCX                  string        = "MCX"
	POST                    string        = "POST"
	GET                     string        = "GET"
	DEFAULT_TIMEOUT         time.Duration = 7
	DEFAULT_FULLTIME_LAYOUT string        = "2006-01-02T15:04:05-0700"
	DEFAULT_DATE_LAYOUT     string        = "2006-01-02"
	ACC_TOKEN_FILE          string        = "ACCTOKEN.txt"
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
		helper.CheckError(err)
	}

	k.Client_API_SECRET, _ = jsonparser.GetString(data, SECRET_FIELD)
	k.Client_API_KEY, _ = jsonparser.GetString(data, API_FIELD)
	k.Client_REQ_TOKEN, _ = jsonparser.GetString(data, REQ_TOK_FIELD)

	//FIle Read, now need to check if we still need to login today.

	// Check if AccessToken is still valid
	validAcc := helper.AccessTokenValidity(ACC_TOKEN_FILE)
	if validAcc {
		contByte, err := ioutil.ReadFile(ACC_TOKEN_FILE)
		contString := string(contByte)
		if err != nil {
			k.SetAccessToken(contString) // If yes then just read and set Access Token
		}
	} else {
		k.Login()
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
func (k *kiteClient) Login() {
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
	helper.CheckError(err)

	//Do the request
	resp, err := hc.Do(req)
	helper.CheckError(err, resp)

	//Read the response
	message, err := (ioutil.ReadAll(resp.Body))
	helper.CheckError(err)

	//parse accesstoken and store
	accToken, _, _, err := jsonparser.Get(message, "data", "access_token")
	helper.CheckError(err)
	k.SetAccessToken(string(accToken))
	if k.Client_ACC_TOKEN != "" {

		fmt.Println("Access Key set")
	}

	//parse publictoken and store
	pubToken, _, _, err := jsonparser.Get(message, "data", "public_token")
	helper.CheckError(err)
	k.SetPublicToken(string(pubToken))
	if k.Client_PUB_TOKEN != "" {
		fmt.Println("Public Key set")
	}

}

//dates of format yyyy-mm-dd
//concurrent safe as long as a million copies are not called, I THINK
func (k *kiteClient) GetHistorical(duration string, exchangeToken string, from string, to string, filename string, ch chan bool) {
	ch <- true
	fmt.Println("WE HERE")
	//Create HTTP client
	fmt.Printf("Starting to acquire %s from %s to %s \n", filename[0:len(filename)-4], from, to)
	hc := helper.HttpClient(30)
	//Create basic request
	//If duration is day then we can just grab the entire interval at once
	if duration == DAY {

		req, err := http.NewRequest(GET, API_ROOT+HISTORICAL+exchangeToken+"/"+duration, nil)
		helper.CheckError(err)
		//Add parameters to request
		form := req.URL.Query()
		form.Add("api_key", k.Client_API_KEY)
		form.Add("access_token", k.Client_ACC_TOKEN)
		form.Add("from", from)
		form.Add("to", to)
		req.URL.RawQuery = form.Encode()
		//Create new request with parameters
		req, err = http.NewRequest(GET, req.URL.String(), nil)
		helper.CheckError(err)
		//Get the historical data
		resp, err := hc.Do(req)
		helper.CheckError(err, resp)

		//Read it
		message, err := ioutil.ReadAll(resp.Body)
		helper.CheckError(err)
		//	fmt.Println(string(message))
		//Parse to get Candles
		data, _, _, err := jsonparser.Get(message, "data", "candles")
		helper.CheckError(err)

		//Format correctly
		data = helper.FormatData(string(data))

		//Store
		err = ioutil.WriteFile("data/"+filename, data, 0644)
		helper.CheckError(err)
	} else { //if Duration is not day then we need to split it into multiple days
		dataFile, _ := os.OpenFile("data/"+filename,
			os.O_WRONLY|os.O_CREATE, 0666)
		curr, _ := time.Parse(DEFAULT_DATE_LAYOUT, from)
		final, _ := time.Parse(DEFAULT_DATE_LAYOUT, to)
		// start is i, increment is i to i + 30 days, stop when the difference between final and now is less than 28
		for curr = curr.Add(-1 * 24 * time.Hour); final.Sub(curr) > 24*29*time.Hour; curr = curr.Add(24 * 29 * time.Hour) {
			curr = curr.Add(24 * time.Hour)
			req, err := http.NewRequest(GET, API_ROOT+HISTORICAL+exchangeToken+"/"+duration, nil)
			helper.CheckError(err)
			//Add parameters to request
			form := req.URL.Query()
			form.Add("api_key", k.Client_API_KEY)
			form.Add("access_token", k.Client_ACC_TOKEN)
			form.Add("from", curr.Format(DEFAULT_DATE_LAYOUT))
			form.Add("to", (curr.Add(29 * 24 * time.Hour)).Format(DEFAULT_DATE_LAYOUT))
			req.URL.RawQuery = form.Encode()
			//fmt.Println(req.URL.String())
			req, err = http.NewRequest(GET, req.URL.String(), nil)
			helper.CheckError(err)
			//Get the historical data
			resp, err := hc.Do(req)
			time.Sleep(1 * time.Second)
			helper.CheckError(err, resp)
			message, err := ioutil.ReadAll(resp.Body)
			helper.CheckError(err)
			//	fmt.Println(string(message))
			//Parse to get Candles
			data, _, _, err := jsonparser.Get(message, "data", "candles")
			helper.CheckError(err)
			data = helper.FormatData(string(data))
			_, err = dataFile.Write(data)
			_, err = dataFile.Write([]byte("\n"))
			helper.CheckError(err)
			//fmt.Println("Looped")

		}
		//TODO: Fix this so there is only the for loop instead of doing it twice
		//repeat for last remaining bit
		req, err := http.NewRequest(GET, API_ROOT+HISTORICAL+exchangeToken+"/"+duration, nil)
		helper.CheckError(err)
		form := req.URL.Query()
		form.Add("api_key", k.Client_API_KEY)
		form.Add("access_token", k.Client_ACC_TOKEN)
		form.Add("from", curr.Format(DEFAULT_DATE_LAYOUT))
		form.Add("to", final.Format(DEFAULT_DATE_LAYOUT))
		req.URL.RawQuery = form.Encode()
		//	fmt.Println(req.URL.String())
		req, err = http.NewRequest(GET, req.URL.String(), nil)
		helper.CheckError(err)
		//Get the historical data
		resp, err := hc.Do(req)
		helper.CheckError(err, resp)
		message, err := ioutil.ReadAll(resp.Body)
		helper.CheckError(err)
		//	fmt.Println(string(message))
		//Parse to get Candles
		data, _, _, err := jsonparser.Get(message, "data", "candles")
		helper.CheckError(err)
		data = helper.FormatData(string(data))
		_, err = dataFile.Write(data)
		helper.CheckError(err)
		dataFile.Close()
		fmt.Printf("FINISHED acquring %s from %s to %s \n", filename[0:len(filename)-4], from, to)
		<-ch

	}

}
