package apiserver

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"path/filepath"
	"strings"
)

type VectorizerResult struct {
	Box    []uint32  `json:"box"`
	Vector []float64 `json:"vector"`
	Score  float64   `json:"score"`
}

func Vectorize(filename string, reader io.Reader, vectorizerUrl string) ([]VectorizerResult, error) {
	bodyBuf := &bytes.Buffer{}
	bodyWriter := multipart.NewWriter(bodyBuf)

	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`, "file", filepath.Base(filename)))
	switch e := strings.ToLower(filepath.Ext(filename)); e {
	case ".png":
		h.Set("Content-Type", "image/png")
	case ".jpg", ".jpeg":
		h.Set("Content-Type", "image/jpeg")
	default:
		return nil, errors.New(fmt.Sprintf("Invalid extension %s", e))
	}
	fileWriter, err := bodyWriter.CreatePart(h)

	if err != nil {
		fmt.Println("error writing to buffer")
		return nil, err
	}

	//iocopy
	_, err = io.Copy(fileWriter, reader)
	if err != nil {
		return nil, err
	}

	contentType := bodyWriter.FormDataContentType()
	bodyWriter.Close()

	resp, err := http.Post(vectorizerUrl, contentType, bodyBuf)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	resp_body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var result []VectorizerResult
	err = json.Unmarshal(resp_body, &result)
	if err != nil {
		return nil, err
	}

	return result, nil
}
