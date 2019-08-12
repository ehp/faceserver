// Copyright 2019 Petr Masopust, Aprar s.r.o.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
