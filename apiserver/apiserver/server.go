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
	"encoding/json"
	"errors"
	"fmt"
	"github.com/google/uuid"
	"github.com/spf13/viper"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
)

var Dbo PgStorage

type JsonPerson struct {
	Id          string   `json:"id,omitempty"`
	Box         []uint32 `json:"box,omitempty"`
	Score       float64  `json:"score,omitempty"`
	Probability float64  `json:"probability,omitempty"`
}

type JsonResponse struct {
	Status    string       `json:"status,omitempty"`
	Url       string       `json:"url,omitempty"`
	Filename  string       `json:"filename,omitempty"`
	Directory string       `json:"directory,omitempty"`
	Persons   []JsonPerson `json:"persons"`
}

func sendError(w http.ResponseWriter, err error) {
	log.Printf("%v\n", err)
	jsonResponse(w, 400, JsonResponse{
		Status: err.Error(),
	})
}

func Learn(w http.ResponseWriter, r *http.Request) {
	filename, uid, result, err := uploadSave(w, r)
	if err != nil {
		sendError(w, err)
		return
	}

	if len(result) != 1 {
		sendError(w, errors.New("More than one face detected."))
		return
	}

	pid := r.FormValue("person")
	if pid == "" {
		sendError(w, errors.New("Person identification is required."))
		return
	}
	directory := r.FormValue("directory")
	if directory == "" {
		sendError(w, errors.New("Directory is required."))
		return
	}

	person := Person{
		Id:          pid,
		Directory:   directory,
		Filename:    filename,
		FilenameUid: uid,
		Score:       result[0].Score,
		Box:         result[0].Box,
		Vector:      result[0].Vector,
	}
	err = Dbo.Store(person)
	if err != nil {
		sendError(w, err)
		return
	}

	jsonResponse(w, http.StatusCreated, JsonResponse{
		Status:    "OK",
		Url:       "/files/" + uid,
		Filename:  filename,
		Directory: directory,
		Persons: []JsonPerson{{
			Id:    pid,
			Box:   person.Box,
			Score: person.Score,
		}},
	})
}

func Recognize(w http.ResponseWriter, r *http.Request) {
	filename, uid, result, err := uploadSave(w, r)
	if err != nil {
		sendError(w, err)
		return
	}

	directory := r.FormValue("directory")
	if directory == "" {
		sendError(w, errors.New("Directory is required."))
		return
	}

	persons, err := Dbo.GetDirectory(directory)
	if err != nil {
		sendError(w, err)
		return
	}

	jp := []JsonPerson{}
	for _, r := range result {
		maxprob := -1.0
		var maxperson Person
		for _, p := range persons {
			cm := CosinMetric(r.Vector, p.Vector)
			if cm > maxprob {
				maxprob = cm
				maxperson = p
			}
		}
		jp = append(jp, JsonPerson{
			Id:          maxperson.Id,
			Box:         r.Box,
			Score:       r.Score,
			Probability: maxprob,
		})
	}

	jsonResponse(w, http.StatusCreated, JsonResponse{
		Status:    "OK",
		Url:       "/files/" + uid,
		Filename:  filename,
		Directory: directory,
		Persons:   jp,
	})
}

func uploadSave(w http.ResponseWriter, r *http.Request) (string, string, []VectorizerResult, error) {
	if err := checkMethod(w, r); err != nil {
		return "", "", nil, err
	}

	if err := r.ParseMultipartForm(32 << 20); err != nil {
		return "", "", nil, err
	}

	file, handle, err := r.FormFile("file")
	if err != nil {
		return "", "", nil, err
	}
	defer file.Close()

	mimeType := handle.Header.Get("Content-Type")
	if err := checkFileType(mimeType); err != nil {
		return "", "", nil, err
	}

	uid, err := saveFile(w, file, handle)
	if err != nil {
		return "", "", nil, err
	}

	reader, err := os.Open("./files/" + uid)
	if err != nil {
		return "", "", nil, err
	}
	defer reader.Close()
	results, err := Vectorize(uid, reader, viper.GetString("vectorizer"))
	if err != nil {
		return "", "", nil, err
	}

	return handle.Filename, uid, results, nil
}

func checkMethod(w http.ResponseWriter, r *http.Request) error {
	if r.Method != http.MethodPost {
		return errors.New("POST method required")
	}

	return nil
}

func checkFileType(mimeType string) error {
	switch mimeType {
	case "image/jpeg", "image/png":
		return nil
	default:
		return errors.New(fmt.Sprintf("Invalid file format %s", mimeType))
	}
}

func generateFilename(filename string) string {
	e := filepath.Ext(filename)
	uid := uuid.New().String()
	return uid + e
}

func saveFile(w http.ResponseWriter, file multipart.File, handle *multipart.FileHeader) (string, error) {
	uid := generateFilename(handle.Filename)
	f, err := os.OpenFile("./files/"+uid, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		return "", err
	}
	defer f.Close()

	_, err = io.Copy(f, file)
	if err != nil {
		return "", err
	}

	return uid, nil
}

func jsonResponse(w http.ResponseWriter, code int, message JsonResponse) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	resp, err := json.Marshal(message)
	if err != nil {
		log.Fatalf("Cannot format %v", message)
	}
	w.Write(resp)
}
