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

package main

import (
    "fmt"
    "log"
    "net/http"
    "strconv"

    "gitea.ehp.cz/Aprar/faceserver/apiserver"
    "github.com/spf13/viper"
)

func main() {
    viper.SetConfigName("apiserver") // name of config file (without extension)
    viper.AddConfigPath("/etc/faceserver/")   // path to look for the config file in
    viper.AddConfigPath("$HOME/.faceserver")  // call multiple times to add many search paths
    viper.AddConfigPath(".")               // optionally look for config in the working directory
    err := viper.ReadInConfig() // Find and read the config file
    if err != nil { // Handle errors reading the config file
        panic(fmt.Errorf("Fatal error config file: %s \n", err))
    }

    apiserver.Dbo, err = apiserver.NewStorage(viper.GetString("dbuser"), viper.GetString("dbpassword"), viper.GetString("dbname"), viper.GetString("dbhost"))
    if err != nil {
        panic(fmt.Errorf("Fatal error database connection: %s \n", err))
    }

    http.Handle("/", http.FileServer(http.Dir("./public")))

    fs := http.FileServer(http.Dir("./files"))
    http.Handle("/files/", http.StripPrefix("/files", fs))

    http.HandleFunc("/learn", apiserver.Learn)
    http.HandleFunc("/recognize", apiserver.Recognize)
    log.Println("Running")
    log.Fatal(http.ListenAndServe(":" + strconv.Itoa(viper.GetInt("port")), nil))
}
