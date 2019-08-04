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
    viper.SetEnvPrefix("AS_")
    viper.AutomaticEnv()
    err := viper.ReadInConfig() // Find and read the config file
    if err != nil { // Handle errors reading the config file
        panic(fmt.Errorf("Fatal error config file: %s \n", err))
    }

    apiserver.Dbo, err = apiserver.NewStorage(viper.GetString("db.user"), viper.GetString("db.password"), viper.GetString("db.name"), viper.GetString("db.host"))
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
