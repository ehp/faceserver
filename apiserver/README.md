# API server  
  
Frontend server written in golang. **Technology demo - do not use in production !**

**Main purpose:**    
* serve stored images  
* send images to vectorizer  
* store vectors in database  
* compare vectors and return ids

No local state, can be scaled.

## Configuration
Edit ``apiserver.yaml`` file:

| Key | Value | Description |
| --- | --- | --- |
| port | 8080 | Port to listen |
| vectorizer | http://vectorizer:8080/vectorize | Vectorizer url |
| dbuser | faceserver | DB user |
| dbpassword | secret | DB password |
| dbname | faceserver | DB name |
| dbhost | db | DB host |

Do not change configuration if you want run prepared docker-compose.

### DB configuration
Only postgresql is supported now. Create new role and user:
```shell script
createuser -D -P -S faceserver
createdb -E UTF8 -O faceserver faceserver
``` 

Create API server tables:

```shell script
psql -U faceserver -h localhost faceserver <../init.sql
```

## Instalation
### Docker image
Build docker image - preferred method:

```shell script
docker build -t apiserver:latest .
```

### Local compilation
Golang 1.12 is required. Run:

```shell script
go build main.go
```

## HTTP API
### Learn

```shell script
curl -X POST -F 'person=PID' -F 'directory=DIR' -F 'file=@portrait.jpg' http://localhost:8080/learn
```

Replace PID with person's id (e.g. database id or name) and DIR with your directory name (e.g. company name). People are recognized only within same directory.  png or jpeg images are supported. Only images with one face are allowed for learning !

Result:

```json
{
"status":"OK",
"url":"/files/01e66d8f-536e-4e5ab3b1-521672739d15.jpg",
"filename":"photo.jpg",
"directory":"mydir",
"persons":[
{"id":"PID","box":[0,15,65,88],"score":0.9909800887107849}
]}
```

|Field|Description|
|--|--|
|status|Status message - either OK or error text|
|url|Relative url to original image|
|filename|Original image filename|
|directory|Directory name|
|persons|Recognized people array|
|id|Person's id|
|box|Box around face|
|score|Face detection score (i.e. probability)|

### Recognize

```shell script
curl -X POST -F 'directory=DIR' -F 'file=@photo.jpg' http://localhost:8080/recognize
```

Replace DIR with your directory name (e.g. company name). People are recognized only within same directory. For each detected face the most probable person's id is returned. png or jpeg images are supported. 

Result:

```json
{
"status":"OK",
"url":"/files/00636b47-e6a5-4fab-8a02-9e44d052c193.jpg",
"filename":"photo.jpg",
"directory":"mydir",
"persons":[
{"id":"PID1","box":[2797,1164,2918,1285],"score":0.999998927116394,"probability":0.8342},
{"id":"PID2","box":[2398,1854,2590,2046],"score":0.9999780654907227,"probability":0.32546},
{"id":"PID3","box":[1753,1148,1905,1300],"score":0.9999217987060547,"probability":0.65785}
]}
```

| Field | Description |
| --- | --- |
| status | Status message - either OK or error text |
| url | Relative url to original image |
| filename | Original image filename |
| directory | Directory name |
| persons | Recognized people array |
| id | Person's id |
| box | Box around face |
| score | Face detection score (i.e. probability) |
| probability | Person recognition probability |

### Files

``/files/...`` path contains all learned or recognized images.

## Licensing

Code in this repository is licensed under the Apache 2.0. See [LICENSE](../LICENSE).
