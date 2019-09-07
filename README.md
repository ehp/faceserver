# Face recognition technology demo

Mass faces identification and recognition in images. 

**You must have git lfs installed before cloning this repo !**

Because of github stupid git lfs pricing there is also [alternative mirror](https://gitea.ehp.cz/Aprar/faceserver) with copy of this repository. 

## Installation

The simplest complete installation is docker compose: ``docker-compose up -d`` in root directory. For detailed installation instructions look at [API server](apiserver/README.md) or [vectorizer](vectorizer/README.md) readme files.
Without nvidia docker support docker runs only on cpu with **very** degraded performance (over minute on 6 cpu cores).  

## Usage

### Learn people faces

```shell script
curl -X POST -F 'person=PID' -F 'directory=DIR' -F 'file=@portrait.jpg' http://localhost:8080/learn
```

Replace PID with person's id (e.g. database id or name) and DIR with your directory name (e.g. company name). People are recognized only within same directory.  png or jpeg images are supported. Only images with one face are allowed for learning !
Usually only one good portrait photo is enough but you can learn more photos for each person.

### Recognize people

```shell script
curl -X POST -F 'directory=DIR' -F 'file=@photo.jpg' http://localhost:8080/recognize
```

Replace DIR with your directory name (e.g. company name). People are recognized only within same directory. For each detected face the most probable person's id is returned. png or jpeg images are supported. 

Example result:

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

## Architecture

This demo consist of three parts - API server, vectorizer and database. API server is frontend server written in golang.
Vectorizer is the main part which identifies faces and creates vectors from faces. Database is simple storage for learned vectors.
Both API server and vectorizer are fully scalable e.g. in kubernetes. The only non scalable part is postgresql database but it can be easily replaced with different storage - e.g. HBase.
Just reimplement storage.go in API server.
Only API server listen to customer requests, the rest are internal components and should not be directly accessible from internet. 

## Future roadmap

 * Training on identified faces (both nets are trained separately now)
 * Face alignment between identification and recognition
 * Web user interface (help needed !)

## FAQ

* Unable to connect to vectorizer or _pickle.UnpicklingError: invalid load key, 'v'.  
Missing model data - probably cloned repository without git lfs.
To download and use pretrained models in already cloned repository:
```shell script
git lfs install
git lfs checkout
git lfs fetch

docker-compose build
``` 

## Based on

Github repositories:

* [https://github.com/rainofmine/Face_Attention_Network](https://github.com/rainofmine/Face_Attention_Network)
* [https://github.com/ronghuaiyang/arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)

Papers:

* [Face Attention Network: An Effective Face Detector for the Occluded Faces](https://arxiv.org/abs/1711.07246)
* [AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations](https://arxiv.org/abs/1905.00292)
* [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
* [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)
* [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)
* [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896)

## Licensing

Code in this repository is licensed under the Apache 2.0. See [LICENSE](LICENSE).
