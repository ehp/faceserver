# Vectorizer 
  
Heart of faceserver app. **Technology demo - do not use in production !**

**Main purpose:**    
* find faces in image
* create vector from face

No local state, can be scaled. GPU is **highly** recommended.

## Configuration

Set environment variables:

| Key | Default value | Description |
| --- | --- | --- |
| VS_PORT | 8080 | Port to listen (for Flask) |
| VS_FAN_MODEL |  | Path to identification model |
| VS_REC_NET | resnet50 | Recognition net name |
| VS_REC_MODEL |  | Path to recognition model |

Do not change configuration if you want run prepared docker-compose.

## Instalation

### Docker image

Build docker image - preferred method if you have nvidia-docker:
```shell script
docker build -t vectorizer:latest .
```

### Local installation

Install PIP dependencies (virtualenv recommended):

```shell script
pip install --upgrade -r requirements.txt
```

And then run server:

```shell script
python3 -m vectorizer.server
```

## HTTP API

### Vectorization

```shell script
curl -X POST -F 'file=@image.jpg' http://localhost:8080/vectorize
```

png or jpeg images are supported.

Result:

```json
[
{"box":[0,15,65,88],"vector":[-0.14234,...,0.32432],"score":0.9909800887107849}
]
```

| Field | Description |
| --- | --- |
| box | Box around face |
| vector | Array of 512 floats |
| score | Face detection score (i.e. probability) |

## Training

**GPU is mandatory for training !** 
Training takes at least several days to achieve reasonable accuracy on single RTX 2070.
Trained models are stored in ``ckpt`` directory. Pretrained models with example parameters are included.

### Identification

Example:

```shell script
python3 -m identification.train --wider_train ~/datasets/wider/wider_face_train_bbx_gt.txt \
--wider_train_prefix ~/datasets/wider/WIDER_train/images \
--wider_val ~/datasets/wider/wider_face_val_bbx_gt.txt \
--wider_val_prefix ~/datasets/wider/WIDER_val/images \
--depth 50 --epochs 30 --batch_size 1 --model_name wider1
```

| Argument | Description | Required / Default value |
| --- | --- | --- |
| --wider_train | Path to file containing WIDER training annotations (wider_face_train_bbx_gt.txt) | Yes |
| --wider_val | Path to file containing WIDER validation annotations (wider_face_val_bbx_gt.txt) |  |
| --wider_train_prefix | Prefix path to WIDER train images | Yes |
| --wider_val_prefix | Prefix path to WIDER validation images |  |
| --depth | Resnet depth, must be one of 18, 34, 50, 101, 152 | 50 |
| --epochs | Number of epochs | 50 |
| --batch_size | Batch size - increase only if you have enough GPU memory (i.e. >16 GB) ! | 2 |
| --model_name | Model name prefix | Yes |
| --parallel | Run training with DataParallel | false |
| --pretrained | Pretrained model (e.g. for crash recovery) |  |

There is also option to train from csv files - see train.py and dataloader.py for details.

### Recognition

Example:

```shell script
python3 -m recognition.train \
--casia_list ~/datasets/CASIA-maxpy-clean/train.txt \
--casia_root ~/datasets/CASIA-maxpy-clean \
--lfw_root ~/datasets/lfw \
--lfw_pair_list lfw_test_pair.txt \
--model_name recongition1 --batch_size 20 \
--loss adacos --print_freq 20 --net resnet50
```

| Argument | Description | Required / Default value |
| --- | --- | --- |
| --casia_list | Path to CASIA dataset file list (train.txt) | Yes |
| --casia_root | Path to CASIA images | Yes |
| --lfw_root | Path to LFW dataset | Yes |
| --lfw_pair_list | Path to LFW pair list file (lfw_test_pair.txt - in this repository) | Yes |
| --net | Net name, must be one of resnet18, resnet34, resnet50, resnet101, resnet152, resnext50, resnext101 or spherenet | resnet50 |
| --epochs | Number of epochs | 50 |
| --batch_size | Batch size | 16 |
| --model_name | Model name prefix | Yes |
| --parallel | Run training with DataParallel | false |
| --loss | One of focal_loss. cross_entropy, arcface, cosface, sphereface, adacos | cross_entropy |
| --optimizer | One of sgd, adam | sgd |
| --weight_decay | Weight decay | 0.0005 |
| --lr | Learning rate | 0.1 |
| --lr_step | Learning rate step | 10 |
| --easy_margin | Use easy margin | false |
| --print_freq | Print every N batch | 100 |

## Datasets for training

* [WIDER](http://shuoyang1213.me/WIDERFACE/)
* [LFW](http://vis-www.cs.umass.edu/lfw/)
* CASIA maxpy clean - no official web but can be downloaded from suspicious sites (use google)

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

## Licensing

Code in this repository is licensed under the Apache 2.0. See [LICENSE](../LICENSE).
