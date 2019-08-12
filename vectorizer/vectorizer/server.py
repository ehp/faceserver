# -*- coding: utf-8 -*-
"""
   Copyright 2019 Petr Masopust, Aprar s.r.o.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import logging
import os
import sys
import tempfile

from flask import Flask, request, abort, jsonify
from werkzeug.utils import secure_filename

import torch
from recognition.nets import get_net_by_depth
from torchvision import transforms as T
from PIL import Image
import identification.detector as fan

is_cuda = torch.cuda.is_available()
print('CUDA: %s' % is_cuda)

fan_file = os.environ.get('VS_FAN_MODEL', None)
if fan_file is None:
    raise Exception('VS_FAN_MODEL is mandatory parameter')
fan_model = fan.load_model(fan_file, is_cuda=is_cuda)

# load recognition model
rec_model = get_net_by_depth(int(os.environ.get('VS_REC_DEPTH', 50)))
rec_file = os.environ.get('VS_REC_MODEL', None)
if rec_file is None:
    raise Exception('VS_REC_MODEL is mandatory parameter')
rec_model.load_state_dict(torch.load(rec_file, map_location=lambda storage, location: storage))
rec_model.eval()
if is_cuda:
    rec_model = rec_model.cuda()

# compute vectors
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

imagesize = 224
transforms = T.Compose([
    T.Resize((imagesize, imagesize)),
    T.ToTensor(),
    normalize
])

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def compute_vector(data):
    with torch.no_grad():
        data = transforms(data)
        if is_cuda:
            data = data.cuda()
        mo = rec_model(data.unsqueeze(dim=0))
        return mo.detach().cpu().numpy()


@app.route('/vectorize', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return 'OK'

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            abort(500)
        f = request.files['file']
        if f:
            filename = secure_filename(f.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            f.save(filepath)

            try:
                img = Image.open(filepath)
                data = img.convert(mode="RGB")

                with torch.no_grad():
                    boxes, scores = fan.fan_detect(fan_model, data, threshold=0.9, is_cuda=is_cuda)

                    if boxes is None or len(boxes) == 0:
                        return jsonify([])

                    boxes = boxes.astype(int)
                    scores = scores.astype(float)
                    extracted = [{'box': arr.tolist(),
                                  'vector': compute_vector(
                                      img.crop((arr[0], arr[1], arr[2], arr[3]))).squeeze().tolist(),
                                  'score': score
                                  } for arr, score in zip(boxes, scores)]
                    return jsonify(extracted)
            finally:
                os.remove(filepath)
        else:
            abort(500)


if __name__ == '__main__':
    logging.basicConfig()
    port = int(os.environ.get('VS_PORT', '8080'))
    app.run(host='0.0.0.0', debug=False, port=port)
