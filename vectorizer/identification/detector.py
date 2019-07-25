import numpy as np
import torch
from PIL import Image

from torchvision import transforms


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=800, max_side=1400):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']

        rows, cols = image.size

        # scale = min_side / rows

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = np.array(image.resize((int(round((cols * scale))), int(round((rows * scale)))), resample=Image.BILINEAR))
        image = image  / 255.0

        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': new_image, 'annot': annots, 'scale': scale}


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots, scales = sample['img'], sample['annot'], sample['scale']

        image = (image.astype(np.float32) - self.mean) / self.std

        sample = {'img': torch.from_numpy(image), 'annot': torch.from_numpy(annots), 'scale': scales}
        return sample


def fan_detect(model, img_data, threshold=0.9, max_detections=100, is_cuda=True):
    input_data = {'img': img_data, 'annot': np.zeros((0, 5)), 'scale': 1}
    transform = transforms.Compose([Resizer(), Normalizer()])
    transformed = transform(input_data)

    model.eval()
    with torch.no_grad():
        img_data = transformed['img'].permute(2, 0, 1).float().unsqueeze(dim=0)
        if is_cuda:
            img_data = img_data.cuda()
        scores, labels, boxes = model(img_data)
        if scores is None:
            return np.array()

        scores = scores.cpu().numpy()
        scale = transformed['scale']
        boxes = boxes.cpu().numpy() / scale

        indices = np.where(scores > threshold)[0]
        scores = scores[indices]
        scores_sort = np.argsort(-scores)[:max_detections]
        image_boxes = boxes[indices[scores_sort], :]

    return image_boxes


def load_model(model_path, is_cuda=True):
    # load possible cuda model as cpu
    model = torch.load(model_path, map_location=lambda storage, location: storage)
    if is_cuda:
        model = model.cuda()

    model.anchors.is_cuda=is_cuda

    return model
