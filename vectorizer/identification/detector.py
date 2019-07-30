import numpy as np
import torch
import argparse
import json
from PIL import Image, ImageDraw

from identification.dataloader import Normalizer, Resizer
from torchvision import transforms


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
            return np.empty((0,0)), np.empty((0,0))

        scores = scores.cpu().numpy()
        scale = transformed['scale']
        boxes = boxes.cpu().numpy() / scale

        indices = np.where(scores > threshold)[0]
        scores = scores[indices]
        scores_sort = np.argsort(-scores)[:max_detections]
        image_boxes = boxes[indices[scores_sort], :]

    return image_boxes, scores[:max_detections]


def img_rectangles(img, output_path, boxes=None):
    if boxes is not None:
        draw = ImageDraw.Draw(img)
        for arr in boxes:
            draw.rectangle(((arr[0], arr[1]), (arr[2], arr[3])), outline="black", width=1)

    img.save(output_path)


def load_model(model_path, is_cuda=True):
    # load possible cuda model as cpu
    model = torch.load(model_path, map_location=lambda storage, location: storage)
    if is_cuda:
        model = model.cuda()

    model.anchors.is_cuda=is_cuda

    return model


def load_image(filepath):
    img = Image.open(filepath)
    img = img.convert(mode="RGB")
    return img


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--model', help='Path to model')
    parser.add_argument('--image', help='Path to image')
    parser.add_argument('--rect', help='Output image with rectangles')
    parser.add_argument('--threshold', help='Probability threshold (default 0.9)', type=float, default=0.9)
    parser.add_argument('--force-cpu', help='Force CPU for detection (default false)', dest='force_cpu',
                        default=False, action='store_true')

    parser = parser.parse_args(args)

    is_cuda = torch.cuda.is_available() and not parser.force_cpu

    model = load_model(parser.model, is_cuda=is_cuda)
    img = load_image(parser.image)
    boxes, scores = fan_detect(model, img, threshold=parser.threshold, is_cuda=is_cuda)
    print(json.dumps({'boxes': boxes.tolist(), 'scores': scores}))
    if parser.rect:
        img = load_image(parser.image)
        img_rectangles(img, parser.rect, boxes)


if __name__ == '__main__':
    main()
