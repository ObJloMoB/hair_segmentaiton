import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch.nn.functional import softmax
import torchvision.transforms as transforms

from lib.model import SegModel

class Predicter:
    def __init__(self, model_path, img_size, force_cpu=False):
        self.device = 'cuda:0' if torch.cuda.is_available() and not force_cpu else 'cpu'
        self.model = SegModel().to(self.device).eval()
        self.model.load_state_dict(torch.load(model_path))
        self.size = img_size
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std= [0.229, 0.224, 0.225])])

    def preprocess_image(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        holder = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        if h > w:
            img = cv2.resize(img, (int(w/h*self.size), self.size))
            tl = (0, int((self.size-int(w/h*self.size))/2))

        else:
            img = cv2.resize(img, (self.size, int(h/w*self.size)))
            tl = (int((self.size-int(h/w*self.size))/2), 0)

        holder[tl[0]:tl[0]+img.shape[0],tl[1]:tl[1]+img.shape[1]] = img

        return holder, tl

    def postprocess_mask(self, tensor, tl):
        output = tensor.permute(0, 2, 3, 1).squeeze()
        output = output.cpu().numpy()
        mask = (output * 255).astype(np.uint8)
        mask = mask[tl[0]:self.size-tl[0], tl[1]:self.size-tl[1]]
        return mask

    def predict(self, image_path):
        img = cv2.imread(image_path)

        preprocessed, tl= self.preprocess_image(img)
        tensor = self.transforms(preprocessed)
        tensor = torch.unsqueeze(tensor, 0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        mask = self.postprocess_mask(output, tl)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        return mask


def main(opts):
    pred = Predicter(opts.weights, opts.size, opts.force_cpu)
    names = os.listdir(opts.input)
    names = [x for x in names if x.endswith('.jpg')]

    if not os.path.isdir(opts.output):
        os.makedirs(opts.output)

    for name in tqdm(names):
        m = pred.predict(os.path.join(opts.input, name))
        cv2.imwrite(os.path.join(opts.output, name), m)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input fld',
                        required=True, type=str)
    parser.add_argument('--output', help='save to fld',
                        required=True, type=str)
    parser.add_argument('--weights', help='DATA',
                        default='weights/checkpoint.pth', type=str)
    parser.add_argument('--size', help='Input image size',
                        default=224, type=int)
    parser.add_argument('--force_cpu', help='Use only cpu',
                        action="store_true")


    args = parser.parse_args()
    main(args)




        
