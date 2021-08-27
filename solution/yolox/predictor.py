import os
import cv2
import time
import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess, vis

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        device="cpu",
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device != 'cpu':
            img = img.cuda()# .half()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

    def inference_batched(self, imgs):
        img_infos = []

        imgs_batched = []
        for img in imgs:
            img_info = {"id": 0}
            img_info["file_name"] = None

            height, width = img.shape[:2]
            img_info["height"] = height
            img_info["width"] = width

            ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
            img_info["ratio"] = ratio

            img, _ = self.preproc(img, None, self.test_size)
            img = torch.from_numpy(img).unsqueeze(0)
            if self.device != 'cpu':
                img = img.cuda()# .half()

            imgs_batched.append(img)
            img_infos.append(img_info)

        imgs_batched = torch.cat(imgs_batched, 0)  # batch-dim concat

        with torch.no_grad():
            outputs_batched = self.model(imgs_batched)
            outputs_batched = postprocess(
                outputs_batched, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs_batched, img_infos

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res