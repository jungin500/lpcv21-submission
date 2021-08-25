from yolox.data.datasets.coco_classes import COCO_CLASSES
from yolox.exps.base.build import get_exp
from yolox.utils import fuse_model
from yolox.predictor import Predictor
from yolox.utils.visualize import _COLORS
import torch
import cv2
import numpy as np
from sys import exit

VIDEO_PATH = 'rtsp://admin:geovision1!@172.24.90.168:554/cam/realmonitor?channel=1&subtype=0'
CHECKPOINT_PATH = 'yolox/weights/yolox_nano.pth'
EXP_PATH = 'yolox/exps/default/nano.py'

IMSHOW_TITLE = 'Video Output'

if __name__ == '__main__':
    device = torch.device('cpu')

    exp = get_exp(EXP_PATH, None)
    # exp.test_conf = 0.25
    # exp.nmsthre = 0.75
    # exp.test_size = (416, 416)
    model = exp.get_model()
    model.eval()

    # Load weight from checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'])

    # Fuse model for optimized inference
    model = fuse_model(model)

    # Create predictor from model
    predictor = Predictor(model, exp, COCO_CLASSES, None,  "cpu", False)

    # Prepare video reader
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Can't open video")
        exit(1)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.namedWindow(IMSHOW_TITLE, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(IMSHOW_TITLE, 1366, 768)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("EOF")
            break
        
        outputs, img_info = predictor.inference(frame)
        output = outputs[0]
        # result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)

        # Postprocessing - Get output bboxes
        cls_conf = predictor.confthre
        class_names = COCO_CLASSES

        ratio = img_info["ratio"]
        result_frame = img_info["raw_img"]
        output = output.cpu()
        bboxes = output[:, 0:4]

        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]


        for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(cls[i])
            score = scores[i]
            if score < cls_conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(result_frame, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                result_frame,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(result_frame, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        cv2.imshow(IMSHOW_TITLE, result_frame)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            raise StopIteration