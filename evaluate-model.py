import cv2
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'solution'))
sys.path.insert(0, os.path.join(os.getcwd(), 'solution', 'yolox'))
from yolox.exps.base.build import get_exp
from yolox.predictor import Predictor
import torch

def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

if __name__ == '__main__':
    args = sys.argv

    video_path = args[1]
    gt_path = args[2]
    
    EXP_PATH = 'solution/yolox/exps/custom/model_yj.py'
    CHECKPOINT_PATH = 'solution/yolox/weights/best_ckpt.pth'
    BALLPERSON_CLASSES = (
        "person",
        "sports ball"
    )

    device = torch.device('cpu')

    exp = get_exp(EXP_PATH, None)
    exp.test_size = (384, 640)
    exp.test_conf = 0.35
    exp.nmsthre = 0.35
    model = exp.get_model().eval()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)['model'])

    predictor = Predictor(model, exp, BALLPERSON_CLASSES, device.type, False)

    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("EOF")
            break
        
        preds, img_infos = predictor.inference_batched([frame])

        det = preds[0]
        img_info = img_infos[0]

        cls_conf = predictor.confthre
        ratio = img_info['ratio']

        xywhs = []
        confs = []
        clses = []

        bboxes_all = det[:, :4] / ratio
        conf_all = det[:, 4:5] * det[:, 5:6]
        clses_all = det[:, 6:7]

        # Write results
        for bbox_id in range(bboxes_all.shape[0]):
            x0, y0, x1, y1 = bboxes_all[bbox_id]
            conf = conf_all[bbox_id]
            cls = clses_all[bbox_id]

            if conf < cls_conf:
                continue

            img_h, img_w, _ = frame.shape  # get image shape
            x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *[x0, y0, x1, y1])
            obj = [x_c, y_c, bbox_w, bbox_h]

            xywhs.append(obj)
            confs.append([conf.item()])
            clses.append([cls.item()])
        
        xywhs = torch.Tensor(xywhs)
        confs = torch.Tensor(confs)
        clses = torch.Tensor(clses)

        colors = [
            (255, 0, 0),
            (0, 255, 0)
        ]

        for i in range(xywhs.shape[0]):
            xc, yc, w, h = xywhs[i]
            conf = confs[i].item()
            cls = clses[i].int().item()

            x1, x2, y1, y2 = xc - (w / 2), xc + (w / 2), yc - (h / 2), yc + (h / 2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls], 2)

        cv2.imshow("Model Output", cv2.resize(frame, (1920, 1080)))
        chr = cv2.waitKey(1)
        if chr == ord('q'):
            break