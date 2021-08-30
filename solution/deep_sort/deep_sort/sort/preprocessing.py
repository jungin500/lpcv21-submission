# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2


def non_max_suppression(detections, max_bbox_overlap):
    clses = [d.clses for d in detections]
    clses_list = np.unique(clses)

    picked_detections = []
    for clsid in clses_list:
        class_detections = list(filter(lambda d: d.clses == clsid, detections))

        boxes = np.array([d.tlwh for d in class_detections])
        scores = np.array([d.confidence for d in class_detections])
    
        if len(boxes) == 0:
            continue

        boxes = boxes.astype(np.float)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2] + boxes[:, 0]
        y2 = boxes[:, 3] + boxes[:, 1]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if scores is not None:
            idxs = np.argsort(scores)
        else:
            idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            picked_detections.append(class_detections[i])

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(
                    ([last], np.where(overlap > max_bbox_overlap)[0])))

    return picked_detections
