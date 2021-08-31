from deep_sort.deep_sort.deep.feature_extractor import Extractor
from deep_sort.deep_sort.sort.preprocessing import non_max_suppression

import numpy as np
import torch

import time


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature, clses, ):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        #self.clses = np.asarray(clses, dtype=np.int)
        self.clses = int(clses)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def __repr__(self):
        return 'Detection[tlwh=%s confidence=%.2f feature=%s clses=%d' % (
            str(self.tlwh), self.confidence, self.feature.shape, self.clses
        )


class CenterBasedTracker:
    NEXT_TRACKER_ID = 1

    def __init__(self, distance_threshold: float):
        self.tracks = []
        self.distance_threshold = distance_threshold

    def update(self, detections: list[Detection]):
        # Match with existing tracks
        matched_tracks, unmatched_detections, unmatched_tracks = self.match_tracks(
            detections)

        # Add current detections to tracks
        for detection_id, track_id in matched_tracks:
            self.tracks[track_id].update(detections[detection_id])

        # Create new track for unmatched detections
        print('Detections:', detections)
        print('unmatched_detections:', unmatched_detections)
        for detection_id in unmatched_detections:
            detection = detections[detection_id]
            self.tracks.append(Track(self.next_id(), detection))

    def match_tracks(self, detections: list[Detection]):
        if not self.tracks:
            return [], list(range(len(detections))), []

        # Match exisitng tracks with detections
        matched_tracks = []
        for detection_id, detection in enumerate(detections):
            # Calculate distance with existing tracks
            distances = []
            for track_id, track in enumerate(self.tracks):
                distance = track.distance_to(detection)
                distances.append((track_id, distance))

            # Find the closest track
            distances.sort()
            shortest_track_id, shortest_track_distance = distances[0]

            # Filter if distance is too far
            if shortest_track_distance <= self.distance_threshold:
                matched_tracks.append((detection_id, shortest_track_id))

        unmatched_detections = [i for i in range(len(detections)) if i not in [
            m[0] for m in matched_tracks]]
        unmatched_tracks = [t.id for t in self.tracks if t.id not in [
            m[1] for m in matched_tracks]]
        return matched_tracks, unmatched_detections, unmatched_tracks

    def next_id(self):
        next_tid = CenterBasedTracker.NEXT_TRACKER_ID
        CenterBasedTracker.NEXT_TRACKER_ID += 1
        return next_tid


class Track:
    def __init__(self, id: int, detection: Detection):
        self.id = id
        self.start_time = time.time()
        self.update(detection)

    def update(self, detection: Detection):
        self.clses = detection.clses
        self.feature = detection.feature
        self.tlwh = detection.tlwh
        self.bbox_center = detection.tlwh[:2] + detection.tlwh[2:] / 2

    def distance_to(self, detection: Detection):
        # Weighted distance of (Detection feature) and (L2 distance of bbox centers)
        detection_bbox_center = detection.tlwh[:2] + detection.tlwh[2:] / 2
        l2_bbox_center = np.linalg.norm(np.sqrt(detection_bbox_center) - np.sqrt(self.bbox_center))
        l2_features = np.linalg.norm(detection.feature - self.feature)
        weighted_distance = l2_bbox_center + l2_features
        print("l2_bbox_center Norm:", l2_bbox_center,
              ", l2_features Norm:", l2_features)
        return weighted_distance

    def to_tlwh(self):
        return self.tlwh

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

class CenterBasedBallPersonTracker:
    def __init__(self, model_path, ball_tracks=5, person_tracks=7):
        self.ball_tracks = ball_tracks
        self.person_tracks = person_tracks
        self.extractor = Extractor(model_path, False)
        self.min_confidence = 0.4
        self.nms_max_overlap = 0.8
        self.tracker = CenterBasedTracker(
            distance_threshold=1.0)  # disable threshold

    def update(self, bbox_xywh, confidences, clses, original_image):
        self.height, self.width, _ = original_image.shape

        # Generate features
        crops = []
        for bbox in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(bbox)
            cropped_image = original_image[y1:y2, x1:x2]
            crops.append(cropped_image)
        features = self.extractor(crops)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i], clses[i])
                      for i, conf in enumerate(confidences) if conf > self.min_confidence]

        # Do NMS over classes
        detections = non_max_suppression(detections, self.nms_max_overlap)

        # print("CenterBasedBallPersonTracker: ", detections)
        self.tracker.update(detections)

        outputs = []
        current_time = time.time()
        for _, track in enumerate(self.tracker.tracks):
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.id
            cls2 = track.clses
            score2 = 100.0  # always 100
            start_time = track.start_time
            stay = int(current_time - start_time)
            outputs.append(
                np.array([x1, y1, x2, y2, track_id, cls2, score2, stay], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x-w/2), 0)
        x2 = min(int(x+w/2), self.width-1)
        y1 = max(int(y-h/2), 0)
        y2 = min(int(y+h/2), self.height-1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2]/2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3]/2.
        return bbox_tlwh
