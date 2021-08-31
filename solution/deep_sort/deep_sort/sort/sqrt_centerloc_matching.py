# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def centerloc(bbox, candidates, img_width, img_height):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_cl = bbox[:2] + bbox[2:] / 2
    bbox_cl = bbox_cl / np.array([img_width, img_height])
    # bbox_cl = np.sqrt(bbox_cl)
    candidates_cl = candidates[:, :2] + candidates[:, 2:] / 2
    candidates_cl = bbox_cl / np.expand_dims(np.array([img_width, img_height]), 0)
    # candidates_cl = np.sqrt(candidates_cl)

    return np.array([1 - np.sqrt(np.linalg.norm(candidate - bbox_cl)) for candidate in candidates_cl])


def centerloc_cost(tracks, detections, img_width, img_height, track_indices=None,
             detection_indices=None):
    """An squared centerlocation distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        # candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - centerloc(bbox, candidates, img_width, img_height)
    return cost_matrix
