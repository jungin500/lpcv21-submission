# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from . import sqrt_centerloc_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, img_width, img_height, max_iou_distance=0.7, max_age=70, max_tracks=100, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.max_tracks = max_tracks
        self.img_width = img_width
        self.img_height = img_height

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # print('matches', matches, ', unmatched_tracks', unmatched_tracks, ', unmatched_detections', unmatched_detections)

        # Move invalid class ids to unmatched detections
        def determine_class_ids(matches_tuple):
            track_idx, detection_idx = matches_tuple
            if self.tracks[track_idx].cls != detections[detection_idx].clses:
                unmatched_detections.append(detection_idx)
                return False
            return True

        matches = [m for m in matches if determine_class_ids(m)]
        
        for detection_idx in unmatched_detections:  # only except ball detections
            if detections[detection_idx].clses == 1:
                btid = self._initiate_track(detections[detection_idx], 5)
                # print("Initiated new ball track", btid)
                continue
            
            # Add new track ONLY if it didn't reached max people tracks
            if self.max_tracks > len(self.tracks):
                _ = self._initiate_track(detections[detection_idx])
            else:
                pass
                # print("Couldn't initiate new track; dropped: Detection[tlwh=%s, confidence=%f, clses=%d]" % (
                #     str(detections[detection_idx].tlwh), detections[detection_idx].confidence, detections[detection_idx].clses
                # ))
        
        # Remove ball detections as it's already initiated
        unmatched_detections = [d for d in unmatched_detections if detections[d].clses == 0]

        # Rematch unmatched detections
        # Do not add new track on unmatched detections -
        # just rematch with existing tracks.
        # (only if tracks are fully initialized)
        re_matches = []
        if len(matches) > 0 and len(unmatched_detections) > 0:
            pre_matched_detection_idxs = [ m[0] for m in matches ]
            re_matches, _, _ = self._rematch([detections[i] for i in unmatched_detections])
            # print("Rematch result: ", re_matches)

        # Update track set.
        for track_idx, detection_idx in matches:
            if self.tracks[track_idx].cls == detections[detection_idx].clses:
                self.tracks[track_idx].update(
                    self.kf, detections[detection_idx])
            else:
                pass
                # print("Skipping due to class mismatch")
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Additionally update re-matched tracks.
        # (Also, only if tracks are fully initialized)
        for track_idx, detection_idx in re_matches:
            # Do not update when lower-distance one (above) is already updated
            if track_idx not in pre_matched_detection_idxs and \
                self.tracks[track_idx].cls == detections[detection_idx].clses:
                self.tracks[track_idx].update(
                    self.kf, detections[detection_idx])
            else:
                pass
                # print("Finally skipping after re-match; dropped: Detection[tlwh=%s, confidence=%f, clses=%d]" % (
                #     str(detections[detection_idx].tlwh), detections[detection_idx].confidence, detections[detection_idx].clses
                # ))

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, img_width, img_height, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.img_width, self.img_height, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # # Associate remaining tracks together with unconfirmed tracks using IOU.
        # iou_track_candidates = unconfirmed_tracks + [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update == 1]
        # unmatched_tracks_a = [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update != 1]
        # matches_b, unmatched_tracks_b, unmatched_detections = \
        #     linear_assignment.min_cost_matching(
        #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        #         detections, iou_track_candidates, unmatched_detections)

        # Associate remaining tracks together with unconfirmed tracks using Sqrted Centerloc.
        centerloc_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                sqrt_centerloc_matching.centerloc_cost, 1.0, self.img_width, self.img_height, self.tracks,
                detections, centerloc_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _rematch(self, detections):
        def gated_metric(tracks, dets, img_width, img_height, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # previously 0.2
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, 100, self.img_width, self.img_height, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        centerloc_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                sqrt_centerloc_matching.centerloc_cost, 1.0, self.img_width, self.img_height, self.tracks,
                detections, centerloc_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, max_age_override=None):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        cls = detection.clses
        score = detection.confidence
        # print("New Track %d" % self._next_id)
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init,
            max_age_override if max_age_override is not None else self.max_age,
            cls, score, detection.feature))
        self._next_id += 1
        return (self._next_id-1)
