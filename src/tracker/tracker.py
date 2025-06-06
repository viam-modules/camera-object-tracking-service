import asyncio
import datetime
import os
from asyncio import Event, create_task, sleep
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from viam.components.camera import CameraClient
from viam.logging import getLogger
from viam.media.video import CameraMimeType
from viam.proto.service.vision import Detection

from src.config.config import TrackerConfig
from src.image.image import ImageObject
from src.tracker.detector.detector import Detector
from src.tracker.embedder.embedder import Embedder
from src.tracker.track import Track

# from src.tracker.encoder.feature_encoder import FeatureEncoder, get_encoder
# from src.tracker.face_id.identifier import FaceIdentifier
# from src.tracker.track import Track
# from src.tracker.tracks_manager import TracksManager
from src.utils import log_cost_matrix, log_tracks_info

LOGGER = getLogger(__name__)


class Tracker:
    def __init__(
        self,
        cfg: TrackerConfig,
        camera: CameraClient,
        detector: Detector,
        embedder: Embedder,
        debug: bool = False,
    ):
        """
        Initialize the Tracker with a Detector for person detection and tracking logic.

        :param model_path: Path to the TFLite model file for the Detector.
        :param iou_threshold: Threshold for IoU matching.
        :param feature_threshold: Threshold for re-id feature matching.
        """
        self.camera: CameraClient = camera
        self.lambda_value = cfg.tracker_config.lambda_value
        self.distance_threshold = cfg.tracker_config.embedder_threshold
        self.max_age_track = cfg.tracker_config.max_age_track
        self.sleep_period = 1 / (cfg.tracker_config.max_frequency)
        self.crop_region = cfg.tracker_config.crop_region

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detector: Detector = detector
        self.embedder: Embedder = embedder
        self.tracks: Dict[str, Track] = {}

        self.track_candidates: List[Track] = []

        self.category_count: Dict[str, int] = {}

        self.labeled_person_embeddings: Dict[str, List[torch.Tensor]] = {}

        self.embedder_threshold = cfg.tracker_config.embedder_threshold
        self.minimum_track_persistance = 5  # TODO: talk with Khari about this
        self.current_tracks_id = set()
        self.background_task = None
        self.new_object_event = Event()
        self.new_object_notifier = NewObjectNotifier(
            self.new_object_event, cfg.tracker_config.cooldown_period
        )
        self.stop_event = Event()

        self.debug = debug
        self.count = 0

        self.start_background_loop = cfg.tracker_config._start_background_loop

        self.last_image: ImageObject = None

    def start(self):
        if self.start_background_loop:
            self.background_task = create_task(self._background_update_loop())

    async def stop(self):
        """
        Stop the background loop by setting the stop event.
        """
        self.stop_event.set()
        self.new_object_notifier.close()
        try:
            if self.background_task is not None:
                await self.background_task  # Wait for the background task to finish
        except Exception as e:
            LOGGER.error(f"Error stopping background task: {e}")

    async def _background_update_loop(self):
        """
        Background loop that continuously gets images from the camera and updates tracks.
        """
        while not self.stop_event.is_set():
            img = await self.get_and_decode_img()
            if img is not None:
                if self.last_image is not None:
                    # Check if the current image is identical to the last one to avoid processing duplicates
                    if torch.equal(img.uint8_tensor, self.last_image.uint8_tensor):
                        continue
                self.last_image = img
                try:
                    await self.update(img)  # Update tracks
                except Exception as e:
                    LOGGER.error(f"Error updating tracker: {e}")
                    await sleep(
                        self.sleep_period * 5
                    )  # sleep a bit more if something bad happened
            await sleep(self.sleep_period)

    async def get_and_decode_img(self):
        try:
            viam_img = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
        except Exception as e:
            LOGGER.error(f"Error getting image: {e}")
            return None
        return ImageObject(viam_img, device=self.device, crop_region=self.crop_region)

    def relabel_tracks(self, dict_old_label_new_label: Dict[str, str]):
        answer = {}
        for track_id, new_label in dict_old_label_new_label.items():
            if track_id not in self.tracks:
                answer[track_id] = (
                    f"DoCommand relabelling error: couldn't find tracks with the ID: {track_id}"
                )
                continue

            self.tracks[track_id].label = new_label
            answer[track_id] = f"success: changed label to '{new_label}'"
        return answer

    def get_current_detections(self):
        """
        Get the current detections.
        """
        dets = []

        for track in self.tracks.values():
            if track.is_detected():
                dets.append(
                    track.get_detection(
                        crop_region=self.crop_region,
                        original_image_width=self.last_image.width,
                        original_image_height=self.last_image.height,
                    )
                )

        for track in self.track_candidates:
            if track.is_detected():
                dets.append(
                    track.get_detection(
                        crop_region=self.crop_region,
                        original_image_width=self.last_image.width,
                        original_image_height=self.last_image.height,
                    )
                )
        return dets

    async def is_new_object_detected(self):
        return self.new_object_event.is_set()

    async def update(
        self,
        img: ImageObject,
    ):
        """
        Update the tracker with new detections.

        :param detections: List of Detection objects detected in the current frame.
        """
        self.clear_detected_track()
        # Get new detections
        try:
            detections = await self.detector.detect(img)
        except Exception as e:
            LOGGER.error(f"Error detecting objects: {e}")
            return

        # Keep track of the old tracks, updated and unmatched tracks
        all_old_tracks_id = set(self.tracks.keys())
        updated_tracks_ids = set()
        unmatched_detections_idx = set(range(len(detections)))
        new_tracks_ids = set()

        if not detections:
            self.current_tracks_id = set()
            self.increment_age_and_delete_tracks()
            if self.debug:
                log_tracks_info(
                    updated_tracks_ids=updated_tracks_ids,
                    new_tracks_ids=new_tracks_ids,
                    lost_tracks_ids=all_old_tracks_id - updated_tracks_ids,
                )
            return

        # Compute feature vectors for the current detections
        try:
            features_vectors = await self.embedder.compute_features(img, detections)
        except Exception as e:
            LOGGER.error(f"Error computing feature vectors: {e}")
            return

        # Solve the linear assignment problem to find the best matching
        row_indices, col_indices, cost_matrix = self.get_matching_tracks(
            tracks=self.tracks, detections=detections, feature_vectors=features_vectors
        )
        # Update matched tracks
        for row, col in zip(row_indices, col_indices):
            distance = cost_matrix[row, col]
            if (
                distance < self.distance_threshold
            ):  # Threshold to determine if a match is valid
                track_id = list(self.tracks.keys())[row]
                track = self.tracks[track_id]
                detection = detections[col]
                self.tracks[track_id].update(
                    bbox=detection.bbox,
                    feature_vector=features_vectors[col],
                    distance=distance,
                )
                track.set_is_detected()
                updated_tracks_ids.add(track_id)
                unmatched_detections_idx.discard(col)

        # Find match with track candidate
        if len(unmatched_detections_idx) > 0:
            if len(self.track_candidates) < 1:
                for detection_id in unmatched_detections_idx:
                    detection = detections[detection_id]
                    feature_vector = features_vectors[detection_id]

                    self.add_track_candidate(
                        detection=detection,
                        feature_vector=feature_vector,
                    )
            else:
                unmatched_detections = [detections[i] for i in unmatched_detections_idx]
                track_candidate_idx, unmatched_detection_idx, cost_matrix = (
                    self.get_matching_track_candidates(
                        detections=unmatched_detections,
                        features_vectors=features_vectors,
                    )
                )
                promoted_track_candidates = []
                for track_candidate_id, unmatched_detection_id in zip(
                    track_candidate_idx, unmatched_detection_idx
                ):
                    distance = cost_matrix[track_candidate_id, unmatched_detection_id]
                    detection = unmatched_detections[unmatched_detection_id]
                    matching_track_candidate = self.track_candidates[track_candidate_id]

                    if distance < self.distance_threshold:
                        matching_track_candidate.update(
                            bbox=detection.bbox,
                            feature_vector=features_vectors[unmatched_detection_id],
                            distance=distance,
                        )
                        matching_track_candidate.increment_persistence()
                        if (
                            matching_track_candidate.get_persistence()
                            > self.minimum_track_persistance
                        ):
                            promoted_track_candidates.append(track_candidate_id)
                            new_track_id = self.promote_to_track(
                                track_candidate_id,
                                feature_vector=features_vectors[unmatched_detection_id],
                            )
                            new_tracks_ids.add(new_track_id)
                            self.tracks[new_track_id].set_is_detected()
                        else:
                            matching_track_candidate.set_is_detected()

                    else:
                        self.add_track_candidate(
                            detection=detection,
                            feature_vector=features_vectors[unmatched_detection_id],
                        )
                for track_candidate_id in sorted(  # sort and reverse the iteration over the promoted track_candidates to not mess up the indexes
                    promoted_track_candidates,
                    reverse=True,
                ):
                    del self.track_candidates[
                        track_candidate_id
                    ]  # delete track candidate that were not found again

                self.track_candidates = [
                    track_candidate
                    for track_candidate in self.track_candidates
                    if track_candidate.is_detected()
                ]  # mark track candidates that were not found again

        self.current_tracks_id = updated_tracks_ids.union(new_tracks_ids)

        self.increment_age_and_delete_tracks(updated_tracks_ids)

        # Set the new_object_event if new tracks were found
        if len(new_tracks_ids) > 0:
            self.new_object_notifier.notify_new_object()

        self.count += 1
        if self.debug:
            log_tracks_info(
                updated_tracks_ids=updated_tracks_ids,
                new_tracks_ids=new_tracks_ids,
                lost_tracks_ids=all_old_tracks_id - updated_tracks_ids,
            )
            log_cost_matrix(
                cost_matrix=cost_matrix,
                track_ids=list(self.tracks.keys()),
                iteration_number=self.count,
            )

    def get_matching_tracks(
        self, tracks: Dict[str, Track], detections: List[Detection], feature_vectors
    ):
        # Initialize cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))

        # Calculate cost for each pair of track and detection
        track_ids = list(tracks.keys())
        for i, track_id in enumerate(track_ids):
            track = tracks[track_id]
            for j, detection in enumerate(detections):
                iou_score = track.iou(detection.bbox)
                feature_dist = self.embedder.compute_distance(
                    track.feature_vector, feature_vectors[j]
                )
                # Cost function: lambda * feature distance + (1 - lambda) * (1 - IoU)
                cost_matrix[i, j] = self.lambda_value * feature_dist + (
                    1 - self.lambda_value
                ) * (1 - iou_score)
        # Solve the linear assignment problem to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        return row_indices, col_indices, cost_matrix

    def get_matching_track_candidates(self, detections: List, features_vectors):
        """
        Should pass the detections that are not matched with current tracks
        """
        # Initialize cost matrix
        cost_matrix = np.zeros((len(self.track_candidates), len(detections)))

        for i, track in enumerate(self.track_candidates):
            for j, detection in enumerate(detections):
                iou_score = track.iou(detection.bbox)
                feature_dist = self.embedder.compute_distance(
                    track.feature_vector, features_vectors[j]
                )
                # Cost function: lambda * feature distance + (1 - lambda) * (1 - IoU)
                cost_matrix[i, j] = self.lambda_value * feature_dist + (
                    1 - self.lambda_value
                ) * (1 - iou_score)
        # Solve the linear assignment problem to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        return row_indices, col_indices, cost_matrix

    def add_track_candidate(self, detection, feature_vector: torch.Tensor):
        new_track_candidate = Track(
            track_id=detection.category,
            bbox=detection.bbox,
            feature_vector=feature_vector,
            distance=0,
            is_candidate=True,
        )
        new_track_candidate.set_is_detected()
        self.track_candidates.append(new_track_candidate)

    def promote_to_track(self, track_candidate_indice: int, feature_vector) -> str:
        """
        takes track indice and returns track_id
        """
        if len(self.track_candidates) <= track_candidate_indice:
            return IndexError(
                f"Can't find track candidate at indice {track_candidate_indice}"
            )

        track_candidate = deepcopy(self.track_candidates[track_candidate_indice])

        track_candidate.is_candidate = False
        track_id = self.generate_track_id(
            track_candidate._get_class_name()
        )  # for a track candidate, the label is the category
        track_candidate.change_track_id(track_id)
        self.tracks[track_id] = track_candidate

        return track_id

    def clear_detected_track(self):
        for track in self.tracks.values():
            track.unset_is_detected()
        for track in self.track_candidates:
            track.unset_is_detected()

    def increment_age_and_delete_tracks(self, updated_tracks_ids=[]):
        # Remove or age out tracks that were not updated
        for track_id in list(self.tracks.keys()):
            if track_id not in updated_tracks_ids:
                self.tracks[track_id].increment_age()

                # Optionally remove old tracks
                if self.tracks[track_id].age > self.max_age_track:
                    del self.tracks[track_id]

    @staticmethod
    def generate_person_data(label, id):
        return {"label": label, "id": id, "renamed": (id != label)}

    def generate_track_id(self, category):
        """
        Generate a unique track ID based on the category and current date/time.

        :param category: The category of the detected object.
        :return: A unique track ID string in the format "<category>_N_YYYYMMDD_HHMMSS".
        """
        # Get the current count of this category
        if category not in self.category_count:
            self.category_count[category] = 0

        # Increment the count
        self.category_count[category] += 1
        count = self.category_count[category]

        # Get the current date and time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the track ID
        track_id = f"{category}_{count}_{timestamp}"

        return track_id


class NewObjectNotifier:
    def __init__(self, new_object_event: Event, cooldown_period_s: int):
        """
        Initialize the notifier with a cooldown period.

        :param cooldown_seconds: Time in seconds for the cooldown period.
        """
        self.cooldown_seconds = cooldown_period_s
        self.new_object_event = new_object_event
        self.cooldown_task = None

    def close(self):
        self.new_object_event.clear()
        if self.cooldown_task is not None:
            self.cooldown_task.cancel()

    def notify_new_object(self):
        """
        Notify that a new object has been detected and restart the cooldown.
        """
        # Set the event to notify about the new object
        self.new_object_event.set()

        # Cancel any existing cooldown task
        if self.cooldown_task is not None:
            self.cooldown_task.cancel()

        # Start a new cooldown task
        self.cooldown_task = asyncio.create_task(self._clear_event_after_cooldown())

    async def _clear_event_after_cooldown(self):
        """
        Clear the event after the cooldown period.
        """
        try:
            await asyncio.sleep(self.cooldown_seconds)
            self.new_object_event.clear()
        except asyncio.CancelledError:
            pass
