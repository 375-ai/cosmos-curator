# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for lifting tracked 2D boxes into 3D cuboids."""

import math

import numpy as np
import pytest

from cosmos_curator.pipelines.video.scene3d.calibration import Calib
from cosmos_curator.pipelines.video.scene3d.detection_source import (
    DetectionTracks,
    FrameDetections,
    Sam3DetectionSource,
    TrackedBox,
)
from cosmos_curator.pipelines.video.scene3d.object_lift import (
    DepthSequence,
    ObjectLiftParams,
    build_scene_objects,
)
from tests.cosmos_curator.pipelines.video.scene3d.scene_fixtures import HEIGHT, WIDTH

FPS = 5.0
OBJECT_HEIGHT_M = 1.5


def _box_for(calib: Calib, position: np.ndarray, *, half_width_px: float = 25.0) -> tuple[float, ...]:
    """Project a ground point into the 2D box a detector would have produced."""
    base = calib.ground_to_pixel(position)
    top = calib.ground_to_pixel(position + np.array([0.0, 0.0, OBJECT_HEIGHT_M]))
    assert base is not None
    assert top is not None
    return (float(base[0] - half_width_px), float(top[1]), float(base[0] + half_width_px), float(base[1]))


def _straight_track(  # noqa: PLR0913  # a readable knob per scenario beats a config object here
    calib: Calib,
    *,
    label: str = "a red car",
    speed_mps: float = 5.0,
    num_frames: int = 20,
    object_id: int = 1,
    start_x: float = 20.0,
    lateral_y: float = 2.0,
) -> DetectionTracks:
    """Build a single object driving along map +X at a constant speed."""
    frames = []
    for index in range(num_frames):
        timestamp = index / FPS
        position = np.array([start_x + speed_mps * timestamp, lateral_y, 0.0])
        frames.append(
            FrameDetections(
                timestamp_s=timestamp,
                boxes=[TrackedBox(object_id=object_id, label=label, box_xyxy=_box_for(calib, position))],
            )
        )
    return DetectionTracks(frames=frames, frame_width=WIDTH, frame_height=HEIGHT)


def test_straight_track_recovers_position_heading_and_speed(calib: Calib, ground_depth: np.ndarray) -> None:
    """A constant-velocity track yields the right yaw, speed and world positions."""
    tracks = _straight_track(calib, speed_mps=5.0)

    records = build_scene_objects(calib, ground_depth, tracks, ObjectLiftParams())

    assert len(records) == len(tracks.frames)
    first = records[0]["entities"][0]
    last = records[-1]["entities"][0]
    assert first["id"] == "obj_1"
    assert first["label"] == "car"
    assert math.degrees(first["cube"]["yaw"]) == pytest.approx(0.0, abs=1.0)
    assert float(first["metadata"]["speed_mps"]) == pytest.approx(5.0, abs=0.2)
    assert first["cube"]["position"][0] == pytest.approx(20.0, abs=0.2)
    # 19 frames at 5 Hz and 5 m/s is 19 m of travel.
    assert last["cube"]["position"][0] - first["cube"]["position"][0] == pytest.approx(19.0, abs=0.3)
    # snap_to_ground puts the cube centre at half its height above z=0.
    assert first["cube"]["position"][2] == pytest.approx(first["cube"]["size"][2] / 2.0, abs=1e-6)
    assert first["arrow"] is not None


def test_prior_sizing_wins_for_a_recognised_label(calib: Calib, ground_depth: np.ndarray) -> None:
    """'auto' uses the class prior when the label matches, since length is unobservable."""
    records = build_scene_objects(calib, ground_depth, _straight_track(calib), ObjectLiftParams(dimension_mode="auto"))
    assert records[0]["entities"][0]["cube"]["size"] == (4.5, 1.8, 1.5)


def test_depth_sizing_measures_the_box(calib: Calib, ground_depth: np.ndarray) -> None:
    """'depth' recovers the metric extent of the 2D box with no class knowledge."""
    records = build_scene_objects(calib, ground_depth, _straight_track(calib), ObjectLiftParams(dimension_mode="depth"))
    _, width_m, height_m = records[0]["entities"][0]["cube"]["size"]
    # The synthetic box is exactly OBJECT_HEIGHT_M tall in the world.
    assert height_m == pytest.approx(OBJECT_HEIGHT_M, rel=0.15)
    assert 0.5 < width_m < 6.0


def test_unknown_label_falls_back_to_measured_extent(calib: Calib, ground_depth: np.ndarray) -> None:
    """A label with no prior is sized from depth even in 'auto' mode."""
    records = build_scene_objects(
        calib, ground_depth, _straight_track(calib, label="a glorpsnaggle"), ObjectLiftParams()
    )
    entity = records[0]["entities"][0]
    assert entity["label"] == "a glorpsnaggle"
    assert entity["cube"]["size"][2] == pytest.approx(OBJECT_HEIGHT_M, rel=0.15)


def test_short_tracks_are_dropped_as_ghosts(calib: Calib, ground_depth: np.ndarray) -> None:
    """A track seen in fewer frames than the threshold produces no cuboids."""
    tracks = _straight_track(calib, num_frames=3)
    records = build_scene_objects(calib, ground_depth, tracks, ObjectLiftParams(min_track_points=4))
    assert records == []


def test_stationary_track_gets_no_heading(calib: Calib, ground_depth: np.ndarray) -> None:
    """Below the net-displacement threshold the object is drawn axis-aligned."""
    tracks = _straight_track(calib, speed_mps=0.0)
    records = build_scene_objects(calib, ground_depth, tracks, ObjectLiftParams(min_net_displacement_m=1.0))
    entity = records[0]["entities"][0]
    assert entity["arrow"] is None
    assert entity["cube"]["yaw"] == 0.0
    assert float(entity["metadata"]["speed_mps"]) == pytest.approx(0.0, abs=0.05)


def test_speed_is_clamped(calib: Calib, ground_depth: np.ndarray) -> None:
    """A lift outlier cannot report an absurd speed."""
    tracks = _straight_track(calib, speed_mps=20.0)
    records = build_scene_objects(calib, ground_depth, tracks, ObjectLiftParams(max_speed_mps=3.0))
    assert float(records[0]["entities"][0]["metadata"]["speed_mps"]) == pytest.approx(3.0)


def test_objects_beyond_max_range_are_culled(calib: Calib, ground_depth: np.ndarray) -> None:
    """The range cull applies to objects as well as to the background cloud."""
    tracks = _straight_track(calib, start_x=20.0, speed_mps=5.0)
    records = build_scene_objects(calib, ground_depth, tracks, ObjectLiftParams(max_range_m=5.0))
    assert records == []


def test_snap_to_ground_uses_the_plane_not_the_noisy_lift(calib: Calib, ground_depth: np.ndarray) -> None:
    """Snapping pins the cuboid base to z=0; disabling it follows the lifted elevation.

    Even on a perfectly planar scene the two differ by centimetres, because the
    depth lookup quantises to whole pixels. That jitter is exactly what snapping
    removes, so the free elevations vary between frames while the snapped ones do not.
    """
    tracks = _straight_track(calib)
    snapped = build_scene_objects(calib, ground_depth, tracks, ObjectLiftParams(snap_to_ground=True))
    free = build_scene_objects(calib, ground_depth, tracks, ObjectLiftParams(snap_to_ground=False))

    half_height = snapped[0]["entities"][0]["cube"]["size"][2] / 2.0
    snapped_z = {record["entities"][0]["cube"]["position"][2] for record in snapped}
    assert snapped_z == {half_height}

    free_z = [record["entities"][0]["cube"]["position"][2] for record in free]
    assert len(set(free_z)) > 1
    # Still the same surface, just followed rather than idealised.
    assert all(abs(z - half_height) < 0.1 for z in free_z)


def test_opposing_flows_get_distinct_arrow_colours(calib: Calib, ground_depth: np.ndarray) -> None:
    """The SVD principal axis two-colours traffic without any domain knowledge."""
    forward = _straight_track(calib, speed_mps=5.0, object_id=1, lateral_y=2.0)
    backward = _straight_track(calib, speed_mps=-5.0, object_id=2, start_x=45.0, lateral_y=-6.0)
    merged = DetectionTracks(
        frames=[
            FrameDetections(timestamp_s=a.timestamp_s, boxes=a.boxes + b.boxes)
            for a, b in zip(forward.frames, backward.frames, strict=True)
        ],
        frame_width=WIDTH,
        frame_height=HEIGHT,
    )

    entities = build_scene_objects(calib, ground_depth, merged, ObjectLiftParams())[0]["entities"]
    colours = {entity["id"]: entity["arrow"]["color"] for entity in entities if entity["arrow"]}
    assert len(colours) == 2
    assert len(set(colours.values())) == 2


def test_lifetime_spans_two_sampling_periods(calib: Calib, ground_depth: np.ndarray) -> None:
    """Each entity persists until its successor arrives."""
    records = build_scene_objects(calib, ground_depth, _straight_track(calib), ObjectLiftParams())
    assert records[0]["entities"][0]["lifetime_ns"] == pytest.approx(2e9 / FPS, rel=1e-3)


def test_boxes_are_rescaled_from_detector_resolution(calib: Calib, ground_depth: np.ndarray) -> None:
    """Detector pixels are mapped onto the depth map's grid before lifting."""
    full = _straight_track(calib)
    half = DetectionTracks(
        frames=[
            FrameDetections(
                timestamp_s=frame.timestamp_s,
                boxes=[
                    TrackedBox(
                        object_id=box.object_id,
                        label=box.label,
                        box_xyxy=tuple(value / 2 for value in box.box_xyxy),  # type: ignore[arg-type]
                    )
                    for box in frame.boxes
                ],
            )
            for frame in full.frames
        ],
        frame_width=WIDTH // 2,
        frame_height=HEIGHT // 2,
    )

    from_full = build_scene_objects(calib, ground_depth, full, ObjectLiftParams())
    from_half = build_scene_objects(calib, ground_depth, half, ObjectLiftParams())
    assert from_full[0]["entities"][0]["cube"]["position"][0] == pytest.approx(
        from_half[0]["entities"][0]["cube"]["position"][0], abs=0.5
    )


class _FakeClip:
    """Minimal stand-in for the Clip fields the SAM3 adapter reads."""

    def __init__(self, frames: object, width: int | None, height: int | None) -> None:
        self.sam3_frames = frames
        self.sam3_frame_width = width
        self.sam3_frame_height = height


def test_sam3_source_adapts_track_records() -> None:
    """The adapter reads prompt/object_id/box_xyxy and keeps real PTS."""
    clip = _FakeClip(
        [
            {
                "frame_idx": 0,
                "timestamp_s": 0.0,
                "detections": [{"prompt": "a car", "object_id": 3, "box_xyxy": [10, 20, 50, 80]}],
            },
            {"frame_idx": 1, "timestamp_s": 0.2, "detections": []},
        ],
        640,
        360,
    )

    tracks = Sam3DetectionSource().tracks(clip)  # type: ignore[arg-type]

    assert tracks is not None
    assert (tracks.frame_width, tracks.frame_height) == (640, 360)
    assert [frame.timestamp_s for frame in tracks.frames] == [0.0, 0.2]
    (box,) = tracks.frames[0].boxes
    assert (box.object_id, box.label, box.box_xyxy) == (3, "a car", (10.0, 20.0, 50.0, 80.0))


def test_sam3_source_returns_none_without_tracks() -> None:
    """No SAM3 run, or no frame geometry, means no objects to lift."""
    assert Sam3DetectionSource().tracks(_FakeClip(None, 640, 360)) is None  # type: ignore[arg-type]
    assert Sam3DetectionSource().tracks(_FakeClip([], 640, 360)) is None  # type: ignore[arg-type]
    frames = [{"timestamp_s": 0.0, "detections": []}]
    assert Sam3DetectionSource().tracks(_FakeClip(frames, 0, 0)) is None  # type: ignore[arg-type]


def test_sam3_source_skips_malformed_detections() -> None:
    """Bad boxes are dropped per-detection rather than failing the clip."""
    clip = _FakeClip(
        [
            {
                "timestamp_s": 0.0,
                "detections": [
                    {"prompt": "a car", "object_id": 1, "box_xyxy": [10, 20, 50]},
                    {"prompt": "a car", "object_id": 2, "box_xyxy": [10, 20, 10, 80]},
                    {"prompt": "a car", "box_xyxy": [10, 20, 50, 80]},
                    {"prompt": "a car", "object_id": 5, "box_xyxy": [50, 80, 10, 20]},
                ],
            }
        ],
        640,
        360,
    )

    tracks = Sam3DetectionSource().tracks(clip)  # type: ignore[arg-type]

    assert tracks is not None
    # Only the reversed-but-recoverable box survives; short, degenerate and
    # id-less detections are dropped.
    assert [box.object_id for box in tracks.frames[0].boxes] == [5]
    assert tracks.frames[0].boxes[0].box_xyxy == (10.0, 20.0, 50.0, 80.0)


def test_per_frame_depth_is_matched_by_time_not_by_index(calib: Calib, ground_depth: np.ndarray) -> None:
    """Depth maps pair with track frames by timestamp, because the two decodes differ.

    The detector samples at ``--sam3-target-fps`` and this stage at
    ``--scene3d-target-fps``; indexing depth by position silently misaligns whenever
    those differ. Here the depth sequence runs at half the track rate, and only the
    time-matched pairing puts the right depth against the right frame.
    """
    tracks = _straight_track(calib, num_frames=8)
    # Depth sampled at half the track rate: 4 maps for 8 track frames.
    depth_timestamps = [frame.timestamp_s for frame in tracks.frames[::2]]
    far = ground_depth * 2.0
    sequence = DepthSequence.build(depth_timestamps, [ground_depth, far, ground_depth, far])

    assert sequence.nearest(tracks.frames[0].timestamp_s) is ground_depth
    # Track frame 2 shares a timestamp with depth index 1 (the doubled map).
    assert sequence.nearest(tracks.frames[2].timestamp_s) is far
    # Track frame 1 sits between the two; nearest wins the tie toward the earlier map.
    assert sequence.nearest(tracks.frames[1].timestamp_s) in (ground_depth, far)

    records = build_scene_objects(calib, ground_depth, tracks, ObjectLiftParams(), depth_by_frame=sequence)
    # Alternating depth scales alternate the reported range, which is exactly the
    # signal that would be lost if maps were paired to frames positionally.
    ranges = [record["entities"][0]["cube"]["position"][0] for record in records]
    assert len(set(ranges)) > 1


def test_depth_sequence_truncates_to_the_shorter_input() -> None:
    """A short depth batch never indexes past its timestamps."""
    maps = [np.ones((4, 4), dtype=np.float32)]
    sequence = DepthSequence.build([0.0, 0.5, 1.0], maps)
    assert len(sequence.depths) == 1
    assert sequence.timestamps_s.tolist() == [0.0]
    assert sequence.nearest(99.0) is maps[0]


def test_empty_depth_sequence_has_no_maps() -> None:
    """An empty batch reads as 'no per-frame depth', falling back to the plate."""
    assert DepthSequence.build([], []).depths == []
