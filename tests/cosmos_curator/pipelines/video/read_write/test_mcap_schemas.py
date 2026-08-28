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
"""Tests for the MCAP jsonschema constants and payload builders."""

import base64
import json
import math

import numpy as np
import numpy.testing as npt
import pytest

from cosmos_curator.pipelines.video.read_write import mcap_schemas


def test_topic_schemas_cover_all_topics() -> None:
    """Every topic constant maps to a titled schema."""
    assert set(mcap_schemas.TOPIC_SCHEMAS) == {
        mcap_schemas.TOPIC_IMAGE_RAW,
        mcap_schemas.TOPIC_SCENE_ANNOTATION,
        mcap_schemas.TOPIC_AUDIO_RAW,
        mcap_schemas.TOPIC_CAMERA_INFO,
        mcap_schemas.TOPIC_TF_STATIC,
        mcap_schemas.TOPIC_CLIP_EMBEDDING,
        mcap_schemas.TOPIC_SCENE_BACKGROUND,
        mcap_schemas.TOPIC_SCENE_OBJECTS,
    }
    for schema in mcap_schemas.TOPIC_SCHEMAS.values():
        assert schema["title"]


def test_payload_builders_round_trip() -> None:
    """Payload builders emit valid JSON matching the schema field types."""
    ns = 1_500_000_000
    video_payload = json.loads(mcap_schemas.compressed_video_message(ns, "camera", b"\x00\x01", "h264"))
    assert video_payload["timestamp"] == {"sec": 1, "nsec": 500_000_000}
    assert base64.b64decode(video_payload["data"]) == b"\x00\x01"

    audio_payload = json.loads(mcap_schemas.raw_audio_message(0, b"\x00\x00", 48000, 1))
    assert audio_payload["format"] == "pcm-s16"

    placeholder = mcap_schemas.placeholder_camera_model(1920, 1080)
    calibration_payload = json.loads(mcap_schemas.camera_calibration_message(0, "camera", placeholder))
    assert len(calibration_payload["K"]) == 9
    assert len(calibration_payload["P"]) == 12
    assert calibration_payload["width"] == 1920
    # The placeholder claims no focal length, only a centred principal point.
    assert calibration_payload["K"][0] == 1.0
    assert calibration_payload["K"][2] == 960.0

    transforms_payload = json.loads(mcap_schemas.frame_transforms_message(0, "camera", placeholder))
    assert transforms_payload["transforms"][0]["child_frame_id"] == "camera"
    assert transforms_payload["transforms"][0]["translation"] == {"x": 0.0, "y": 0.0, "z": 0.0}

    vector = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    embedding_payload = json.loads(mcap_schemas.clip_embedding_message(0, "m", "v", vector))
    decoded = np.frombuffer(base64.b64decode(embedding_payload["data"]), dtype="<f4")
    npt.assert_allclose(decoded, [1.0, 2.0, 3.0, 4.0])


def test_calibration_and_transform_accept_a_real_camera_model() -> None:
    """A scene3d calibration payload replaces the identity placeholders."""
    calibration = {
        "width": 640,
        "height": 360,
        "K": [500.0, 0.0, 320.0, 0.0, 500.0, 180.0, 0.0, 0.0, 1.0],
        "D": [0.0] * 5,
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 180.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "translation": [0.0, 0.0, 8.25],
        "rotation": [0.1, 0.2, 0.3, 0.9273618],
    }
    payload = json.loads(mcap_schemas.camera_calibration_message(0, "camera", calibration))
    # Image size follows the calibration, not the source video, because the
    # estimate was made at the depth map's resolution.
    assert (payload["width"], payload["height"]) == (640, 360)
    assert payload["K"][0] == 500.0

    transform = json.loads(mcap_schemas.frame_transforms_message(0, "camera", calibration))["transforms"][0]
    assert transform["parent_frame_id"] == mcap_schemas.MAP_FRAME_ID
    assert transform["translation"]["z"] == 8.25
    assert transform["rotation"]["w"] == 0.9273618

    # The placeholder model takes the same path and yields the identity transform.
    identity = json.loads(
        mcap_schemas.frame_transforms_message(0, "camera", mcap_schemas.placeholder_camera_model(1920, 1080))
    )["transforms"][0]
    assert identity["translation"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert identity["rotation"] == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def test_point_cloud_message_round_trips_packed_points() -> None:
    """The packed (N, 7) array reaches the wire unchanged at 28 bytes per point."""
    packed = np.array(
        [[1.0, 2.0, 3.0, 1.0, 0.5, 0.25, 1.0], [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    payload = json.loads(mcap_schemas.point_cloud_message(0, "map", packed))
    assert payload["frame_id"] == "map"
    assert payload["point_stride"] == mcap_schemas.POINT_CLOUD_STRIDE_BYTES
    assert [field["name"] for field in payload["fields"]] == ["x", "y", "z", "red", "green", "blue", "alpha"]
    assert [field["offset"] for field in payload["fields"]] == [0, 4, 8, 12, 16, 20, 24]

    raw = base64.b64decode(payload["data"])
    assert len(raw) == packed.shape[0] * mcap_schemas.POINT_CLOUD_STRIDE_BYTES
    npt.assert_allclose(np.frombuffer(raw, dtype="<f4").reshape(-1, 7), packed)


def test_scene_update_message_builds_full_entities() -> None:
    """Cuboid records become SceneEntities with every primitive array present."""
    entity = {
        "id": "obj_7",
        "lifetime_ns": 400_000_000,
        "cube": {
            "position": (10.0, -2.0, 0.75),
            "size": (4.5, 1.8, 1.5),
            "yaw": math.pi / 2,
            "color": (0.2, 0.55, 1.0, 0.75),
        },
        "arrow": {"position": (10.0, -2.0, 1.8), "yaw": math.pi / 2, "length": 4.0, "color": (0.1, 0.85, 1.0, 1.0)},
        "text": {"position": (10.0, -2.0, 2.2), "value": "car #7", "font_size": 0.5},
        "metadata": {"class": "car", "speed_mps": "11.2"},
    }
    payload = json.loads(mcap_schemas.scene_update_message(1_000_000_000, "map", [entity]))
    assert payload["deletions"] == []
    (rendered,) = payload["entities"]
    assert rendered["id"] == "obj_7"
    assert rendered["frame_id"] == "map"
    assert rendered["timestamp"] == {"sec": 1, "nsec": 0}
    assert rendered["lifetime"] == {"sec": 0, "nsec": 400_000_000}
    # Foxglove rejects entities with missing primitive keys, so all are emitted.
    for key in ("arrows", "cubes", "spheres", "cylinders", "lines", "triangles", "texts", "models"):
        assert key in rendered
    assert rendered["spheres"] == []
    assert rendered["cubes"][0]["size"] == {"x": 4.5, "y": 1.8, "z": 1.5}
    # A yaw of pi/2 is a pure z-rotation: (0, 0, sin(pi/4), cos(pi/4)).
    orientation = rendered["cubes"][0]["pose"]["orientation"]
    assert orientation["x"] == 0.0
    assert orientation["y"] == 0.0
    assert orientation["z"] == pytest.approx(math.sqrt(0.5))
    assert orientation["w"] == pytest.approx(math.sqrt(0.5))
    assert rendered["arrows"][0]["shaft_length"] == pytest.approx(2.8)
    assert rendered["arrows"][0]["head_length"] == pytest.approx(1.2)
    assert rendered["texts"][0]["text"] == "car #7"
    assert rendered["texts"][0]["billboard"] is True
    assert rendered["metadata"] == [{"key": "class", "value": "car"}, {"key": "speed_mps", "value": "11.2"}]


def test_scene_update_message_omits_arrow_for_stationary_objects() -> None:
    """A track with no trusted heading renders as a cube and label only."""
    entity = {
        "id": "obj_1",
        "lifetime_ns": 0,
        "cube": {"position": (1.0, 1.0, 0.5), "size": (1.0, 1.0, 1.0), "yaw": 0.0, "color": (0.7, 0.7, 0.7, 0.75)},
        "arrow": None,
        "text": {"position": (1.0, 1.0, 1.2), "value": "object #1", "font_size": 0.5},
        "metadata": {},
    }
    (rendered,) = json.loads(mcap_schemas.scene_update_message(0, "map", [entity]))["entities"]
    assert rendered["arrows"] == []
    assert len(rendered["cubes"]) == 1
    assert rendered["metadata"] == []
