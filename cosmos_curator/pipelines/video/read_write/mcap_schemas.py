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
"""MCAP topic constants, jsonschema definitions, and message payload builders.

The channel layout mirrors the Foxglove-compatible session recordings the
curated MCAP output is meant to interoperate with: video frames on
``/camera/image-raw`` (`foxglove.CompressedVideo`), captions and detection
payloads on ``/scene-annotation`` (`midcentury.SceneAnnotation`), audio on
``/camera/audio-raw`` (`foxglove.RawAudio`), one-shot calibration/transform
messages, and clip embeddings on ``/clip/embedding``
(`midcentury.ClipEmbedding`, curator-specific).

All payloads are JSON (schema encoding ``jsonschema``, message encoding
``json``); binary fields (video bitstream, audio samples, embedding vectors)
are base64 strings per the Foxglove convention.
"""

import base64
import json
import math
from collections.abc import Buffer, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from cosmos_curator.pipelines.video.utils.ns_timing import NS_PER_SECOND

TOPIC_IMAGE_RAW = "/camera/image-raw"
TOPIC_SCENE_ANNOTATION = "/scene-annotation"
TOPIC_AUDIO_RAW = "/camera/audio-raw"
TOPIC_CAMERA_INFO = "/camera/camera-info"
TOPIC_TF_STATIC = "/tf-static"
TOPIC_CLIP_EMBEDDING = "/clip/embedding"
TOPIC_SCENE_BACKGROUND = "/scene/background"
TOPIC_SCENE_OBJECTS = "/scene/objects"

SESSION_METADATA_RECORD_NAME = "session-metadata"

# Parent frame of the static transform; the frame 3D geometry is expressed in.
MAP_FRAME_ID = "map"

JSONSCHEMA_ENCODING = "jsonschema"
JSON_MESSAGE_ENCODING = "json"

_TIME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "time",
    "properties": {
        "sec": {"type": "integer", "minimum": 0},
        "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
    },
}

_VECTOR3_SCHEMA: dict[str, Any] = {
    "title": "foxglove.Vector3",
    "type": "object",
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
    },
    "required": ["x", "y", "z"],
}

_QUATERNION_SCHEMA: dict[str, Any] = {
    "title": "foxglove.Quaternion",
    "type": "object",
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
        "w": {"type": "number"},
    },
    "required": ["x", "y", "z", "w"],
}

COMPRESSED_VIDEO_SCHEMA_NAME = "foxglove.CompressedVideo"
COMPRESSED_VIDEO_SCHEMA: dict[str, Any] = {
    "title": COMPRESSED_VIDEO_SCHEMA_NAME,
    "description": "A single frame of a compressed video bitstream",
    "type": "object",
    "properties": {
        "timestamp": {**_TIME_SCHEMA, "description": "Timestamp of video frame"},
        "frame_id": {"type": "string", "description": "Frame of reference for the video."},
        "data": {"type": "string", "contentEncoding": "base64", "description": "Compressed video frame data."},
        "format": {"type": "string", "description": "Video format. Supported values: h264, h265, vp9, av1."},
    },
    "required": ["timestamp", "frame_id", "data", "format"],
}

RAW_AUDIO_SCHEMA_NAME = "foxglove.RawAudio"
RAW_AUDIO_SCHEMA: dict[str, Any] = {
    "title": RAW_AUDIO_SCHEMA_NAME,
    "description": "A single block of an audio bitstream",
    "type": "object",
    "properties": {
        "timestamp": {**_TIME_SCHEMA, "description": "Timestamp of the start of the audio block"},
        "data": {
            "type": "string",
            "contentEncoding": "base64",
            "description": "Audio data. The samples in the data must be interleaved and little-endian",
        },
        "format": {"type": "string", "description": "Audio format. Only 'pcm-s16' is currently supported"},
        "sample_rate": {"type": "integer", "minimum": 0, "description": "Sample rate in Hz"},
        "number_of_channels": {"type": "integer", "minimum": 0, "description": "Number of channels in the audio block"},
    },
    "required": ["timestamp", "data", "format", "sample_rate", "number_of_channels"],
}

CAMERA_CALIBRATION_SCHEMA_NAME = "foxglove.CameraCalibration"
CAMERA_CALIBRATION_SCHEMA: dict[str, Any] = {
    "title": CAMERA_CALIBRATION_SCHEMA_NAME,
    "description": "Camera calibration parameters",
    "type": "object",
    "properties": {
        "timestamp": {**_TIME_SCHEMA, "description": "Timestamp of calibration data"},
        "frame_id": {"type": "string"},
        "width": {"type": "integer", "minimum": 0, "description": "Image width"},
        "height": {"type": "integer", "minimum": 0, "description": "Image height"},
        "distortion_model": {"type": "string", "description": "Name of distortion model"},
        "D": {"type": "array", "items": {"type": "number"}, "description": "Distortion parameters"},
        "K": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 9,
            "maxItems": 9,
            "description": "Intrinsic camera matrix (3x3 row-major)",
        },
        "R": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 9,
            "maxItems": 9,
            "description": "Rectification matrix (3x3 row-major)",
        },
        "P": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 12,
            "maxItems": 12,
            "description": "Projection matrix (3x4 row-major)",
        },
    },
    "required": ["timestamp", "frame_id", "width", "height", "distortion_model", "D", "K", "R", "P"],
}

FRAME_TRANSFORMS_SCHEMA_NAME = "foxglove.FrameTransforms"
FRAME_TRANSFORMS_SCHEMA: dict[str, Any] = {
    "title": FRAME_TRANSFORMS_SCHEMA_NAME,
    "description": "An array of FrameTransform messages",
    "type": "object",
    "properties": {
        "transforms": {
            "type": "array",
            "items": {
                "title": "foxglove.FrameTransform",
                "type": "object",
                "properties": {
                    "timestamp": _TIME_SCHEMA,
                    "parent_frame_id": {"type": "string"},
                    "child_frame_id": {"type": "string"},
                    "translation": _VECTOR3_SCHEMA,
                    "rotation": _QUATERNION_SCHEMA,
                },
                "required": ["timestamp", "parent_frame_id", "child_frame_id", "translation", "rotation"],
            },
            "description": "Array of transforms",
        },
    },
    "required": ["transforms"],
}

SCENE_ANNOTATION_SCHEMA_NAME = "midcentury.SceneAnnotation"
SCENE_ANNOTATION_SCHEMA: dict[str, Any] = {
    "title": SCENE_ANNOTATION_SCHEMA_NAME,
    "type": "object",
    "properties": {
        "timestamp": _TIME_SCHEMA,
        "data": {"type": "string", "description": "JSON-encoded detection payload or scene description."},
    },
    "required": ["timestamp", "data"],
}

CLIP_EMBEDDING_SCHEMA_NAME = "midcentury.ClipEmbedding"
CLIP_EMBEDDING_SCHEMA: dict[str, Any] = {
    "title": CLIP_EMBEDDING_SCHEMA_NAME,
    "type": "object",
    "properties": {
        "timestamp": _TIME_SCHEMA,
        "model_name": {"type": "string", "description": "Embedding model/algorithm name."},
        "model_version": {"type": "string", "description": "Embedding model version."},
        "data": {
            "type": "string",
            "contentEncoding": "base64",
            "description": "Little-endian float32 embedding vector.",
        },
    },
    "required": ["timestamp", "model_name", "model_version", "data"],
}


POINT_CLOUD_SCHEMA_NAME = "foxglove.PointCloud"
POINT_CLOUD_SCHEMA: dict[str, Any] = {
    "title": POINT_CLOUD_SCHEMA_NAME,
    "description": "A collection of N-dimensional points in a frame of reference",
    "type": "object",
    "properties": {
        "timestamp": {**_TIME_SCHEMA, "description": "Timestamp of point cloud"},
        "frame_id": {"type": "string", "description": "Frame of reference the points are in."},
        "pose": {
            "type": "object",
            "title": "foxglove.Pose",
            "properties": {
                "position": _VECTOR3_SCHEMA,
                "orientation": _QUATERNION_SCHEMA,
            },
            "description": "Origin of the point cloud relative to the frame of reference.",
        },
        "point_stride": {
            "type": "integer",
            "minimum": 0,
            "description": "Number of bytes between points in the data.",
        },
        "fields": {
            "type": "array",
            "items": {
                "title": "foxglove.PackedElementField",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "type": {"type": "integer", "minimum": 0, "description": "Numeric type; 7 = FLOAT32."},
                },
                "required": ["name", "offset", "type"],
            },
            "description": "Fields in each point, in order.",
        },
        "data": {
            "type": "string",
            "contentEncoding": "base64",
            "description": "Point data, interpreted using `fields`.",
        },
    },
    "required": ["timestamp", "frame_id", "pose", "point_stride", "fields", "data"],
}

SCENE_UPDATE_SCHEMA_NAME = "foxglove.SceneUpdate"
SCENE_UPDATE_SCHEMA: dict[str, Any] = {
    "title": SCENE_UPDATE_SCHEMA_NAME,
    "description": "An update to the entities displayed in a 3D scene",
    "type": "object",
    "properties": {
        "deletions": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Scene entities to delete.",
        },
        "entities": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Scene entities to add or replace.",
        },
    },
    "required": ["deletions", "entities"],
}

# Every topic carries exactly one schema; channel registration is driven by this map.
TOPIC_SCHEMAS: dict[str, dict[str, Any]] = {
    TOPIC_IMAGE_RAW: COMPRESSED_VIDEO_SCHEMA,
    TOPIC_SCENE_ANNOTATION: SCENE_ANNOTATION_SCHEMA,
    TOPIC_AUDIO_RAW: RAW_AUDIO_SCHEMA,
    TOPIC_CAMERA_INFO: CAMERA_CALIBRATION_SCHEMA,
    TOPIC_TF_STATIC: FRAME_TRANSFORMS_SCHEMA,
    TOPIC_CLIP_EMBEDDING: CLIP_EMBEDDING_SCHEMA,
    TOPIC_SCENE_BACKGROUND: POINT_CLOUD_SCHEMA,
    TOPIC_SCENE_OBJECTS: SCENE_UPDATE_SCHEMA,
}


def ns_to_time(ns: int) -> dict[str, int]:
    """Convert integer nanoseconds to a Foxglove ``time`` object."""
    return {"sec": ns // NS_PER_SECOND, "nsec": ns % NS_PER_SECOND}


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def compressed_video_message(ns: int, frame_id: str, data: Buffer, video_format: str) -> bytes:
    """Build a ``foxglove.CompressedVideo`` JSON payload."""
    return _encode(
        {
            "timestamp": ns_to_time(ns),
            "frame_id": frame_id,
            "data": base64.b64encode(data).decode("ascii"),
            "format": video_format,
        }
    )


def raw_audio_message(ns: int, data: Buffer, sample_rate: int, number_of_channels: int) -> bytes:
    """Build a ``foxglove.RawAudio`` JSON payload (interleaved little-endian pcm-s16)."""
    return _encode(
        {
            "timestamp": ns_to_time(ns),
            "data": base64.b64encode(data).decode("ascii"),
            "format": "pcm-s16",
            "sample_rate": sample_rate,
            "number_of_channels": number_of_channels,
        }
    )


def scene_annotation_message(ns: int, text: str) -> bytes:
    """Build a ``midcentury.SceneAnnotation`` JSON payload.

    ``text`` is either a plain-text scene description (caption) or a
    JSON-encoded detection payload; both share the topic.
    """
    return _encode({"timestamp": ns_to_time(ns), "data": text})


def clip_embedding_message(ns: int, model_name: str, model_version: str, vector: npt.NDArray[np.float32]) -> bytes:
    """Build a ``midcentury.ClipEmbedding`` JSON payload."""
    raw = vector.reshape(-1).astype("<f4", copy=False).tobytes()
    return _encode(
        {
            "timestamp": ns_to_time(ns),
            "model_name": model_name,
            "model_version": model_version,
            "data": base64.b64encode(raw).decode("ascii"),
        }
    )


# Pose half of a camera model: map and camera coincide. Kept separate because a
# transform is publishable even for a video whose frame size is unknown.
IDENTITY_CAMERA_POSE: dict[str, Any] = {
    "translation": [0.0, 0.0, 0.0],
    "rotation": [0.0, 0.0, 0.0, 1.0],
}


def placeholder_camera_model(width: int, height: int) -> dict[str, Any]:
    """Build an identity-style camera model for a video with no measured calibration.

    Expressed as a value in the same shape a real estimate takes, so
    :func:`camera_calibration_message` has one code path rather than a branch. K/R/P
    are identity-style with the principal point at the image centre: well-formed for
    Foxglove without claiming a focal length the pipeline does not know.
    """
    cx = width / 2
    cy = height / 2
    return {
        "width": width,
        "height": height,
        "K": [1.0, 0.0, cx, 0.0, 1.0, cy, 0.0, 0.0, 1.0],
        "D": [0.0, 0.0, 0.0, 0.0, 0.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [1.0, 0.0, cx, 0.0, 0.0, 1.0, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        **IDENTITY_CAMERA_POSE,
    }


def camera_calibration_message(ns: int, frame_id: str, calibration: dict[str, Any]) -> bytes:
    """Build a ``foxglove.CameraCalibration`` JSON payload from a camera model.

    ``calibration`` is either ``Scene3DStage``'s estimate or
    :func:`placeholder_camera_model`; both carry the same keys, and its
    ``width``/``height`` are the single source of truth for the published size.
    """
    return _encode(
        {
            "timestamp": ns_to_time(ns),
            "frame_id": frame_id,
            "width": int(calibration["width"]),
            "height": int(calibration["height"]),
            "distortion_model": "plumb_bob",
            "D": [float(value) for value in calibration["D"]],
            "K": [float(value) for value in calibration["K"]],
            "R": [float(value) for value in calibration["R"]],
            "P": [float(value) for value in calibration["P"]],
        }
    )


def frame_transforms_message(ns: int, child_frame_id: str, pose: dict[str, Any]) -> bytes:
    """Build a ``foxglove.FrameTransforms`` JSON payload for the map->camera transform.

    ``pose`` is any mapping carrying ``translation`` and ``rotation`` — a full camera
    model, or :data:`IDENTITY_CAMERA_POSE` when nothing is known. A reconstructed
    video publishes the camera's real height and orientation above the fitted ground
    plane; everything else publishes the identity.
    """
    translation = [float(value) for value in pose["translation"]]
    rotation = [float(value) for value in pose["rotation"]]
    return _encode(
        {
            "transforms": [
                {
                    "timestamp": ns_to_time(ns),
                    "parent_frame_id": MAP_FRAME_ID,
                    "child_frame_id": child_frame_id,
                    "translation": {"x": translation[0], "y": translation[1], "z": translation[2]},
                    "rotation": {
                        "x": rotation[0],
                        "y": rotation[1],
                        "z": rotation[2],
                        "w": rotation[3],
                    },
                }
            ]
        }
    )


# foxglove.PackedElementField numeric type for float32.
_PACKED_FLOAT32 = 7
# x, y, z, red, green, blue, alpha -- 7 float32 values per point.
POINT_CLOUD_STRIDE_BYTES = 28
_POINT_CLOUD_FIELDS: list[dict[str, Any]] = [
    {"name": "x", "offset": 0, "type": _PACKED_FLOAT32},
    {"name": "y", "offset": 4, "type": _PACKED_FLOAT32},
    {"name": "z", "offset": 8, "type": _PACKED_FLOAT32},
    {"name": "red", "offset": 12, "type": _PACKED_FLOAT32},
    {"name": "green", "offset": 16, "type": _PACKED_FLOAT32},
    {"name": "blue", "offset": 20, "type": _PACKED_FLOAT32},
    {"name": "alpha", "offset": 24, "type": _PACKED_FLOAT32},
]
_IDENTITY_POSE: dict[str, Any] = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}


def point_cloud_message(ns: int, frame_id: str, packed: npt.NDArray[np.float32]) -> bytes:
    """Build a ``foxglove.PointCloud`` JSON payload from a packed ``(N, 7)`` array.

    Colours are separate float32 red/green/blue/alpha channels in ``0..1``, which
    Foxglove's "RGBA (separate fields)" colour mode reads unambiguously and
    auto-selects, so the 3D panel needs no manual setup. The array is already in
    the wire layout (see ``scene3d.lifting.pack_point_cloud``), so this only
    base64-encodes it.
    """
    # `pack_point_cloud` already yields C-contiguous float32, so encode its buffer directly.
    raw = np.ascontiguousarray(packed, dtype="<f4")
    return _encode(
        {
            "timestamp": ns_to_time(ns),
            "frame_id": frame_id,
            "pose": _IDENTITY_POSE,
            "point_stride": POINT_CLOUD_STRIDE_BYTES,
            "fields": _POINT_CLOUD_FIELDS,
            "data": base64.b64encode(raw.data).decode("ascii"),
        }
    )


def _pose(position: Sequence[float], yaw: float = 0.0) -> dict[str, Any]:
    """Build a ``foxglove.Pose`` with a rotation about the map z-axis only."""
    return {
        "position": {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])},
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": float(math.sin(yaw / 2.0)),
            "w": float(math.cos(yaw / 2.0)),
        },
    }


def _color(rgba: Sequence[float]) -> dict[str, float]:
    """Build a ``foxglove.Color`` from an ``(r, g, b, a)`` sequence."""
    return {"r": float(rgba[0]), "g": float(rgba[1]), "b": float(rgba[2]), "a": float(rgba[3])}


def _scene_entity(ns: int, frame_id: str, entity: dict[str, Any]) -> dict[str, Any]:
    """Render one ``scene3d`` cuboid record as a ``foxglove.SceneEntity``.

    Every primitive array is emitted even when empty: some Foxglove builds reject
    an entity with missing primitive keys.
    """
    cube = entity["cube"]
    cubes = [
        {
            "pose": _pose(cube["position"], float(cube["yaw"])),
            "size": {
                "x": float(cube["size"][0]),
                "y": float(cube["size"][1]),
                "z": float(cube["size"][2]),
            },
            "color": _color(cube["color"]),
        }
    ]
    arrows: list[dict[str, Any]] = []
    arrow = entity.get("arrow")
    if arrow is not None:
        length = float(arrow["length"])
        arrows.append(
            {
                "pose": _pose(arrow["position"], float(arrow["yaw"])),
                "shaft_length": length * 0.7,
                "shaft_diameter": 0.5,
                "head_length": length * 0.3,
                "head_diameter": 1.0,
                "color": _color(arrow["color"]),
            }
        )
    texts: list[dict[str, Any]] = []
    text = entity.get("text")
    if text is not None:
        texts.append(
            {
                "pose": _pose(text["position"]),
                "billboard": True,
                "font_size": float(text["font_size"]),
                "scale_invariant": False,
                "color": {"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0},
                "text": str(text["value"]),
            }
        )
    lifetime_ns = int(entity["lifetime_ns"])
    return {
        "timestamp": ns_to_time(ns),
        "frame_id": frame_id,
        "id": str(entity["id"]),
        "lifetime": ns_to_time(lifetime_ns),
        "frame_locked": False,
        "metadata": [{"key": key, "value": str(value)} for key, value in entity["metadata"].items()],
        "arrows": arrows,
        "cubes": cubes,
        "spheres": [],
        "cylinders": [],
        "lines": [],
        "triangles": [],
        "texts": texts,
        "models": [],
    }


def scene_update_message(ns: int, frame_id: str, entities: list[dict[str, Any]]) -> bytes:
    """Build a ``foxglove.SceneUpdate`` JSON payload from ``scene3d`` cuboid records."""
    return _encode(
        {
            "deletions": [],
            "entities": [_scene_entity(ns, frame_id, entity) for entity in entities],
        }
    )
