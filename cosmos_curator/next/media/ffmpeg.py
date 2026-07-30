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

"""Typed FFprobe and FFmpeg contracts for fixed-stride video clips."""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from cosmos_curator.next.media.spans import Span, nanoseconds_to_ffmpeg_timestamp, seconds_to_nanoseconds


class TranscodeSettings(Protocol):
    """Structural settings required by the reusable FFmpeg transcoder."""

    @property
    def video_encoder(self) -> str:
        """Return the FFmpeg video encoder name."""
        ...

    @property
    def video_bitrate(self) -> str:
        """Return the FFmpeg target video bitrate."""
        ...

    @property
    def audio_mode(self) -> str:
        """Return the optional audio stream handling mode."""
        ...

    @property
    def encoder_threads(self) -> int:
        """Return the video encoder thread count."""
        ...


@dataclass(frozen=True)
class VideoMetadata:
    """Media fields needed by the v1 source and clip outcome contract."""

    duration_ns: int
    width: int
    height: int
    frame_rate: float
    frame_count: int | None
    video_codec: str


class TranscodeError(RuntimeError):
    """Raised when FFmpeg cannot produce one planned clip."""


class ProbeError(RuntimeError):
    """Raised when FFprobe cannot inspect a source or transcoded clip."""


def probe_video_bytes(video_bytes: bytes) -> VideoMetadata:
    """Probe the first video stream in an in-memory media object."""
    with tempfile.TemporaryDirectory(prefix="curator_next_probe_") as tmp_dir:
        path = Path(tmp_dir) / "media"
        path.write_bytes(video_bytes)
        return probe_video_path(path)


def probe_video_path(path: Path) -> VideoMetadata:
    """Probe a local media path with FFprobe and integer-nanosecond duration conversion."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, timeout=120)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        msg = f"FFprobe timed out after {exc.timeout}s on {path}"
        raise ProbeError(msg) from exc
    except subprocess.CalledProcessError as exc:
        diagnostic = exc.stderr.decode("utf-8", errors="replace").strip()
        msg = diagnostic or f"FFprobe exited with status {exc.returncode}"
        raise ProbeError(msg) from exc
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        msg = "FFprobe returned a non-object payload"
        raise TypeError(msg)
    return _metadata_from_ffprobe(payload)


def transcode_span(source_bytes: bytes, span: Span, config: TranscodeSettings) -> bytes:
    """Transcode one logical span to an MP4 using the v1 stream-selection contract."""
    with tempfile.TemporaryDirectory(prefix="curator_next_transcode_") as tmp_dir:
        root = Path(tmp_dir)
        source_path = root / "source"
        clip_path = root / "clip.mp4"
        source_path.write_bytes(source_bytes)

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            nanoseconds_to_ffmpeg_timestamp(span.start_ns),
            "-i",
            str(source_path),
            "-t",
            nanoseconds_to_ffmpeg_timestamp(span.duration_ns),
            "-map",
            "0:v:0",
            "-c:v",
            config.video_encoder,
            "-b:v",
            config.video_bitrate,
            "-threads",
            str(config.encoder_threads),
            "-map",
            "0:a:0?",
            "-c:a",
            config.audio_mode,
            "-movflags",
            "+faststart",
            str(clip_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=120)  # noqa: S603
        except subprocess.TimeoutExpired as exc:
            msg = f"FFmpeg timed out after {exc.timeout}s on {source_path}"
            raise TranscodeError(msg) from exc
        except subprocess.CalledProcessError as exc:
            diagnostic = exc.stderr.decode("utf-8", errors="replace").strip()
            msg = diagnostic or f"FFmpeg exited with status {exc.returncode}"
            raise TranscodeError(msg) from exc
        if not clip_path.exists():
            msg = "FFmpeg completed without creating the expected MP4"
            raise TranscodeError(msg)
        return clip_path.read_bytes()


def _metadata_from_ffprobe(payload: dict[str, Any]) -> VideoMetadata:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        msg = "FFprobe payload does not contain a streams list"
        raise TypeError(msg)
    video_stream = next(
        (stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        msg = "No video stream found"
        raise ValueError(msg)

    format_payload = payload.get("format")
    format_duration = format_payload.get("duration") if isinstance(format_payload, dict) else None
    duration_ns = _first_valid_duration_ns(video_stream.get("duration"), format_duration)
    frame_rate = _first_valid_frame_rate(video_stream.get("avg_frame_rate"), video_stream.get("r_frame_rate"))
    raw_frame_count = video_stream.get("nb_frames")
    frame_count = (
        int(raw_frame_count)
        if isinstance(raw_frame_count, str) and raw_frame_count.isdigit()
        else round((duration_ns / 1_000_000_000) * frame_rate)
    )
    return VideoMetadata(
        duration_ns=duration_ns,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        frame_rate=frame_rate,
        frame_count=frame_count,
        video_codec=str(video_stream["codec_name"]),
    )


def _first_valid_duration_ns(*values: Any) -> int:  # noqa: ANN401
    for value in values:
        if value is None:
            continue
        try:
            duration_ns = seconds_to_nanoseconds(str(value))
        except (ArithmeticError, ValueError):
            continue
        if duration_ns >= 0:
            return duration_ns
    msg = "FFprobe did not report a valid source duration"
    raise ValueError(msg)


def _first_valid_frame_rate(*values: Any) -> float:  # noqa: ANN401
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            rate = Fraction(value)
        except (ValueError, ZeroDivisionError):
            continue
        if rate > 0:
            return float(rate)
    msg = "FFprobe did not report a valid positive frame rate"
    raise ValueError(msg)
