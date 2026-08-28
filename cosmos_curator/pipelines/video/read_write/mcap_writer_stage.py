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
"""MCAP writer stage and post-pipeline consolidation for the split pipeline.

``McapWriterStage`` runs right after ``ClipWriterStage`` and writes one MCAP
*fragment* per (video, clip-chunk) to
``<output>/mcap_fragments/<relative_input_path>/<chunk_index>.mcap``. Because
``ClipTranscodingStage`` re-chunks tasks, no single stage invocation sees all
clips of one input video; ``consolidate_mcap_fragments`` runs on the driver
after the pipeline finishes and merges each video's fragments into one final
``<output>/mcap/<relative_input_path>.mcap``.

Timestamps are 0-based source-video offsets: each message is logged at
``clip.start_ns`` plus its offset within the clip, so gaps between clips on
the source timeline remain gaps in the MCAP. Video messages are written in
demux (decode) order, which readers like Foxglove require; with B-frames the
per-message ``log_time`` can therefore be locally non-monotonic, which MCAP
permits.
"""

import functools
import io
import json
import pathlib
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from importlib import metadata as importlib_metadata
from typing import IO, Any

import av
import smart_open  # type: ignore[import-untyped]
from av.bitstream import BitStreamFilterContext
from loguru import logger
from mcap.reader import McapReader, make_reader
from mcap.writer import CompressionType, Writer

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.sensors.utils.video import pts_to_ns
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.core.utils.storage import storage_client, storage_utils
from cosmos_curator.core.utils.storage.storage_utils import (
    StorageWriter,
    get_files_relative,
    get_full_path,
    read_bytes,
)
from cosmos_curator.pipelines.video.read_write import mcap_schemas
from cosmos_curator.pipelines.video.read_write.metadata_writer_stage import (
    ClipWriterStage,
    drop_clip_intermediate_data,
    select_clip_embedding,
    window_ns_bounds,
)
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, Window
from cosmos_curator.pipelines.video.utils.ns_timing import seconds_to_ns

MCAP_LIBRARY = "cosmos-curator split-pipeline mcap-writer"
DEFAULT_FRAME_ID = "camera"

# PyAV codec name -> foxglove.CompressedVideo ``format`` value. h264/hevc additionally
# need an AVCC -> Annex-B bitstream filter; av1/vp9 packets are already in the
# low-overhead form Foxglove expects.
_FOXGLOVE_VIDEO_FORMATS = {"h264": "h264", "hevc": "h265", "av1": "av1", "vp9": "vp9"}
_ANNEXB_BSF_NAMES = {"h264": "h264_mp4toannexb", "hevc": "hevc_mp4toannexb"}


@functools.cache
def _curator_version() -> str:
    try:
        return importlib_metadata.version("cosmos_curator")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


@contextmanager
def _open_mcap_writer(out_file: IO[bytes]) -> Iterator[Writer]:
    """Yield a started zstd-chunked MCAP writer, finishing (sealing) it only on success.

    On error the file is deliberately left without a valid MCAP footer, so a
    partial write can never be mistaken for a complete file downstream.
    """
    writer = Writer(out_file, compression=CompressionType.ZSTD)
    writer.start(library=MCAP_LIBRARY)
    yield writer
    writer.finish()  # type: ignore[no-untyped-call]


class _McapChannels:
    """Lazily register channels on first message and track per-channel sequences.

    Topic -> schema is the fixed 1:1 map in ``mcap_schemas.TOPIC_SCHEMAS``.
    """

    def __init__(self, writer: Writer) -> None:
        self._writer = writer
        self._channel_ids: dict[str, int] = {}
        self._sequences: dict[str, int] = {}

    def add_message(self, topic: str, log_time_ns: int, payload: bytes) -> None:
        if topic not in self._channel_ids:
            schema = mcap_schemas.TOPIC_SCHEMAS[topic]
            schema_id = self._writer.register_schema(
                name=schema["title"],
                encoding=mcap_schemas.JSONSCHEMA_ENCODING,
                data=json.dumps(schema).encode("utf-8"),
            )
            self._channel_ids[topic] = self._writer.register_channel(
                schema_id=schema_id,
                topic=topic,
                message_encoding=mcap_schemas.JSON_MESSAGE_ENCODING,
            )
        sequence = self._sequences.get(topic, 0)
        self._sequences[topic] = sequence + 1
        self._writer.add_message(
            channel_id=self._channel_ids[topic],
            log_time=log_time_ns,
            data=payload,
            publish_time=log_time_ns,
            sequence=sequence,
        )


class McapWriterStage(CuratorStage):
    """Write one MCAP fragment per (video, clip-chunk) with clip media and annotations.

    Runs after ``ClipWriterStage`` (constructed with ``retain_clip_data=True`` so the
    small annotations — captions and embeddings — survive that stage's cleanup;
    ``build_output_stages`` wires this pairing). Clip mp4 bytes are deliberately NOT
    retained: this stage reads each clip back from the just-written ``clips/`` output
    (a page-cache hit locally), so the feature adds no clip payloads to the Ray
    object store and each worker holds at most one clip in memory at a time.
    """

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        input_path: str,
        output_s3_profile_name: str,
        *,
        embedding_algorithm: str,
        embedding_model_version: str,
        caption_models: list[str],
        dry_run: bool = False,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Construct the MCAP fragment writer stage."""
        self._timer = StageTimer(self)
        self._output_path = output_path
        self._input_path = input_path.rstrip("/") + "/"
        self._output_s3_profile_name = output_s3_profile_name
        self._embedding_algorithm = embedding_algorithm
        self._embedding_model_version = embedding_model_version
        self._caption_models = caption_models
        self._dry_run = dry_run
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage."""
        return CuratorStageResource(cpus=0.5)

    def stage_setup(self) -> None:
        """Initialize the fragment storage writer and the clip read client."""
        self._fragments_writer = StorageWriter(
            ClipWriterStage.get_output_path_mcap_fragments(self._output_path),
            profile_name=self._output_s3_profile_name,
        )
        self._storage_client = storage_utils.get_storage_client(
            self._output_path,
            profile_name=self._output_s3_profile_name,
        )

    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:  # type: ignore[override]
        """Write one MCAP fragment per video chunk, then drop retained clip payloads.

        Exceptions propagate (like ``ClipWriterStage``) so Xenna's run attempts can
        retry the task; the retained payloads are dropped only once every video's
        fragment is durably written, keeping retries reproducible.
        """
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            for video in task.videos:
                with self._timer.time_process(len(video.clips)):
                    self._write_video_fragment(video)
            for video in task.videos:
                for clip in video.clips:
                    drop_clip_intermediate_data(clip)
            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats
        return tasks

    def _relative_video_path(self, video: Video) -> str:
        input_video_path = video.input_path
        assert input_video_path.startswith(self._input_path)
        return input_video_path[len(self._input_path) :]

    def _write_video_fragment(self, video: Video) -> None:
        # A fully filtered / zero-clip chunk 0 still writes a metadata-only fragment so
        # every processed video yields a final MCAP; other empty chunks write nothing.
        if not video.clips and video.clip_chunk_index != 0:
            return
        if self._dry_run:
            logger.info(f"Dry-run: skipping MCAP fragment for {video.input_path} chunk {video.clip_chunk_index}")
            return
        sub_path = f"{self._relative_video_path(video)}/{video.clip_chunk_index}.mcap"
        with self._fragments_writer.open_writer(sub_path, mode="wb") as out_file:
            self._write_chunk_mcap(out_file, video)
        if self._verbose:
            logger.info(f"Wrote MCAP fragment {sub_path} for {video.input_path}")

    def _write_chunk_mcap(self, out_file: IO[bytes], video: Video) -> None:
        """Write one video chunk's clips, annotations, and embeddings as a complete MCAP file.

        Annotations and embeddings are written for every camera video: fragment paths
        are per-camera so nothing can collide, secondary cameras simply lack captions
        and embeddings (lazy channel registration omits those channels), and SAM3
        detections genuinely exist per camera.
        """
        with _open_mcap_writer(out_file) as writer:
            channels = _McapChannels(writer)
            calibration = _video_scene3d_calibration(video)
            if video.clip_chunk_index == 0:
                self._write_session_start(writer, channels, video, calibration)
            elif calibration is not None:
                # Chunks are written independently, so a later chunk cannot know whether
                # chunk 0 had a reconstruction to publish. Re-emitting the calibration
                # keeps this chunk's 3D geometry anchored to a real map frame even when
                # every clip in chunk 0 failed and published a placeholder.
                #
                # Stamped at this chunk's first clip rather than 0: the merged file would
                # otherwise carry two contradictory models at the same instant on a
                # latched topic, and which one a viewer picks would be undefined. A later
                # timestamp makes the measured model deterministically supersede the
                # placeholder from the point its clips begin.
                self._write_calibration(channels, video, calibration, log_time=_chunk_base_ns(video))
            for clip in video.clips:
                base_ns = _clip_base_ns(clip)
                media = self._read_clip_mp4(clip, video.relative_path)
                if media is not None:
                    _write_clip_media(channels, clip, base_ns, media)
                self._write_clip_annotations(channels, clip, base_ns)
                self._write_clip_embedding(channels, clip, base_ns)
                _write_clip_scene3d(channels, clip, base_ns)

    def _read_clip_mp4(self, clip: Clip, relative_path: str) -> bytes | None:
        """Read one clip's mp4 back from the ``clips/`` output written by ClipWriterStage."""
        clip_uri = ClipWriterStage.get_clip_mp4_uri(self._output_path, clip.uuid, relative_path)
        try:
            return read_bytes(clip_uri, self._storage_client)
        except FileNotFoundError:
            # Mirrors the pre-existing "clip has no data" path: the writer already
            # logged why the mp4 is missing; the MCAP keeps annotations only.
            logger.warning(f"Clip {clip.uuid} from {clip.source_video} has no written mp4; skipping MCAP media")
            return None

    def _session_metadata(self, video: Video, calibration: dict[str, Any] | None) -> dict[str, str]:
        meta = video.metadata
        values: dict[str, Any] = {
            "source-video": video.input_path,
            "video-uuid": str(ClipWriterStage.get_video_uuid(video.input_path)),
            "width": meta.width,
            "height": meta.height,
            "framerate": meta.framerate,
            "num-frames": meta.num_frames,
            "duration-s": meta.duration,
            "video-codec": meta.video_codec,
            "pixel-format": meta.pixel_format,
            "audio-codec": meta.audio_codec,
            "num-total-clips": video.num_total_clips,
            "num-clip-chunks": video.num_clip_chunks,
            "embedding-algorithm": self._embedding_algorithm,
            "embedding-model-version": self._embedding_model_version,
            "curator-version": _curator_version(),
        }
        if calibration is not None:
            values["scene3d-camera-height-m"] = round(float(calibration["translation"][2]), 4)
            values["scene3d-focal-px"] = round(float(calibration["K"][0]), 4)
            values["scene3d-calibration-source"] = str(calibration.get("source", "unknown"))
            values["scene3d-ground-inlier-frac"] = round(float(calibration.get("ground_inlier_frac", 0.0)), 4)
        return {key: str(value) for key, value in values.items() if value is not None}

    def _write_session_start(
        self,
        writer: Writer,
        channels: _McapChannels,
        video: Video,
        calibration: dict[str, Any] | None,
    ) -> None:
        """Write the one-shot records: session metadata, camera calibration, static transform.

        Emitted at log_time 0 (the source-video start on the 0-based timeline),
        from chunk 0 only so the merged file carries the metadata exactly once.
        """
        writer.add_metadata(mcap_schemas.SESSION_METADATA_RECORD_NAME, self._session_metadata(video, calibration))
        self._write_calibration(channels, video, calibration)

    def _write_calibration(
        self,
        channels: _McapChannels,
        video: Video,
        calibration: dict[str, Any] | None,
        *,
        log_time: int = 0,
    ) -> None:
        """Write ``/camera/camera-info`` and ``/tf-static``.

        Falls back to a placeholder camera model when the video was not reconstructed,
        so both records describe the same camera. Intrinsics need a frame size, so a
        video with neither an estimate nor known dimensions publishes no calibration
        rather than one full of zeroes — but the transform does not depend on frame
        size and is always published, since without it Foxglove cannot place the image
        stream in the 3D panel at all.
        """
        model = calibration
        if model is None and video.metadata.width and video.metadata.height:
            model = mcap_schemas.placeholder_camera_model(video.metadata.width, video.metadata.height)
        if model is not None:
            channels.add_message(
                mcap_schemas.TOPIC_CAMERA_INFO,
                log_time,
                mcap_schemas.camera_calibration_message(log_time, DEFAULT_FRAME_ID, model),
            )
        channels.add_message(
            mcap_schemas.TOPIC_TF_STATIC,
            log_time,
            mcap_schemas.frame_transforms_message(
                log_time, DEFAULT_FRAME_ID, model or mcap_schemas.IDENTITY_CAMERA_POSE
            ),
        )

    def _write_clip_annotations(self, channels: _McapChannels, clip: Clip, base_ns: int) -> None:
        """Write window captions and SAM3 per-frame detections on ``/scene-annotation``.

        Mirrors the reference recordings, where plain-text scene descriptions and
        JSON-encoded detection payloads share one topic.
        """
        for window in clip.windows:
            caption = _select_window_caption(window, self._caption_models)
            if caption is None:
                continue
            window_start_ns, _ = window_ns_bounds(clip, window)
            log_time = base_ns + (window_start_ns if window_start_ns is not None else 0)
            channels.add_message(
                mcap_schemas.TOPIC_SCENE_ANNOTATION,
                log_time,
                mcap_schemas.scene_annotation_message(log_time, caption),
            )
        for entry in clip.sam3_frames or []:
            timestamp_s = entry.get("timestamp_s")
            offset_ns = seconds_to_ns(float(timestamp_s)) if timestamp_s is not None else 0
            log_time = base_ns + offset_ns
            payload = json.dumps({"frame_idx": entry.get("frame_idx"), "detections": entry.get("detections", [])})
            channels.add_message(
                mcap_schemas.TOPIC_SCENE_ANNOTATION,
                log_time,
                mcap_schemas.scene_annotation_message(log_time, payload),
            )

    def _write_clip_embedding(self, channels: _McapChannels, clip: Clip, base_ns: int) -> None:
        embedding = select_clip_embedding(clip, self._embedding_algorithm)
        if embedding is None:
            return
        channels.add_message(
            mcap_schemas.TOPIC_CLIP_EMBEDDING,
            base_ns,
            mcap_schemas.clip_embedding_message(
                base_ns, self._embedding_algorithm, self._embedding_model_version, embedding
            ),
        )


def _chunk_base_ns(video: Video) -> int:
    """Source-timeline start of a chunk: the earliest base of the clips it holds."""
    return min((_clip_base_ns(clip) for clip in video.clips), default=0)


def _video_scene3d_calibration(video: Video) -> dict[str, Any] | None:
    """Return the video's 3D calibration, taken from its first reconstructed clip.

    One transform per video is the right granularity: ``/tf-static`` and
    ``/camera/camera-info`` are session-level records written once from chunk 0,
    and a clip that fails reconstruction should not leave the file without a
    camera model.
    """
    for clip in video.clips:
        calibration: dict[str, Any] | None = clip.scene3d_calibration
        if calibration is not None:
            return calibration
    return None


def _write_clip_scene3d(channels: _McapChannels, clip: Clip, base_ns: int) -> None:
    """Write the clip's 3D background cloud and per-frame object cuboids.

    The cloud is emitted once at the clip start; Foxglove's 3D panel shows the
    most recent one, so the backdrop follows clip boundaries. Cuboid timestamps use
    the same clip-relative arithmetic as the SAM3 annotations, keeping every
    per-frame topic on one timeline.
    """
    background = clip.scene3d_background.resolve()
    if background is not None and background.size:
        channels.add_message(
            mcap_schemas.TOPIC_SCENE_BACKGROUND,
            base_ns,
            mcap_schemas.point_cloud_message(base_ns, mcap_schemas.MAP_FRAME_ID, background),
        )
    for record in clip.scene3d_objects or []:
        entities = record.get("entities") or []
        if not entities:
            continue
        timestamp_s = record.get("timestamp_s")
        offset_ns = seconds_to_ns(float(timestamp_s)) if timestamp_s is not None else 0
        log_time = base_ns + offset_ns
        channels.add_message(
            mcap_schemas.TOPIC_SCENE_OBJECTS,
            log_time,
            mcap_schemas.scene_update_message(log_time, mcap_schemas.MAP_FRAME_ID, entities),
        )


def _clip_base_ns(clip: Clip) -> int:
    """Source-timeline start of the clip in ns (span-seconds fallback for errored clips)."""
    if clip.start_ns is not None:
        return clip.start_ns
    return seconds_to_ns(clip.span[0])


def _select_window_caption(window: Window, caption_models: list[str]) -> str | None:
    for model in caption_models:
        if model in window.caption:
            return window.caption[model]
    return None


def _write_clip_media(channels: _McapChannels, clip: Clip, base_ns: int, media: bytes) -> None:
    """Demux one clip mp4 and write its video packets and decoded audio blocks."""
    with av.open(io.BytesIO(media), mode="r") as container:
        maybe_writers = (
            _ClipVideoWriter.create(channels, container, clip, base_ns),
            _ClipAudioWriter.create(channels, container, base_ns),
        )
        writers = {w.stream_index: w for w in maybe_writers if w is not None}
        if not writers:
            return
        for packet in container.demux():
            if packet.dts is None:  # demux flush sentinel
                continue
            if (writer := writers.get(packet.stream_index)) is not None:
                writer.write_packet(packet)
        for writer in writers.values():
            writer.flush()


class _ClipVideoWriter:
    """Write a clip's compressed video packets as ``foxglove.CompressedVideo`` messages.

    h264/hevc packets are converted from AVCC (length-prefixed NALs, mp4) to
    Annex-B (start codes, SPS/PPS inline) via the ``*_mp4toannexb`` bitstream
    filter, as required by the CompressedVideo spec.
    """

    def __init__(  # noqa: PLR0913
        self,
        channels: _McapChannels,
        clip_uuid: uuid.UUID,
        base_ns: int,
        *,
        video_format: str,
        stream_index: int,
        bsf: BitStreamFilterContext | None,
    ) -> None:
        self._channels = channels
        self._clip_uuid = clip_uuid
        self._base_ns = base_ns
        self._video_format = video_format
        self.stream_index = stream_index
        self._bsf = bsf
        # Seeded from the first demuxed packet (the IDR frame): its PTS is the clip's zero point.
        self._first_pts_ns: int | None = None
        self._warned_missing_pts = False

    @classmethod
    def create(
        cls,
        channels: _McapChannels,
        container: "av.container.InputContainer",
        clip: Clip,
        base_ns: int,
    ) -> "_ClipVideoWriter | None":
        if not container.streams.video:
            logger.warning(f"Clip {clip.uuid} from {clip.source_video} has no video stream")
            return None
        stream = container.streams.video[0]
        codec_name = stream.codec_context.name
        video_format = _FOXGLOVE_VIDEO_FORMATS.get(codec_name)
        if video_format is None:
            logger.warning(
                f"Clip {clip.uuid} from {clip.source_video} uses codec {codec_name!r}, which "
                "foxglove.CompressedVideo does not support; skipping video frames"
            )
            return None
        bsf_name = _ANNEXB_BSF_NAMES.get(codec_name)
        bsf = BitStreamFilterContext(bsf_name, stream) if bsf_name is not None else None
        return cls(
            channels,
            clip.uuid,
            base_ns,
            video_format=video_format,
            stream_index=stream.index,
            bsf=bsf,
        )

    def write_packet(self, packet: "av.Packet[Any]") -> None:
        out_packets = self._bsf.filter(packet) if self._bsf is not None else [packet]
        for out_packet in out_packets:
            self._write_filtered_packet(out_packet)

    def flush(self) -> None:
        if self._bsf is None:
            return
        for out_packet in self._bsf.filter(None):
            self._write_filtered_packet(out_packet)

    def _write_filtered_packet(self, packet: "av.Packet[Any]") -> None:
        if packet.pts is None or packet.time_base is None:
            if not self._warned_missing_pts:
                self._warned_missing_pts = True
                logger.warning(f"Clip {self._clip_uuid} has video packets without PTS; skipping those frames")
            return
        packet_pts_ns = pts_to_ns(packet.pts, packet.time_base)
        if self._first_pts_ns is None:
            self._first_pts_ns = packet_pts_ns
        log_time = self._base_ns + (packet_pts_ns - self._first_pts_ns)
        self._channels.add_message(
            mcap_schemas.TOPIC_IMAGE_RAW,
            log_time,
            mcap_schemas.compressed_video_message(log_time, DEFAULT_FRAME_ID, packet, self._video_format),
        )


class _ClipAudioWriter:
    """Decode a clip's audio track and write it as pcm-s16 ``foxglove.RawAudio`` messages."""

    def __init__(self, channels: _McapChannels, stream: "av.audio.stream.AudioStream", base_ns: int) -> None:
        self._channels = channels
        self._stream = stream
        self.stream_index = stream.index
        self._base_ns = base_ns
        self._resampler: av.AudioResampler | None = None
        self._elapsed_ns = 0

    @classmethod
    def create(
        cls,
        channels: _McapChannels,
        container: "av.container.InputContainer",
        base_ns: int,
    ) -> "_ClipAudioWriter | None":
        if not container.streams.audio:
            return None
        return cls(channels, container.streams.audio[0], base_ns)

    def write_packet(self, packet: "av.Packet[Any]") -> None:
        for frame in self._stream.codec_context.decode(packet):
            self._write_frame(frame)

    def flush(self) -> None:
        for frame in self._stream.codec_context.decode(None):
            self._write_frame(frame)
        if self._resampler is not None:
            for resampled in self._resampler.resample(None):
                self._write_resampled(resampled)

    def _write_frame(self, frame: "av.AudioFrame") -> None:
        if self._resampler is None:
            self._resampler = av.AudioResampler(format="s16", layout=frame.layout.name, rate=frame.sample_rate)
        for resampled in self._resampler.resample(frame):
            self._write_resampled(resampled)

    def _write_resampled(self, frame: "av.AudioFrame") -> None:
        data = frame.to_ndarray().tobytes()
        if not data:
            return
        number_of_channels = len(frame.layout.channels)
        # Resampled PCM output is contiguous, so a running sample clock is the single
        # source of truth; decoder PTS gaps or absences cannot rewind or overlap blocks.
        log_time = self._base_ns + self._elapsed_ns
        self._elapsed_ns += seconds_to_ns(frame.samples / frame.sample_rate)
        self._channels.add_message(
            mcap_schemas.TOPIC_AUDIO_RAW,
            log_time,
            mcap_schemas.raw_audio_message(log_time, data, frame.sample_rate, number_of_channels),
        )


def consolidate_mcap_fragments(output_path: str, output_s3_profile_name: str) -> None:
    """Merge per-chunk MCAP fragments into one MCAP per input video and delete the fragments."""
    fragments_root = ClipWriterStage.get_output_path_mcap_fragments(output_path)
    client = storage_utils.get_storage_client(
        fragments_root,
        profile_name=output_s3_profile_name,
        can_overwrite=True,
        can_delete=True,
    )

    # Fragments live at <relative_input_path>/<chunk_index>.mcap; group per video.
    # No existence pre-check: a missing root simply lists as empty (a head_object on
    # a bare remote prefix would 404 even when fragments exist under it).
    fragments_by_video: dict[str, list[tuple[int, str]]] = {}
    for fname in get_files_relative(fragments_root, client):
        relative_path, _, chunk_name = fname.rpartition("/")
        if not fname.endswith(".mcap") or not relative_path:
            continue
        try:
            chunk_index = int(chunk_name.removesuffix(".mcap"))
        except ValueError:
            logger.warning(f"Unexpected MCAP fragment name {fname!r}; skipping")
            continue
        fragments_by_video.setdefault(relative_path, []).append((chunk_index, fname))
    if not fragments_by_video:
        return

    final_writer = StorageWriter(
        ClipWriterStage.get_output_path_mcap(output_path),
        profile_name=output_s3_profile_name,
    )

    consolidated = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _consolidate_video_fragments,
                relative_path,
                sorted(entries),
                fragments_root=fragments_root,
                client=client,
                final_writer=final_writer,
            ): relative_path
            for relative_path, entries in fragments_by_video.items()
        }
        for future, relative_path in futures.items():
            try:
                consolidated += 1 if future.result() else 0
            except Exception:  # noqa: BLE001 - collected and re-raised as one error below
                logger.exception(f"Failed to consolidate MCAP fragments for {relative_path}")
                failed.append(relative_path)
    logger.info(f"Consolidated MCAP fragments for {consolidated}/{len(fragments_by_video)} videos")
    if failed:
        msg = f"MCAP consolidation failed for {len(failed)} of {len(fragments_by_video)} videos: {failed[:10]}"
        raise RuntimeError(msg)


def _expected_chunk_count(chunk0_reader: McapReader) -> int | None:
    """Read the expected chunk count from the chunk-0 fragment's session metadata, if present."""
    for record in chunk0_reader.iter_metadata():
        if record.name == mcap_schemas.SESSION_METADATA_RECORD_NAME:
            value = record.metadata.get("num-clip-chunks")
            return int(value) if value is not None else None
    return None


def _consolidate_video_fragments(
    relative_path: str,
    entries: list[tuple[int, str]],
    *,
    fragments_root: str,
    client: "storage_client.StorageClient | None",
    final_writer: StorageWriter,
) -> bool:
    """Merge one video's fragments into its final MCAP; returns False for an incomplete set.

    Incomplete sets (missing chunk 0, gaps, or fewer chunks than the count recorded in
    chunk 0's session metadata — e.g. leftovers of an interrupted run) are skipped with
    their fragments preserved, so a future complete run can still consolidate them.
    Corrupt/truncated fragments raise before the final file is created.
    """
    chunk_indices = [chunk_index for chunk_index, _ in entries]
    fragment_uris = [get_full_path(fragments_root, fname) for _, fname in entries]

    with ExitStack() as stack:
        # Open (and for remote, download) every fragment once, validating each via the
        # MCAP reader up front so a truncated leftover can never ship as a final file.
        readers = [make_reader(stack.enter_context(_open_fragment(uri, client))) for uri in fragment_uris]  # type: ignore[no-untyped-call]
        for reader in readers:
            reader.get_summary()

        expected_chunks = _expected_chunk_count(readers[0]) if chunk_indices[0] == 0 else None
        complete = (
            chunk_indices == list(range(expected_chunks))
            if expected_chunks is not None
            else chunk_indices == list(range(len(chunk_indices)))
        )
        if not complete:
            logger.warning(
                f"Video {relative_path} has an incomplete MCAP fragment set (chunks {chunk_indices}, "
                f"expected {expected_chunks}); leaving fragments in place"
            )
            return False

        with final_writer.open_writer(f"{relative_path}.mcap", mode="wb") as out_file:
            _merge_fragments(out_file, list(zip(fragment_uris, readers, strict=True)))

    for fragment_uri in fragment_uris:
        if isinstance(fragment_uri, pathlib.Path):
            fragment_uri.unlink(missing_ok=True)
        elif client is not None:
            try:
                client.delete_object(fragment_uri)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to delete MCAP fragment {fragment_uri}: {exc}")
    return True


def _download_fragment_to(
    uri: "storage_client.StoragePrefix",
    client: "storage_client.StorageClient | None",
    out_file: IO[bytes],
) -> None:
    """Stream a remote fragment into *out_file* with constant memory, retrying transient errors."""
    client_params = storage_utils.get_smart_open_client_params(client) if client is not None else {}

    def _download() -> None:
        out_file.seek(0)
        out_file.truncate()
        with smart_open.open(str(uri), "rb", **client_params) as src:
            shutil.copyfileobj(src, out_file)

    # Same retry policy as storage_utils.read_bytes.
    do_with_retries(_download, max_attempts=5, backoff_factor=4.0, max_wait_time_s=256.0)


@contextmanager
def _open_fragment(
    uri: "storage_client.StoragePrefix | pathlib.Path",
    client: "storage_client.StorageClient | None",
) -> Iterator[IO[bytes]]:
    """Yield a seekable binary handle: the file itself locally, a temp-file download for remote."""
    if isinstance(uri, pathlib.Path):
        with uri.open("rb") as fh:
            yield fh
    else:
        with tempfile.TemporaryFile(suffix=".mcap") as tmp_file:
            _download_fragment_to(uri, client, tmp_file)
            tmp_file.seek(0)
            yield tmp_file


def _merge_fragments(
    out_file: IO[bytes],
    fragments: list[tuple["storage_client.StoragePrefix | pathlib.Path", McapReader]],
) -> None:
    """Rewrite the fragments' records into one MCAP, remapping schema/channel ids.

    Fragments are visited in chunk order and their messages are copied in file
    order (NOT log_time order), preserving the decode-order guarantee for video
    packets; since chunks partition the source timeline in order, per-topic
    log_time stays globally ascending up to B-frame jitter within a chunk.
    """
    with _open_mcap_writer(out_file) as writer:
        schema_ids: dict[tuple[str, str, bytes], int] = {}
        channel_ids: dict[tuple[str, str, str], int] = {}
        # Fragments restart sequences at 0, so renumber per channel across the merge.
        sequences: dict[int, int] = {}
        metadata_seen: set[str] = set()
        for fragment_uri, reader in fragments:
            for metadata_record in reader.iter_metadata():
                if metadata_record.name in metadata_seen:
                    continue
                metadata_seen.add(metadata_record.name)
                writer.add_metadata(metadata_record.name, dict(metadata_record.metadata))
            for schema, channel, message in reader.iter_messages(log_time_order=False):
                if schema is None:
                    logger.warning(f"Fragment {fragment_uri} has a channel without schema; skipping its messages")
                    continue
                schema_key = (schema.name, schema.encoding, schema.data)
                if schema_key not in schema_ids:
                    schema_ids[schema_key] = writer.register_schema(
                        name=schema.name,
                        encoding=schema.encoding,
                        data=schema.data,
                    )
                channel_key = (channel.topic, schema.name, channel.message_encoding)
                if channel_key not in channel_ids:
                    channel_ids[channel_key] = writer.register_channel(
                        schema_id=schema_ids[schema_key],
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        metadata=dict(channel.metadata),
                    )
                channel_id = channel_ids[channel_key]
                sequence = sequences.get(channel_id, 0)
                sequences[channel_id] = sequence + 1
                writer.add_message(
                    channel_id=channel_id,
                    log_time=message.log_time,
                    data=message.data,
                    publish_time=message.publish_time,
                    sequence=sequence,
                )
