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
"""Tests for the MCAP writer stage and fragment consolidation."""

import base64
import functools
import io
import json
import uuid
from pathlib import Path

import av
import numpy as np
import numpy.testing as npt
import pytest
from mcap.reader import make_reader
from mcap.records import Channel, Message, Metadata, Schema

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.pipelines.video.read_write import mcap_schemas
from cosmos_curator.pipelines.video.read_write.mcap_writer_stage import (
    McapWriterStage,
    consolidate_mcap_fragments,
)
from cosmos_curator.pipelines.video.utils.data_model import (
    Clip,
    SplitPipeTask,
    Video,
    VideoMetadata,
    Window,
)
from cosmos_curator.pipelines.video.utils.ns_timing import NS_PER_SECOND

NUM_FRAMES = 10
FPS = 30
CLIP_DURATION_S = NUM_FRAMES / FPS


@functools.lru_cache
def _make_mp4(*, with_audio: bool = False) -> bytes:
    """Encode a tiny synthetic h264 mp4 (optionally with an aac audio track) in memory."""
    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=FPS)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    audio_stream = None
    if with_audio:
        audio_stream = container.add_stream("aac", rate=48000)
        audio_stream.layout = "mono"

    for i in range(NUM_FRAMES):
        array = np.full((stream.height, stream.width, 3), i * 10, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)

    if audio_stream is not None:
        for i in range(16):
            audio_frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
            for plane in audio_frame.planes:
                plane.update(b"\x00" * plane.buffer_size)
            audio_frame.sample_rate = 48000
            audio_frame.pts = i * 1024
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
        for packet in audio_stream.encode(None):
            container.mux(packet)

    container.close()
    return buffer.getvalue()


def _make_clip(
    video_path: Path,
    *,
    mp4_bytes: bytes,
    start_s: float = 0.0,
    with_annotations: bool = True,
) -> Clip:
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=video_path.as_posix(),
        span=(start_s, start_s + CLIP_DURATION_S),
        encoded_data=bytes_to_numpy(mp4_bytes),
        windows=[Window(start_frame=0, end_frame=NUM_FRAMES - 1, caption={"qwen": "a test scene"})]
        if with_annotations
        else [],
    )
    clip.pts_ns = (np.arange(NUM_FRAMES) * (NS_PER_SECOND // FPS)).astype(np.int64)
    clip.start_ns = round(start_s * NS_PER_SECOND)
    clip.end_ns = clip.start_ns + round(CLIP_DURATION_S * NS_PER_SECOND)
    if with_annotations:
        clip.intern_video_2_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        clip.sam3_frames = [
            {"frame_idx": 0, "timestamp_s": 0.0, "detections": [{"prompt": "car", "object_id": 1}]},
            {"frame_idx": 5, "timestamp_s": 5 / FPS, "detections": []},
        ]
    return clip


def _make_video(
    video_path: Path,
    clips: list[Clip],
    *,
    clip_chunk_index: int = 0,
    num_clip_chunks: int = 1,
) -> Video:
    return Video(
        input_video=video_path,
        metadata=VideoMetadata(
            height=16,
            width=16,
            framerate=float(FPS),
            num_frames=NUM_FRAMES,
            duration=CLIP_DURATION_S,
            video_codec="h264",
            pixel_format="yuv420p",
        ),
        clips=clips,
        filtered_clips=[],
        num_total_clips=len(clips),
        num_clip_chunks=num_clip_chunks,
        clip_chunk_index=clip_chunk_index,
    )


def _process(tmp_path: Path, *videos: Video) -> None:
    """Run a fresh stage over one task holding *videos* (first video is primary)."""
    stage = McapWriterStage(
        output_path=str(tmp_path / "output"),
        input_path=str(tmp_path / "input"),
        output_s3_profile_name="default",
        embedding_algorithm="internvideo2",
        embedding_model_version="v1",
        caption_models=["qwen"],
    )
    stage.stage_setup()
    stage.process_data([SplitPipeTask(session_id="test-session", videos=list(videos))])


def _read_mcap(path: Path) -> tuple[list[tuple[Schema | None, Channel, Message]], list[Metadata]]:
    with path.open("rb") as fh:
        reader = make_reader(fh)
        messages = list(reader.iter_messages())
        metadata_records = list(reader.iter_metadata())
    return messages, metadata_records


def _messages_by_topic(
    messages: list[tuple[Schema | None, Channel, Message]],
) -> dict[str, list[tuple[Schema | None, Message]]]:
    by_topic: dict[str, list[tuple[Schema | None, Message]]] = {}
    for schema, channel, message in messages:
        by_topic.setdefault(channel.topic, []).append((schema, message))
    return by_topic


def _fragment_path(tmp_path: Path, video_path: Path, chunk_index: int) -> Path:
    relative = video_path.relative_to(tmp_path / "input")
    return tmp_path / "output" / "mcap_fragments" / relative / f"{chunk_index}.mcap"


def test_fragment_writes_expected_channels(tmp_path: Path) -> None:
    """Chunk-0 fragment carries media, annotations, embedding, and one-shot session records."""
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(video_path, mp4_bytes=_make_mp4())
    video = _make_video(video_path, [clip])

    _process(tmp_path, video)
    assert "McapWriterStage" not in video.errors

    fragment = _fragment_path(tmp_path, video_path, 0)
    assert fragment.is_file()
    messages, metadata_records = _read_mcap(fragment)
    by_topic = _messages_by_topic(messages)

    assert set(by_topic) == {
        mcap_schemas.TOPIC_IMAGE_RAW,
        mcap_schemas.TOPIC_SCENE_ANNOTATION,
        mcap_schemas.TOPIC_CAMERA_INFO,
        mcap_schemas.TOPIC_TF_STATIC,
        mcap_schemas.TOPIC_CLIP_EMBEDDING,
    }

    # Video frames: one CompressedVideo message per frame, Annex-B bitstream, h264.
    frames = by_topic[mcap_schemas.TOPIC_IMAGE_RAW]
    assert len(frames) == NUM_FRAMES
    for schema, _ in frames:
        assert schema is not None
        assert schema.name == mcap_schemas.COMPRESSED_VIDEO_SCHEMA_NAME
        assert schema.encoding == mcap_schemas.JSONSCHEMA_ENCODING
    first_payload = json.loads(frames[0][1].data)
    assert first_payload["format"] == "h264"
    assert first_payload["frame_id"] == "camera"
    bitstream = base64.b64decode(first_payload["data"])
    assert bitstream.startswith((b"\x00\x00\x00\x01", b"\x00\x00\x01"))
    assert min(message.log_time for _, message in frames) == clip.start_ns

    # Annotations: one caption window plus two SAM3 frame payloads, sharing the topic.
    annotations = by_topic[mcap_schemas.TOPIC_SCENE_ANNOTATION]
    assert len(annotations) == 3
    annotation_data = [json.loads(message.data)["data"] for _, message in annotations]
    caption_index = annotation_data.index("a test scene")
    assert annotations[caption_index][1].log_time == clip.start_ns
    detection_payloads = [json.loads(data) for i, data in enumerate(annotation_data) if i != caption_index]
    assert detection_payloads[0]["detections"] == [{"prompt": "car", "object_id": 1}]
    assert detection_payloads[1]["detections"] == []

    # Embedding round-trip.
    embeddings = by_topic[mcap_schemas.TOPIC_CLIP_EMBEDDING]
    assert len(embeddings) == 1
    embedding_payload = json.loads(embeddings[0][1].data)
    assert embedding_payload["model_name"] == "internvideo2"
    assert embedding_payload["model_version"] == "v1"
    decoded = np.frombuffer(base64.b64decode(embedding_payload["data"]), dtype="<f4")
    npt.assert_allclose(decoded, [0.1, 0.2, 0.3])

    # Session metadata record.
    assert len(metadata_records) == 1
    session_metadata = metadata_records[0]
    assert session_metadata.name == mcap_schemas.SESSION_METADATA_RECORD_NAME
    assert session_metadata.metadata["source-video"] == video_path.as_posix()
    assert session_metadata.metadata["num-clip-chunks"] == "1"

    # Payloads dropped after the stage.
    assert clip.encoded_data.resolve() is None
    assert clip.intern_video_2_embedding is None
    assert clip.windows[0].caption == {}


def test_fragment_audio_channel(tmp_path: Path) -> None:
    """A clip with an audio track adds pcm-s16 RawAudio messages; one without does not."""
    video_path = tmp_path / "input" / "video.mp4"
    video = _make_video(video_path, [_make_clip(video_path, mp4_bytes=_make_mp4(with_audio=True))])

    _process(tmp_path, video)
    assert "McapWriterStage" not in video.errors

    messages, _ = _read_mcap(_fragment_path(tmp_path, video_path, 0))
    audio = _messages_by_topic(messages)[mcap_schemas.TOPIC_AUDIO_RAW]
    assert audio
    payload = json.loads(audio[0][1].data)
    assert payload["format"] == "pcm-s16"
    assert payload["sample_rate"] == 48000
    assert payload["number_of_channels"] == 1
    assert base64.b64decode(payload["data"])


def test_chunk1_fragment_omits_session_records(tmp_path: Path) -> None:
    """Only chunk 0 writes session metadata, camera-info, and tf-static."""
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(video_path, mp4_bytes=_make_mp4(), start_s=CLIP_DURATION_S)
    video = _make_video(video_path, [clip], clip_chunk_index=1, num_clip_chunks=2)

    _process(tmp_path, video)

    messages, metadata_records = _read_mcap(_fragment_path(tmp_path, video_path, 1))
    by_topic = _messages_by_topic(messages)
    assert not metadata_records
    assert mcap_schemas.TOPIC_CAMERA_INFO not in by_topic
    assert mcap_schemas.TOPIC_TF_STATIC not in by_topic
    assert len(by_topic[mcap_schemas.TOPIC_IMAGE_RAW]) == NUM_FRAMES
    # Frames are logged at source-timeline offsets: chunk 1 starts one clip in.
    assert min(m.log_time for _, m in by_topic[mcap_schemas.TOPIC_IMAGE_RAW]) == clip.start_ns


def test_zero_clip_chunks(tmp_path: Path) -> None:
    """A zero-clip chunk 0 writes a metadata-only fragment; other empty chunks write nothing."""
    video_path = tmp_path / "input" / "video.mp4"

    empty_chunk1 = _make_video(video_path, [], clip_chunk_index=1, num_clip_chunks=2)
    _process(tmp_path, empty_chunk1)
    assert not _fragment_path(tmp_path, video_path, 1).exists()

    empty_chunk0 = _make_video(video_path, [], clip_chunk_index=0, num_clip_chunks=2)
    _process(tmp_path, empty_chunk0)
    fragment = _fragment_path(tmp_path, video_path, 0)
    assert fragment.is_file()
    messages, metadata_records = _read_mcap(fragment)
    assert len(metadata_records) == 1
    by_topic = _messages_by_topic(messages)
    assert set(by_topic) == {mcap_schemas.TOPIC_CAMERA_INFO, mcap_schemas.TOPIC_TF_STATIC}


def test_span_fallback_when_pts_missing(tmp_path: Path) -> None:
    """Clips without decoded timestamps fall back to span seconds for the base offset."""
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(video_path, mp4_bytes=_make_mp4(), start_s=1.0)
    clip.pts_ns = None
    clip.start_ns = None
    clip.end_ns = None
    video = _make_video(video_path, [clip])

    _process(tmp_path, video)
    assert "McapWriterStage" not in video.errors

    messages, _ = _read_mcap(_fragment_path(tmp_path, video_path, 0))
    frames = _messages_by_topic(messages)[mcap_schemas.TOPIC_IMAGE_RAW]
    assert len(frames) == NUM_FRAMES
    assert min(m.log_time for _, m in frames) == NS_PER_SECOND


def test_secondary_video_keeps_sam3_annotations(tmp_path: Path) -> None:
    """Secondary cameras keep their per-camera SAM3 detections; absent data yields no channel."""
    primary_path = tmp_path / "input" / "cam0.mp4"
    secondary_path = tmp_path / "input" / "cam1.mp4"
    primary = _make_video(primary_path, [_make_clip(primary_path, mp4_bytes=_make_mp4())])
    # Secondary cameras carry SAM3 detections but no captions/embeddings.
    secondary_clip = _make_clip(secondary_path, mp4_bytes=_make_mp4(), with_annotations=False)
    secondary_clip.sam3_frames = [
        {"frame_idx": 2, "timestamp_s": 2 / FPS, "detections": [{"prompt": "truck", "object_id": 7}]},
    ]
    secondary = _make_video(secondary_path, [secondary_clip])

    _process(tmp_path, primary, secondary)

    messages, _ = _read_mcap(_fragment_path(tmp_path, secondary_path, 0))
    by_topic = _messages_by_topic(messages)
    annotations = by_topic[mcap_schemas.TOPIC_SCENE_ANNOTATION]
    assert len(annotations) == 1
    detections = json.loads(json.loads(annotations[0][1].data)["data"])["detections"]
    assert detections == [{"prompt": "truck", "object_id": 7}]
    # No embeddings/captions exist on the secondary, so that channel never appears.
    assert mcap_schemas.TOPIC_CLIP_EMBEDDING not in by_topic
    assert len(by_topic[mcap_schemas.TOPIC_IMAGE_RAW]) == NUM_FRAMES


def test_consolidate_merges_fragments(tmp_path: Path) -> None:
    """Two chunk fragments merge into one MCAP named after the input video's relative path."""
    video_path = tmp_path / "input" / "video.mp4"

    chunk0_clip = _make_clip(video_path, mp4_bytes=_make_mp4())
    chunk0 = _make_video(video_path, [chunk0_clip], clip_chunk_index=0, num_clip_chunks=2)
    _process(tmp_path, chunk0)

    chunk1_clip = _make_clip(video_path, mp4_bytes=_make_mp4(), start_s=CLIP_DURATION_S)
    chunk1 = _make_video(video_path, [chunk1_clip], clip_chunk_index=1, num_clip_chunks=2)
    _process(tmp_path, chunk1)

    fragment0 = _fragment_path(tmp_path, video_path, 0)
    fragment1 = _fragment_path(tmp_path, video_path, 1)
    fragment_messages = len(_read_mcap(fragment0)[0]) + len(_read_mcap(fragment1)[0])

    consolidate_mcap_fragments(str(tmp_path / "output"), "default")

    final_path = tmp_path / "output" / "mcap" / "video.mp4.mcap"
    assert final_path.is_file()
    messages, metadata_records = _read_mcap(final_path)
    assert len(messages) == fragment_messages
    assert len(metadata_records) == 1

    # Exactly one channel per topic and one schema per name after id remapping.
    with final_path.open("rb") as fh:
        summary = make_reader(fh).get_summary()
    assert summary is not None
    assert len({c.topic for c in summary.channels.values()}) == len(summary.channels)
    assert len({s.name for s in summary.schemas.values()}) == len(summary.schemas)
    assert mcap_schemas.TOPIC_IMAGE_RAW in {c.topic for c in summary.channels.values()}

    # Merged timeline covers both chunks and fragments are removed.
    frames = [m for _, c, m in messages if c.topic == mcap_schemas.TOPIC_IMAGE_RAW]
    frame_times = [m.log_time for m in frames]
    assert min(frame_times) == chunk0_clip.start_ns
    assert max(frame_times) >= chunk1_clip.start_ns
    # Sequences are renumbered across the merge instead of restarting per chunk.
    assert sorted(m.sequence for m in frames) == list(range(len(frames)))
    assert not fragment0.exists()
    assert not fragment1.exists()


def test_consolidate_single_fragment(tmp_path: Path) -> None:
    """A single-fragment video is validated and rewritten to the final MCAP, fragment removed."""
    video_path = tmp_path / "input" / "video.mp4"
    _process(tmp_path, _make_video(video_path, [_make_clip(video_path, mp4_bytes=_make_mp4())]))

    fragment = _fragment_path(tmp_path, video_path, 0)
    fragment_messages, fragment_metadata = _read_mcap(fragment)

    consolidate_mcap_fragments(str(tmp_path / "output"), "default")

    final_path = tmp_path / "output" / "mcap" / "video.mp4.mcap"
    messages, metadata_records = _read_mcap(final_path)
    assert len(messages) == len(fragment_messages)
    assert len(metadata_records) == len(fragment_metadata)
    assert not fragment.exists()


def test_consolidate_skips_incomplete_fragment_sets(tmp_path: Path) -> None:
    """Fragment sets missing chunks are preserved untouched instead of merged and deleted."""
    # Chunk 0 announces two chunks in its session metadata, but chunk 1 is missing.
    tail_missing_path = tmp_path / "input" / "tail_missing.mp4"
    chunk0 = _make_video(
        tail_missing_path,
        [_make_clip(tail_missing_path, mp4_bytes=_make_mp4())],
        clip_chunk_index=0,
        num_clip_chunks=2,
    )
    _process(tmp_path, chunk0)

    # A set without chunk 0 (leftover of an interrupted run).
    head_missing_path = tmp_path / "input" / "head_missing.mp4"
    chunk1 = _make_video(
        head_missing_path,
        [_make_clip(head_missing_path, mp4_bytes=_make_mp4(), start_s=CLIP_DURATION_S)],
        clip_chunk_index=1,
        num_clip_chunks=2,
    )
    _process(tmp_path, chunk1)

    consolidate_mcap_fragments(str(tmp_path / "output"), "default")

    assert not (tmp_path / "output" / "mcap" / "tail_missing.mp4.mcap").exists()
    assert not (tmp_path / "output" / "mcap" / "head_missing.mp4.mcap").exists()
    assert _fragment_path(tmp_path, tail_missing_path, 0).is_file()
    assert _fragment_path(tmp_path, head_missing_path, 1).is_file()


def test_consolidate_raises_on_corrupt_fragment(tmp_path: Path) -> None:
    """A truncated/corrupt fragment fails consolidation loudly and is preserved for inspection."""
    corrupt = tmp_path / "output" / "mcap_fragments" / "video.mp4" / "0.mcap"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not a valid mcap file")

    with pytest.raises(RuntimeError, match="MCAP consolidation failed"):
        consolidate_mcap_fragments(str(tmp_path / "output"), "default")

    assert not (tmp_path / "output" / "mcap" / "video.mp4.mcap").exists()
    assert corrupt.is_file()


def test_write_failure_propagates_with_payloads_intact(tmp_path: Path) -> None:
    """A fragment-write failure raises (enabling Xenna run attempts) and keeps clip payloads."""
    video_path = tmp_path / "input" / "video.mp4"
    clip = _make_clip(video_path, mp4_bytes=b"not-an-mp4")
    video = _make_video(video_path, [clip])

    with pytest.raises(av.FFmpegError):
        _process(tmp_path, video)

    # Payloads survive the failure so a retry can re-produce the fragment.
    assert clip.encoded_data.resolve() is not None
    assert clip.windows[0].caption == {"qwen": "a test scene"}
    assert clip.intern_video_2_embedding is not None


def test_consolidate_no_fragments_is_noop(tmp_path: Path) -> None:
    """Consolidation returns quietly when there is nothing to merge."""
    consolidate_mcap_fragments(str(tmp_path / "missing-output"), "default")
    assert not (tmp_path / "missing-output").exists()
