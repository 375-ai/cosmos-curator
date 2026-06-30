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
"""Tests proving the reusable OpenAI / Gemini provider stub fixtures work.

These exercise the shared stub fixtures defined in ``conftest.py`` /
``provider_stub_utils.py`` against the real provider call sites:

* OpenAI chat completions + ``model='auto'`` resolution, driven end-to-end
  through the real ``openai`` SDK to the localhost HTTP stub.
* OpenAI embeddings via the vLLM chat-embedding extension.
* OpenAI Responses API (``output_text``).
* Gemini chat via the ``AsyncMock``-based fake client.

The ``openai`` SDK only lives in the pixi ``default`` env, so the SDK-dependent
tests ``importorskip('openai')`` (they run in CI's unit_tests/default env and
skip elsewhere instead of breaking collection).
"""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from cosmos_curator.core.utils.config.config import ConfigFileData, Gemini, OpenAIConfig, OpenAIEndpointConfig
from cosmos_curator.pipelines.common import openai_embedding_utils
from cosmos_curator.pipelines.common.openai_embedding_utils import call_openai_embedding_api
from cosmos_curator.pipelines.video.captioning import gemini_caption_stage, openai_caption_stage
from cosmos_curator.pipelines.video.captioning.gemini_caption_stage import GeminiCaptionStage
from cosmos_curator.pipelines.video.captioning.openai_caption_stage import OpenAICaptionStage
from cosmos_curator.pipelines.video.utils.data_model import Clip, SplitPipeTask, Video, Window
from tests.cosmos_curator.pipelines.provider_stub_utils import GeminiStub, OpenAIStubServer


def _make_video_task(mp4_bytes: bytes) -> SplitPipeTask:
    """Build a minimal one-clip / one-window SplitPipeTask for captioning."""
    clip = Clip(uuid=uuid4(), source_video="source.mp4", span=(0.0, 1.0))
    clip.windows.append(Window(start_frame=0, end_frame=1, mp4_bytes=mp4_bytes))
    video = Video(input_video=Path("source.mp4"))
    video.clips.append(clip)
    return SplitPipeTask(session_id="test-session", video=video)


def _caption_config(base_url: str) -> ConfigFileData:
    return ConfigFileData(openai=OpenAIConfig(caption=OpenAIEndpointConfig(api_key="test-key", base_url=base_url)))


# ---------------------------------------------------------------------------
# OpenAI HTTP stub
# ---------------------------------------------------------------------------


def test_openai_caption_stage_against_stub(
    openai_stub_server: OpenAIStubServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenAICaptionStage should drive the real openai SDK to the local stub.

    Uses ``model_name='auto'`` so the stage first hits ``GET /v1/models`` to
    resolve the model, then ``POST /v1/chat/completions`` for the caption.
    """
    openai = pytest.importorskip("openai")
    monkeypatch.setattr(openai_caption_stage, "openai", openai, raising=False)
    monkeypatch.setattr(openai_caption_stage, "maybe_load_config", lambda: _caption_config(openai_stub_server.base_url))

    openai_stub_server.set_models(["stub-vlm"])
    openai_stub_server.set_chat_content("a stubbed caption", model="stub-vlm")

    stage = OpenAICaptionStage(model_name="auto", max_caption_retries=1, retry_delay_seconds=0)
    stage.stage_setup()
    task = _make_video_task(b"\x00\x01\x02")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    window = task.video.clips[0].windows[0]
    assert window.caption["openai"] == "a stubbed caption"
    assert window.caption_status == "success"

    assert openai_stub_server.requests_for("/models"), "model='auto' should query GET /v1/models"
    chat_requests = openai_stub_server.requests_for("/chat/completions")
    assert len(chat_requests) == 1
    body = chat_requests[0].body
    assert body is not None
    assert body["model"] == "stub-vlm"
    assert isinstance(body["messages"], list)
    content_parts = body["messages"][0]["content"]
    assert any(part["type"] == "video_url" for part in content_parts)


def test_openai_embedding_against_stub(openai_stub_server: OpenAIStubServer, monkeypatch: pytest.MonkeyPatch) -> None:
    """call_openai_embedding_api should round-trip a vector through the stub."""
    openai = pytest.importorskip("openai")
    from openai.types.create_embedding_response import CreateEmbeddingResponse  # noqa: PLC0415

    # The provider modules only ``import openai`` under the pixi "default" env;
    # inject the real symbols so the retry decorator / response cast resolve.
    monkeypatch.setattr(openai_embedding_utils, "openai", openai, raising=False)
    monkeypatch.setattr(openai_embedding_utils, "CreateEmbeddingResponse", CreateEmbeddingResponse, raising=False)

    openai_stub_server.set_embedding([1.0, 2.0, 3.0, 4.0], model="embed-model")
    client = openai.OpenAI(api_key="test-key", base_url=openai_stub_server.base_url)
    content_parts = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]

    embedding = call_openai_embedding_api(client, "embed-model", content_parts, max_retries=1, retry_delay_seconds=0)

    np.testing.assert_array_equal(embedding, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert embedding.dtype == np.float32

    requests = openai_stub_server.requests_for("/embeddings")
    assert len(requests) == 1
    body = requests[0].body
    assert body is not None
    assert body["model"] == "embed-model"
    assert body["encoding_format"] == "float"
    assert body["messages"][0]["content"] == content_parts


def test_openai_responses_against_stub(openai_stub_server: OpenAIStubServer) -> None:
    """The stub's Responses route should return a parseable ``output_text``."""
    openai = pytest.importorskip("openai")

    openai_stub_server.set_responses_text("an enhanced caption")
    client = openai.OpenAI(api_key="test-key", base_url=openai_stub_server.base_url)

    response = client.responses.create(model="enhance-model", input="hello")

    assert response.output_text == "an enhanced caption"
    requests = openai_stub_server.requests_for("/responses")
    assert len(requests) == 1
    assert requests[0].body is not None
    assert requests[0].body["model"] == "enhance-model"


def test_openai_stub_records_each_request(openai_stub_server: OpenAIStubServer) -> None:
    """The stub should expose every received request for assertions."""
    openai = pytest.importorskip("openai")

    client = openai.OpenAI(api_key="test-key", base_url=openai_stub_server.base_url)
    client.models.list()

    assert [req.method for req in openai_stub_server.requests_for("/models")] == ["GET"]


# ---------------------------------------------------------------------------
# Gemini AsyncMock stub
# ---------------------------------------------------------------------------


def test_gemini_caption_stage_against_stub(
    gemini_stub: Callable[..., GeminiStub], monkeypatch: pytest.MonkeyPatch
) -> None:
    """GeminiCaptionStage should caption via the fake genai client and capture kwargs."""
    monkeypatch.setattr(
        gemini_caption_stage,
        "load_config",
        lambda: ConfigFileData(gemini=Gemini(api_key="test-key")),
    )
    stub = gemini_stub(text="a gemini caption")
    monkeypatch.setattr(gemini_caption_stage, "genai", SimpleNamespace(Client=lambda **_kwargs: stub.client))

    stage = GeminiCaptionStage(model_name="models/test", max_caption_retries=1, retry_delay_seconds=0)
    stage.stage_setup()
    task = _make_video_task(b"\x00\x01\x02")
    try:
        stage.process_data([task])
    finally:
        stage.destroy()

    window = task.video.clips[0].windows[0]
    assert window.caption["gemini"] == "a gemini caption"
    assert window.caption_status == "success"

    assert len(stub.calls) == 1
    assert stub.calls[0]["model"] == "models/test"
    assert "contents" in stub.calls[0]
