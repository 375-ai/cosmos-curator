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

"""Shared storage utilities for Curator Next recipes."""

import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix, is_s3path
from cosmos_curator.core.utils.storage.storage_utils import get_storage_client

_PRECONDITION_FAILED = 412


def artifact_uri(location: str) -> str:
    """Return the normalized durable URI recorded in manifests, rows, and receipts."""
    if is_s3path(location):
        return S3Prefix(location).path
    if location.startswith("file://"):
        return _file_uri_to_path(location).resolve().as_uri()
    return Path(location).expanduser().resolve().as_uri()


def write_media(location: str, data: bytes, *, storage_profile: str = "default") -> None:
    """Atomically replace a deterministic local/S3 media object with complete bytes."""
    if is_s3path(location):
        client = get_storage_client(location, profile_name=storage_profile, can_overwrite=True)
        if client is None:
            msg = f"Could not create an S3 client for {location}"
            raise RuntimeError(msg)
        client.upload_bytes(S3Prefix(location), data)
        return

    destination = Path(location)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(destination)
        _fsync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_local_if_absent(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    while True:
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        break

    try:
        with os.fdopen(descriptor, "wb") as artifact:
            artifact.write(data)
            artifact.flush()
            os.fsync(artifact.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_s3_if_absent(location: str, data: bytes, *, storage_profile: str) -> None:
    prefix = S3Prefix(location)
    client = get_storage_client(location, profile_name=storage_profile)
    if not isinstance(client, S3Client):
        msg = f"Could not create an S3 client for {location}"
        raise TypeError(msg)
    try:
        client.s3.put_object(Bucket=prefix.bucket, Key=prefix.prefix, Body=data, IfNoneMatch="*")
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = exc.response.get("Error", {}).get("Code")
        if status == _PRECONDITION_FAILED or code in {"PreconditionFailed", "ConditionalRequestConflict"}:
            return
        raise


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_uri_to_path(uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"} or not parsed.path:
        msg = f"Unsupported local file URI: {uri}"
        raise ValueError(msg)
    return Path(unquote(parsed.path))
