#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the redistributable media stack inside a container."""
# ruff: noqa: T201

import argparse
import subprocess
import tempfile
from pathlib import Path

FFMPEG_PREFIX = Path("/opt/ffmpeg")
FFMPEG = FFMPEG_PREFIX / "bin" / "ffmpeg"
FFPROBE = FFMPEG_PREFIX / "bin" / "ffprobe"
LIBOPENH264_SEARCH_ROOTS = (
    FFMPEG_PREFIX,
    Path("/usr/lib"),
    Path("/usr/local/lib"),
    Path("/opt/cosmos-curator/.pixi"),
)


def _run(command: list[str]) -> str:
    print(f"+ {' '.join(command)}", flush=True)
    result = subprocess.run(  # noqa: S603
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, flush=True)
    return result.stdout


def _require_path(path: Path) -> None:
    if not path.exists():
        msg = f"missing required path: {path}"
        raise AssertionError(msg)


def _find_libopenh264() -> list[str]:
    roots = [str(path) for path in LIBOPENH264_SEARCH_ROOTS if path.exists()]
    if not roots:
        return []
    output = _run(["find", *roots, "-name", "libopenh264*"])
    return [line for line in output.splitlines() if line.strip()]


def _assert_cv2_has_ffmpeg() -> None:
    import cv2  # noqa: PLC0415

    info = cv2.getBuildInformation()
    ffmpeg_line = next((line for line in info.splitlines() if "FFMPEG:" in line), "FFMPEG: <missing>")
    print(f"cv2 version: {cv2.__version__}", flush=True)
    print(ffmpeg_line, flush=True)
    if "FFMPEG:                      YES" not in info:
        msg = "OpenCV was not built with FFmpeg support"
        raise AssertionError(msg)


def _smoke_imports() -> None:
    import av  # noqa: PLC0415

    print(f"PyAV version: {av.__version__}", flush=True)
    print(f"PyAV libraries: {av.library_versions}", flush=True)
    _assert_cv2_has_ffmpeg()


def _assert_no_libopenh264() -> None:
    found = _find_libopenh264()
    if found:
        msg = "redistributable image contains libopenh264:\n" + "\n".join(found)
        raise AssertionError(msg)


def _assert_mounted_ffmpeg_supports_h264() -> None:
    version = _run([str(FFMPEG), "-version"])
    if "--enable-libopenh264" not in version:
        msg = "mounted FFmpeg does not report --enable-libopenh264"
        raise AssertionError(msg)

    encoders = _run([str(FFMPEG), "-hide_banner", "-encoders"])
    if "libopenh264" not in encoders:
        msg = "mounted FFmpeg does not expose the libopenh264 encoder"
        raise AssertionError(msg)

    with tempfile.TemporaryDirectory(prefix="redistributable-media-smoke-") as tmpdir:
        output_path = Path(tmpdir) / "h264.mp4"
        _run(
            [
                str(FFMPEG),
                "-hide_banner",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x64:rate=1:duration=2",
                "-frames:v",
                "2",
                "-c:v",
                "libopenh264",
                str(output_path),
            ]
        )

        codec_name = _run(
            [
                str(FFPROBE),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                str(output_path),
            ]
        ).strip()
        if codec_name != "h264":
            msg = f"expected generated smoke video codec h264, got {codec_name!r}"
            raise AssertionError(msg)

        _run([str(FFMPEG), "-v", "error", "-i", str(output_path), "-f", "null", "-"])
        _assert_pyav_decodes(output_path)
        _assert_cv2_decodes(output_path)


def _assert_pyav_decodes(path: Path) -> None:
    import av  # noqa: PLC0415

    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    print(f"PyAV decoded {len(frames)} H.264 frames", flush=True)
    if not frames:
        msg = "PyAV did not decode any H.264 frames"
        raise AssertionError(msg)


def _assert_cv2_decodes(path: Path) -> None:
    import cv2  # noqa: PLC0415

    capture = cv2.VideoCapture(str(path))
    try:
        ok, _frame = capture.read()
    finally:
        capture.release()
    if not ok:
        msg = "OpenCV did not decode the generated H.264 smoke video"
        raise AssertionError(msg)
    print("OpenCV decoded the generated H.264 smoke video", flush=True)


def run_smoke(mode: str) -> None:
    """Run the requested media-stack smoke test mode."""
    _require_path(FFMPEG)
    _require_path(FFPROBE)
    _run([str(FFMPEG), "-version"])
    _run([str(FFPROBE), "-version"])
    _smoke_imports()

    if mode == "bare":
        _assert_no_libopenh264()
    elif mode == "mounted-ffmpeg":
        _assert_mounted_ffmpeg_supports_h264()
    else:
        msg = f"unsupported smoke mode: {mode}"
        raise ValueError(msg)


def main() -> int:
    """Run the smoke-test CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bare", "mounted-ffmpeg"), required=True)
    args = parser.parse_args()

    run_smoke(args.mode)
    print(f"redistributable media smoke passed: {args.mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
