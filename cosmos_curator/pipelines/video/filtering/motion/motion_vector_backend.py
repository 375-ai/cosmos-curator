# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Backend using h264 motion vectors for optical flow filtering."""

import attrs
import cv2
import numpy as np
import numpy.typing as npt
import torch
from nvtx import nvtx  # type: ignore[import-untyped]
from torch import Tensor

from cosmos_curator.core.sensors.data.camera_data import MotionVectorData

# We error on any video with a width or height less than this.
# The motion detection algorithm can't handle any resolutions less than this.
_MIN_SIDE_RESOLUTION = 256
_LEGACY_MOTION_VECTOR_COLUMNS = 10


class VideoResolutionTooSmallError(Exception):
    """Exception raised when video resolution is below the minimum required size.

    This error occurs when either the width or height of the video is less than
    the minimum resolution threshold required for motion detection.
    """


@attrs.define
class MotionInfo:
    """Container for motion detection results.

    This class stores the results of motion detection analysis, including:
    - Whether the video has small motion
    - The minimum motion value in a 256x256 patch
    - The global average motion value across the entire videoq
    """

    is_small_motion: bool
    per_patch_min_256: float
    global_mean: float


@attrs.define
class DecodedData:
    """Container for decoded video frames containing motion vector data.

    This class stores a list of decoded frames, each containing motion vector data,
    and the dimensions of the RGB decoded frame used to construct the flow vector.
    """

    # List of decoded frames containing motion vector data, not RGB
    frames: list[npt.NDArray]  # type: ignore[type-arg]
    # pass the dimensions of the RGB decoded frame to construct flow vector
    frame_size: torch.Size

    def get_major_size(self) -> int:
        """Calculate total size in bytes of all frames in the decoded data.

        Returns:
            Total size in bytes.

        """
        total_size = 0
        for frame in self.frames:
            total_size += frame.nbytes
        return total_size


def sensor_motion_vector_data_to_legacy_matrices(
    motion_vectors: MotionVectorData,
) -> list[npt.NDArray[np.float64]]:
    """Project sensor motion vectors into the legacy PyAV matrix shape.

    This is the only adapter between ``CameraData.motion_vectors`` and the motion filter
    path. The first column is ``source`` so the output matches the per-frame matrix layout
    consumed by ``check_if_small_motion``.
    """
    matrices: list[npt.NDArray[np.float64]] = []
    for frame in motion_vectors.frames:
        if len(frame.source) == 0:
            matrices.append(np.empty((0, _LEGACY_MOTION_VECTOR_COLUMNS + 1), dtype=np.float64))
            continue
        matrices.append(
            np.asarray(
                np.column_stack(
                    (
                        frame.source,
                        frame.w,
                        frame.h,
                        frame.src_x,
                        frame.src_y,
                        frame.dst_x,
                        frame.dst_y,
                        frame.flags,
                        frame.motion_x,
                        frame.motion_y,
                        frame.motion_scale,
                    )
                ),
                dtype=np.float64,
            )
        )
    return matrices


@nvtx.annotate()  # type: ignore[untyped-decorator]
def motion_vectors_to_flowfield(mvs: Tensor, size: list[int], flow: Tensor | None = None) -> Tensor:
    """Compute a canonical flow from motion vectors."""
    # get relevant info for later
    batch_size = mvs.shape[0]
    n_vectors = mvs.shape[1]
    block_sizes = mvs[..., 0:2]
    dst = mvs[..., 4:6]
    motion = mvs[..., 7:9]
    motion_scale = mvs[..., 9].unsqueeze(-1)
    device = mvs.device

    # Make indices for the batch number, this will be used as part of an index later
    batch_indices = torch.arange(batch_size, device=device)
    batch_pos = batch_indices.view(-1, 1).expand(-1, n_vectors).view(batch_size, n_vectors, 1)

    # compute sub-pixel src locations
    # `src = dst + motion / motion_scale`
    delta = -motion / motion_scale

    # add batch position to the source x, y positions
    dst_complete = torch.cat([batch_pos, dst], dim=-1)

    # create a "flow" where each source pixel is annotated with its destination pixel
    if flow is None or flow.shape != (batch_size, *size, 2):
        flow = torch.zeros(batch_size, *size, 2, device=device)
    else:
        flow.zero_()

    # choices for block size
    block_options = [
        torch.as_tensor([8, 8], device=device),
        torch.as_tensor([16, 16], device=device),
        torch.as_tensor([16, 8], device=device),
        torch.as_tensor([8, 16], device=device),
    ]

    # separate motion vectors by block size
    selected_blocks = [(block_sizes == b).all(-1).unsqueeze(-1) for b in block_options]

    # make offsets from center point for each of the block sizes.
    # this is ugly because we need to prepend 0 as a "batch offset" since the points are (batch, x, y)
    offsets = [
        torch.stack(torch.meshgrid(*[torch.arange(-b // 2, b // 2, device=device) for b in bs])) for bs in block_options
    ]

    # tile each of the mvs so that they contain the center point repeated for the size of the entire block
    dst_tiles = [
        dst_complete.masked_select(blocks).view(-1, 1, 1, 1).tile(1, 1, *block_size).view(-1, 3, *block_size)
        for blocks, block_size in zip(selected_blocks, block_options, strict=False)
    ]

    # # split each tensor into three indices, batch, h, w
    dst_b, dst_x, dst_y = zip(*[d.split(1, dim=1) for d in dst_tiles], strict=False)  # type: ignore[no-untyped-call]

    # Add in the offsets
    dst_x1 = [(mv_b + offset_b[0]) for mv_b, offset_b in zip(dst_x, offsets, strict=False)]
    dst_y1 = [(mv_b + offset_b[1]) for mv_b, offset_b in zip(dst_y, offsets, strict=False)]

    # check bounds (needed for indexing)
    dst_x2 = [mv_b.where(mv_b > 0, torch.as_tensor(0, device=device)) for mv_b in dst_x1]
    dst_y2 = [mv_b.where(mv_b > 0, torch.as_tensor(0, device=device)) for mv_b in dst_y1]

    dst_x3 = [mv_b.where(mv_b < size[-1], size[-1] - 1) for mv_b in dst_x2]
    dst_y3 = [mv_b.where(mv_b < size[-2], size[-2] - 1) for mv_b in dst_y2]

    # flatten the indices and concat them
    dst_x4 = torch.cat([mv.flatten() for mv in dst_x3], dim=0)
    dst_y4 = torch.cat([mv.flatten() for mv in dst_y3], dim=0)
    dst_b4 = torch.cat([mv.flatten() for mv in dst_b], dim=0)

    # tile and offset the source positions
    delta_tiles = [
        delta.masked_select(blocks).view(-1, 1, 1).tile(1, *block_size).view(-1, 2, *block_size)
        for blocks, block_size in zip(selected_blocks, block_options, strict=False)
    ]
    delta_flat = torch.cat([mv.movedim(1, -1).flatten(0, 2) for mv in delta_tiles], dim=0)

    # index the flow image and set it to the destination
    flow.index_put_((dst_b4.long(), dst_y4.long(), dst_x4.long()), delta_flat, accumulate=False)

    return flow


@nvtx.annotate()  # type: ignore[untyped-decorator]
def check_if_small_motion(  # noqa: PLR0913
    mv_list: list[npt.NDArray],  # type: ignore[type-arg]
    frame_shape: torch.Size,
    global_mean_threshold: float = 0.00098,
    per_patch_min_256_threshold: float = 0.000001,
    *,
    use_gpu: bool = False,
    batch_size: int = 256,
) -> MotionInfo:
    """Check if a video has small motion.

    This function checks if a video has small motion by calculating the global mean
    and per-pixel average motion values.

    Args:
        mv_list: List of motion vectors.
        frame_shape: Shape of the frame.
        global_mean_threshold: Threshold for global mean motion.
        per_patch_min_256_threshold: Threshold for per-patch minimum motion.
        use_gpu: Whether to use GPU for computation.
        batch_size: Size of the batch for processing.

    Returns:
        MotionInfo object containing the results of the motion detection.

    """
    device = torch.device("cuda" if use_gpu else "cpu")

    global_sum_tensor = torch.tensor(0.0, device=device)
    per_pixel_sum_tensor = torch.zeros((frame_shape[0], frame_shape[1]), device=device)
    num_frames = 0
    preallocated_flow = torch.zeros(batch_size, frame_shape[0], frame_shape[1], 2, device=device)

    for batch_offset in range(0, len(mv_list), batch_size):
        current_batch_size = min(batch_size, len(mv_list) - batch_offset)

        # Pad all tensors to the same dimensions (the number of blocks varies per frame).
        # Padding with zeros is valid because the blocks of size (0, 0) will not be selected.
        max_n_vectors = max(mv_item.shape[0] for mv_item in mv_list[batch_offset : batch_offset + current_batch_size])
        mv_data_padded = torch.zeros(current_batch_size, max_n_vectors, 10, dtype=torch.float32, device=device)

        for batch_id in range(batch_offset, batch_offset + current_batch_size):
            data = torch.tensor(mv_list[batch_id][:, 1:], dtype=torch.float32, device=device)
            mv_data_padded[batch_id - batch_offset, : data.shape[0], :] = data

        flow_field = motion_vectors_to_flowfield(mv_data_padded, [frame_shape[0], frame_shape[1]], preallocated_flow)
        magnitudes = torch.linalg.vector_norm(flow_field, dim=3) / (frame_shape[1] + frame_shape[0])

        global_sum_tensor += magnitudes.sum()
        per_pixel_sum_tensor += magnitudes.sum(dim=0)
        num_frames += current_batch_size

    total_elements = num_frames * frame_shape[0] * frame_shape[1]
    global_mean = (global_sum_tensor / total_elements).item()
    per_pixel_avg = (per_pixel_sum_tensor / num_frames).cpu().numpy()
    per_patch_min_256 = float(cv2.resize(per_pixel_avg, None, fx=1 / 256, fy=1 / 256).min())

    is_small_motion = global_mean < global_mean_threshold or per_patch_min_256 < per_patch_min_256_threshold

    return MotionInfo(is_small_motion, per_patch_min_256, global_mean)
