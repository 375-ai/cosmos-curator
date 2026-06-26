# Sensor Library LiDAR Data Design

This note documents the proposed design for the first generic LiDAR data
structure in the Cosmos Curator Sensor Library. `LidarData` would be
implemented at `cosmos_curator/core/sensors/data/lidar_data.py`. This note
does not define `LidarSensor`, packet parsing, calibration, motion-compensation
algorithms, or downstream perception models.

The note is structured as a decision document: each open question is laid out
with options, trade-offs, and a tentative recommendation that reviewers can
ratify, alter, or defer.

## Proposed Data Model

`LidarData` is a structure-of-arrays batch of LiDAR points. The batch's row
dimension `N` is the number of alignment rows (one per `align_timestamps_ns`
entry); points are stored flat across all rows in a single contiguous block,
with a CSR-style `align_offsets: (N+1,) int64` array partitioning the flat
buffer into per-row windows. `LidarData` satisfies the existing `SensorData`
protocol ([SensorData](#ref-sensor-data)):

- `align_timestamps_ns` is the reference timeline requested from the
  `SamplingGrid`.
- `sensor_timestamps_ns` is the source measurement time selected for each
  alignment row.
- Flat per-point arrays — `points_xyz`, `points_intensity`,
  `points_timestamps_ns`, plus optional `points_ring`, `points_return_index`,
  `points_reflectivity`, `points_ambient`, `points_validity`,
  `points_radial_velocity` — share the leading dimension `P_total`.
- `align_offsets: (N+1,) int64` carries the CSR-style window boundaries:
  the points associated with alignment row `i` are
  `points_xyz[align_offsets[i]:align_offsets[i+1]]`. When a producer's
  natural unit is a spinning-LiDAR sweep, the windows are sweep boundaries.
  When the producer instead queries "points near alignment instant `t_i`,"
  the windows are time-of-interest neighborhoods. Pipelines that ignore the
  per-row partitioning can read the flat arrays directly.
- Optional `ego_poses` carries one rig-to-global-frame transform per
  alignment row aligned to `align_timestamps_ns`, used by consumers that
  need motion compensation or multi-row aggregation. The target global
  frame is named by `LidarMetadata.ego_poses_frame` (e.g., `"world"`,
  `"map"`, `"odom"`) and is independent of `LidarMetadata.reference_frame`,
  which describes only the coordinate frame of `points_xyz`.
- Optional `extrinsics` carries the static sensor-to-rig calibration.
- `metadata` (`LidarMetadata`, required) carries motion-compensation status,
  the reference frame, the motion-compensation reference instant (required
  when `motion_compensated=True`), and sensor model.

This lets `LidarData` represent decoded source sweeps when the reference grid
is the LiDAR's own sweep timestamps (`align_timestamps_ns ==
sensor_timestamps_ns`) and aligned point batches when the LiDAR is resampled
onto a camera, IMU, or arbitrary reference grid via `AlignedFrame`
([AlignedFrame](#ref-aligned-frame)).

"Decoded source points" here means parsed LiDAR packets converted to Sensor
Library field names, NumPy arrays, and SI units. It does not mean raw vendor
UDP packets, MCAP message payloads, or pre-motion-compensated point streams.

## Existing Sensor Library Conventions

The current Sensor Library establishes these conventions, and `LidarData`
should follow them rather than introduce a sensor-specific batch model:

- `SensorData` requires `align_timestamps_ns` and `sensor_timestamps_ns`
  ([SensorData](#ref-sensor-data)).
- `AlignedFrame` requires every payload's `align_timestamps_ns` to exactly
  match the frame's reference timeline ([AlignedFrame](#ref-aligned-frame)).
- `CameraData` and `VideoIndex` use attrs classes, structure-of-arrays layout,
  read-only NumPy views via `as_readonly_view`, and explicit shape/dtype
  validators ([CameraData](#ref-camera-data),
  [validation helpers](#ref-validation-helpers)).
- Public timestamp arrays use `np.int64` nanoseconds.
- Boolean masks use `np.bool_`.
- Static sensor-to-rig calibration is represented by a single `SensorExtrinsics`
  4x4 row-major `float64` matrix ([SensorExtrinsics](#ref-sensor-extrinsics)).
- Ragged side data aligned 1:1 with the primary payload has two existing
  precedents: tuple-of-frozen-containers ([MotionVectorData](#ref-motion-vector-data)
  for video frames) and flat-with-offsets (no existing in-package use, but
  the standard CSR / Apache Arrow pattern).

LiDAR adopts the **flat-with-offsets** form rather than the MotionVectorData
tuple form. The original recommendation was the tuple form for consistency
with the camera ragged-side-data precedent, but review feedback (see
[D1](#d1-payload-shape)) noted that not every consumer pipeline organizes
points by sweep — many pipelines just want "points near an alignment
timestamp," with sweep boundaries either irrelevant or absent (e.g.,
solid-state LiDARs). Flat-with-offsets keeps the per-row partitioning
expressible without forcing it as required structure.

## Reference Models Reviewed

LiDAR vocabulary and schema choices are well-established across open AV
datasets, vendor SDKs, and robotics middleware. The following sources are
reviewed for this design, prioritized by their direct relevance to the data
model.

### ROS `sensor_msgs/PointCloud2`

`sensor_msgs/PointCloud2` is the dominant generic point-cloud message in
robotics ([ROS PointCloud2](#ref-ros-pointcloud2)). Its `PointField` schema is
self-describing: each field has a name, byte offset, datatype enum (`FLOAT32`,
`UINT16`, etc.), and a count, allowing arbitrary per-point attributes. A
`PointCloud2` has one `header.stamp` per cloud and no per-point timestamps;
per-point timing, when present, is conventionally encoded as an additional
field (e.g., `t` or `time` in nanoseconds relative to the cloud's reference
time).

This supports providing a fixed set of named per-point arrays on
`LidarData` rather than a self-describing offset/datatype table: the
sensor library is consumed by Python ML pipelines that benefit more from
predictable typed access than from arbitrary-schema flexibility. It also
supports keeping a separate explicit `points_timestamps_ns` field rather
than relying on a single row-level timestamp.

### nuScenes

nuScenes serializes each sweep as a flat `.pcd.bin` buffer with 5 `float32`
channels per point: `x`, `y`, `z`, `intensity`, `ring`
([nuScenes data format](#ref-nuscenes)). Each sweep has a single reference
timestamp at the file level, not per-point timestamps. Multi-sweep aggregation
(`LidarPointCloud.from_file_multisweep` in the devkit) composes
`T_global_egopose * T_egopose_lidar` per sweep to bring all points into a
common reference frame.

This supports the choice of `(P_i, 3) float32` for `xyz` and a separate
`(P_i,) intensity` channel. It also supports keeping ego pose external to the
per-sweep payload and bringing it together at the batch level rather than
duplicating per-point.

### Waymo Open Dataset

Waymo preserves the LiDAR's range-image topology (a `(rings, azimuth_bins, C)`
dense tensor with `range`, `intensity`, `elongation`, `is_in_no_label_zone`)
alongside a derived point-cloud projection ([Waymo Open Dataset](#ref-waymo)).
Each range-image pixel has a per-pixel `range_image_pose` providing
motion-compensation transforms.

This is the strongest existing reference for explicit per-point timing and
motion compensation. It supports including `points_timestamps_ns` and
`ego_poses` in the first `LidarData`. It also supports treating range-image
representation as an optional later extension rather than a required first
form: most downstream ML stacks consume unordered point lists.

### KITTI

KITTI's Velodyne `.bin` files are flat `float32` buffers with 4 channels per
point: `x`, `y`, `z`, `reflectance` ([KITTI](#ref-kitti)). No per-point
timestamps, no ring index, no return index, no per-point motion compensation
at the file level. KITTI is the historical minimum schema.

This supports keeping `points_ring` and `points_return_index` optional
rather than required: a `LidarData` loaded from a KITTI-style source should
be valid with only `points_xyz`, `points_intensity`, and synthesized
`points_timestamps_ns` populated.

### Ouster SDK

Ouster's `LidarScan` representation in the Ouster SDK preserves both the
range-image topology (`(rings, columns, channels)`) and the projected point
cloud, with explicit per-pixel `t_offset_ns` fields for sub-sweep timing and
per-pixel `signal`, `reflectivity`, `near_ir` channels
([Ouster SDK](#ref-ouster-sdk)). Dual returns are exposed as a separate axis.

This supports including a `points_timestamps_ns` field on `LidarData` and
treating multiple returns as a per-point `points_return_index` rather than a
separate batch axis. It also supports keeping additional channels
(reflectivity, near-infrared) as optional, vendor-shaped extensions rather
than required core fields.

### PCL Point Types and PCD Format

The Point Cloud Library defines a set of canonical point types
(`PointXYZ`, `PointXYZI`, `PointXYZRGB`, `PointXYZINormal`, etc.) widely
echoed in downstream libraries ([PCL](#ref-pcl)). PCD files use a header that
explicitly enumerates `FIELDS`, `SIZE`, `TYPE`, `COUNT`, `WIDTH`, `HEIGHT`,
`VIEWPOINT`, and `POINTS` to describe arbitrary per-point schemas.

This supports field naming choices for `LidarData`'s per-point arrays:
matching PCL vocabulary where possible reduces surprise for users coming
from PCL/ROS ecosystems.

### OpenPCDet Dataset Normalization

OpenPCDet's per-dataset loaders (KITTI, nuScenes, Waymo, Lyft) normalize their
respective source formats into a common in-memory dict layout
([OpenPCDet](#ref-openpcdet)). The common keys are `points: (N, C) float32`
where `C` covers `xyz`, `intensity`, and dataset-specific extras like
`elongation` or `ring`; per-sweep poses are carried separately as part of the
sample metadata.

This supports the structure-of-arrays approach: a community-evolved
"canonical" lidar schema looks more like a small set of typed arrays than a
self-describing schema, and is consumed flat-concatenated when feeding 3-D
detectors. Combined with review feedback that not every consumer organizes
points by sweep, this directly motivated the flat-with-offsets payload
choice in [D1](#d1-payload-shape).

### Proprietary AV SDK

A widely-used proprietary embedded-AV LiDAR SDK was also reviewed for this
design. The specific SDK is intentionally left unnamed; the relevant
structural takeaways are summarized below.

The SDK's decoded-packet representation pairs per-packet metadata (a host
timestamp, the sensor timestamp of the first point, a packet duration, FOV
extents, a scan-complete flag, and a returns table with up to ten entries
per packet) with two per-point variants: a Cartesian form (`x`, `y`, `z`,
intensity) and a spherical form (azimuth, elevation, radius, intensity).

Sensor capabilities are surfaced in a separate properties struct (device
string, spin frequency, points/packets per spin, horizontal/vertical FOV,
per-row calibration tables, an available-returns bitmask, and a validity
bitmask over a set of auxiliary per-point data types). The aux types
include per-point time, the spherical mirror coordinates, radial velocity,
SNR, pulse width, ring/line index, detector and zone identifiers, an
existence probability, a blockage / weather-obstruction flag, a per-point
validity flag, and an invalidity-reason enumeration.

The decoded point cloud also carries a coordinate-frame enum
(sensor / rig / custom) and a motion-compensation sub-struct (a compensated
flag plus the reference timestamp all points were warped to) on every batch.

This supports several choices already in this note and motivates a few
additions:

- **Motion-compensation contract (D3) maps directly.** The proposed
  `motion_compensated` flag mirrors the SDK's compensated flag, and
  `reference_frame` mirrors the SDK's coordinate-frame enum (sensor / rig,
  with custom corresponding to `"world"` / `"map"`). The SDK also carries
  the **reference instant** all points were warped to; this design note
  adds `motion_compensation_timestamp_ns` to `LidarMetadata` to preserve
  it.

- **Multi-return (D5) confirmed.** The SDK supports up to ten returns per
  pulse via its returns table. A `uint8 return_index` covers this range.
  The SDK additionally distinguishes return **type** (first / last /
  strongest, plus variants thereof) from return **index**; how to document
  `points_return_index` ordering semantics is added to open follow-ups.

- **Per-point timing (D6) confirmed required.** The SDK treats per-point
  time as a standard aux channel and provides packet sensor timestamp +
  duration for sources without explicit per-point timing.

- **New optional channels motivated.** `points_validity` (boolean mask,
  mirroring the SDK's per-point validity flag) and `points_radial_velocity`
  (`float32`, mirroring the SDK's per-point radial-velocity channel) are
  added to the optional per-point field list in D6 — they appear in this
  SDK's schema and have no nearby analogue in the other references
  reviewed. The remaining aux channel types (SNR, pulse width, ring/line
  index, detector ID, zone ID, etc.) remain vendor-shaped extensions; they
  should not be promoted to core fields until a concrete consumer needs
  them.

- **Sensor capabilities stay outside `LidarData`.** Properties-struct
  fields (per-row calibration tables, FOV, spin frequency, available
  returns) belong on a future `LidarSensor` rather than the per-batch
  data class — they are static sensor metadata, not per-row payload.

- **Nominal vs. corrected extrinsics.** The SDK separates an as-designed
  (nominal) sensor-to-rig transform from a calibrated (corrected)
  sensor-to-rig transform. The current `extrinsics` slot carries a single
  matrix; how to surface the nominal/corrected distinction (if at all) is
  added to open follow-ups.

**Status**: reviewed against the SDK's decoded-packet, per-point
Cartesian / spherical, sensor-properties, decoded-point-cloud,
motion-compensation, and auxiliary-per-point-data-type public types.

## Open Design Decisions

The choices below shape the structure of `LidarData` and `LidarMetadata`.
Each lists options, trade-offs, and a tentative recommendation.

### D1. Payload shape

Should the per-row ragged point data be represented as a tuple of frozen
per-row containers, or as flat concatenated arrays with a per-row offsets
index?

**Option A — tuple of `LidarSweepData`** (mirrors `MotionVectorData`)

```python
sweeps: tuple[LidarSweepData, ...]   # length N; each row self-contained
```

Pros: matches the existing ragged-side-data pattern in
`MotionVectorData`; each row is independently validatable; trivial to
slice one row out; explicit `LidarSweepData.empty()` placeholder
mirrors `MotionVectorFrameData.empty()` for empty rows.

Cons: `N` Python objects per batch; cross-row vectorized numpy
operations require iteration in Python. Forces a "sweep" mental model
onto consumers that don't use it (e.g., solid-state LiDAR pipelines and
pipelines that query points by time window irrespective of sweep
boundaries).

**Option B — flat + offsets**  (recommended)

```python
points_xyz:        npt.NDArray[np.float32]   # shape (P_total, 3)
points_intensity:  npt.NDArray[np.float32]   # shape (P_total,)
align_offsets:     npt.NDArray[np.int64]     # shape (N+1,)
# points for alignment row i = points_xyz[align_offsets[i]:align_offsets[i+1]]
```

Pros: vectorizes cross-row operations in single numpy calls; closer to
how KITTI/nuScenes/Waymo serialize and how OpenPCDet/MMDetection3D batch
for training; lower per-batch object overhead. The `align_offsets` array
is a partition that consumers can use or ignore — pipelines that work
with "points near an alignment timestamp" without caring about sweep
membership can read the flat arrays directly.

Cons: more bookkeeping during filtering and re-batching; mutation
patterns (drop/append points) require offset rebuilds; no existing
precedent in the sensors package.

**Recommendation: Option B.** Rationale:

- Review feedback observed that not every LiDAR pipeline organizes points
  by sweep. Some pipelines query "points near an alignment timestamp"
  irrespective of sweep boundary; solid-state LiDAR data has no clean sweep
  concept to begin with. The tuple-of-`LidarSweepData` form forces a
  sweep-shaped abstraction on those consumers. The flat layout keeps
  per-row partitioning expressible (via `align_offsets`) without making it
  load-bearing.
- The flat layout matches how every reviewed open AV dataset
  (KITTI / nuScenes / Waymo) serializes points to disk and how every
  reviewed downstream detection framework (OpenPCDet, MMDetection3D)
  batches them internally. The Option-A → Option-B conversion that
  detector-style consumers would otherwise have to perform is avoided.
- Cross-row operations (range filter, intensity threshold, multi-row
  aggregation in a common frame) become single numpy calls instead of
  Python loops over `N` containers.
- "Sweep" remains an interpretation of `align_offsets`, not a required
  structure. Producers whose source data is sweep-shaped (spinning LiDARs,
  KITTI-style file-per-sweep) populate `align_offsets` with sweep
  boundaries. Producers whose source is solid-state or time-window-queried
  populate it accordingly. Consumers that don't care simply ignore it.

Revisit if a future consumer needs strict per-row encapsulation
(e.g., heavy per-row mutation, ragged-tensor framework integration)
that the flat form makes awkward.

### D2. Ego-pose placement

LiDAR consumers commonly need a rig pose at each alignment row's reference
time for motion compensation and multi-row aggregation. Where should ego
pose live? No part of the sensors package currently carries ego pose;
`LidarData` is the first place that needs it.

**Option A — optional `ego_poses` field on `LidarData`**

```python
ego_poses: npt.NDArray[np.float64] | None   # shape (N, 4, 4); rig-to-global
                                            # frame named by LidarMetadata.ego_poses_frame
```

`ego_poses` is interpreted as the rig-to-global-frame trajectory and lives
in a global frame named by `LidarMetadata.ego_poses_frame` (e.g.,
`"world"`, `"map"`, `"odom"`). This frame is independent of
`LidarMetadata.reference_frame`, which describes the frame of
`points_xyz`; the two can be set independently.

Pros: localizes the change to LiDAR; doesn't disturb `CameraData` or
`AlignedFrame`; consumers fetch pose directly from the LiDAR object.

Cons: ego pose is logically a rig property, not a LiDAR property; if
other sensors (radar, etc.) later need ego pose at sweep/scan time, the
field will need to be lifted up, requiring a refactor.

**Option B — add `ego_poses` to `AlignedFrame`**

```python
class AlignedFrame:
    align_timestamps_ns: ...
    sensor_data: Mapping[str, SensorData]
    ego_poses: npt.NDArray[np.float64] | None   # new field
```

Pros: ego pose is shared across all sensors at the bundle level, matching
its physical meaning; only one trajectory stored per bundle.

Cons: many places currently construct `AlignedFrame`; modifies a class
shared across all sensors; couples `AlignedFrame` semantics to
"trajectory of a rig," which may not always be defined (e.g., static
captures).

**Option C — add an `EgoTrajectory` type satisfying `SensorData`**

```python
@attrs.define(hash=False, frozen=True)
class EgoTrajectory:
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    poses: npt.NDArray[np.float64]   # shape (N, 4, 4)
    frame: str   # e.g., "world", "map", "odom"
```

Then carry it inside `AlignedFrame.sensor_data` keyed by id (e.g.,
`"ego"`):

```python
frame.sensor_data["ego"]   # an EgoTrajectory
frame.sensor_data["lidar_top"]   # a LidarData
```

Pros: ego pose participates in the same alignment guarantees as every
other stream; no `AlignedFrame` schema change; cleanly handles multiple
trajectory sources (e.g., ground-truth GPS+IMU vs. SLAM estimate) by
using different keys.

Cons: changes the "what counts as a sensor?" mental model;
`SensorData`-as-trajectory is a slight overload.

**Recommendation: Option A for the first cut, with Option C as the migration
target.** Rationale:

- Option A makes LiDAR self-contained for the first implementation, with no
  cross-class refactoring.
- If a second sensor needs ego pose, or if a non-aligned LiDAR consumer needs
  it, the field can be lifted to a sibling `EgoTrajectory` (Option C) with a
  thin shim. Option A is forward-compatible with Option C.
- Option B is rejected for the first cut because it modifies the shared
  bundle for a single-sensor need.

Revisit at the point a second sensor (radar, second LiDAR with independent
timing, etc.) needs ego pose.

### D3. Motion-compensation contract

When `LidarData` is constructed, are the `xyz` coordinates expected to be in
the sensor frame at the per-point capture time (raw), or already transformed
into a single reference frame for the sweep (motion-compensated)?

**Option A — `LidarData` is agnostic; producer declares**

A `LidarMetadata` field captures the contract:

```python
@attrs.define(hash=False, frozen=True)
class LidarMetadata:
    motion_compensated: bool
    reference_frame: str           # "sensor", "rig", "world", "map", ... — frame of points_xyz
    sensor_model: str | None
    motion_compensation_timestamp_ns: int | None   # required when motion_compensated=True
    ego_poses_frame: str | None    # "world", "map", "odom", ... — required when ego_poses is non-None
```

Pros: supports both pipelines; doesn't force a costly transform at
parse time when the consumer doesn't need it; makes the contract
explicit and validatable. Matches the reviewed proprietary AV SDK
one-for-one (a compensated flag plus a reference timestamp).

Cons: consumers must check the flag and apply or skip the transform
themselves; a misconfigured producer can mislead downstream stages.

**Option B — `LidarData` is always motion-compensated**

Pros: removes ambiguity; downstream perception stages get the clean
shape they want.

Cons: requires the producer to know an ego trajectory at parse time;
loses information for stages that want raw points; not all source
formats provide compensation natively.

**Option C — `LidarData` is always raw**

Pros: closest to source data; producers do less work.

Cons: pushes the transform into every consumer; for multi-sweep
aggregation each consumer reimplements compensation.

**Recommendation: Option A.** Rationale:

- The sensors package convention is to carry source data with explicit
  semantics, not to force a specific transformation contract
  ([ImuData](#ref-imu-data) "Coordinate Frames" section).
- A motion-compensated stage can be added as a derived helper without
  changing `LidarData`.
- The flag plus `reference_frame` together let validators reject
  inconsistent payloads (e.g., `motion_compensated=True` with
  `reference_frame="sensor"` is a contradiction).
- `motion_compensation_timestamp_ns` records the instant all points were
  warped to. It is required when `motion_compensated=True` and is the only
  way a consumer can correctly compose this batch with a downstream pose.

### D4. Range-image representation

Some sensors (Ouster, Waymo) preserve a dense `(rings, azimuth_bins, C)`
range-image representation alongside the point cloud. Should the first
`LidarData` support it?

**Option A — point list only**

Add range-image support later as an optional sibling payload when a consumer
needs it.

**Option B — both, with point list as required and range image as optional**

```python
range_image: RangeImageData | None  # optional sibling like MotionVectorData
```

**Recommendation: Option A for the first cut.** Rationale:

- Most downstream ML stacks (OpenPCDet, MMDetection3D, PyTorch3D batched
  detectors) consume point lists, not range images.
- Range-image support requires per-sensor convention bookkeeping (ring
  ordering, azimuth zero direction, missing-pixel encoding) that varies by
  vendor.
- Range image can be added as an optional sibling payload in v2 without
  disturbing existing v1 consumers, mirroring how `motion_vectors` was added
  to `CameraData`.

### D5. Multi-return handling

Spinning LiDARs (Velodyne, Hesai) and time-of-flight LiDARs (Ouster) can
report multiple returns per laser pulse (first, last, strongest). How is this
represented?

**Option A — `points_return_index` as an optional per-point field**

```python
points_return_index: npt.NDArray[np.uint8] | None   # 0 = first, 1 = second, ...
```

All returns from all pulses are concatenated in the flat point arrays; the
per-point `points_return_index` disambiguates.

**Option B — separate flat point arrays per return**

```python
points_xyz_first_return: npt.NDArray[np.float32]            # (P_first, 3)
points_xyz_last_return:  npt.NDArray[np.float32] | None     # (P_last, 3)
# plus parallel arrays for every other per-point field, per return
```

**Recommendation: Option A.** Rationale:

- Matches Ouster SDK, KITTI, and nuScenes conventions: multi-return points
  share one cloud with a return-index column.
- Keeps the data class shape stable across single-return and dual-return
  sensors.
- Avoids `None` proliferation for the (common) case where only first returns
  are used.

### D6. Required vs optional per-point fields

What is the minimum set of per-point fields a `LidarData` must carry? All
per-point arrays are flat with leading dimension `P_total` (see D1).

**Required (proposed):**

- `points_xyz: (P_total, 3) float32`
- `points_intensity: (P_total,) float32` (or `uint16`; see below)
- `points_timestamps_ns: (P_total,) int64`

**Optional generic:**

- `points_ring: (P_total,) uint16`
- `points_return_index: (P_total,) uint8`
- `points_reflectivity: (P_total,) uint16`
- `points_ambient: (P_total,) uint16` (Ouster near-IR; vendor-specific)
- `points_validity: (P_total,) bool` (False marks points the source flagged
  as invalid / blocked / filtered; mirrors the per-point validity flag in
  the reviewed proprietary AV SDK)
- `points_radial_velocity: (P_total,) float32` m/s (Doppler-capable / FMCW
  lidars only; mirrors the per-point radial-velocity channel in the
  reviewed proprietary AV SDK)

**Recommendation as listed.** Rationale:

- `points_xyz` is universally required.
- `points_intensity` is present in every reviewed reference (KITTI,
  nuScenes, Waymo, Ouster, ROS conventions); making it required reduces
  consumer branching.
- `points_timestamps_ns` is required because the motion-compensation
  contract in D3 needs them; if the source provides only a row-level
  timestamp, a parser should populate this field by interpolation from row
  start/end and document the choice.
- `points_ring` and `points_return_index` are sensor-specific; KITTI lacks
  both. Keeping them optional allows loading minimal-schema sources.
- `points_reflectivity` (calibrated reflectance) is distinct from raw
  `points_intensity` (return strength); some sensors report both. Optional.
- `points_validity` carries an explicit per-point invalid/blocked mask when
  the source provides one. Default policy is still "drop invalid points at
  parser time"; the field exists for consumers (e.g., blockage analysis,
  weather classification) that need to retain them.
- `points_radial_velocity` is only meaningful for FMCW/Doppler-capable
  sensors (e.g., Aeva) and absent on time-of-flight lidars. Kept optional.

Choice of `points_intensity` dtype: `float32` is recommended over `uint16`
for homogeneity with `points_xyz` and downstream tensor conversion, at the
cost of 2 bytes per point versus a packed `uint16`. Reconsider if storage
footprint becomes a constraint.

## Tentative `LidarData`

This section reflects the recommendations above (D1=B flat-with-offsets,
D2=A, D3=A, D4=A, D5=A, D6 as listed). It is tentative and will be updated
as decisions are ratified.

```python
@attrs.define(hash=False, frozen=True)
class LidarMetadata:
    motion_compensated:               bool
    reference_frame:                  str            # "sensor", "rig", "world", "map", ...
    sensor_model:                     str | None = None
    motion_compensation_timestamp_ns: int | None = None   # required when motion_compensated=True
    ego_poses_frame:                  str | None = None   # "world" | "map" | "odom" | ... — required when ego_poses is non-None


@attrs.define(hash=False, frozen=True)
class LidarData:
    __hash__ = None

    # Batch-level (length N = number of alignment rows).
    align_timestamps_ns:  npt.NDArray[np.int64]                # (N,)
    sensor_timestamps_ns: npt.NDArray[np.int64]                # (N,)
    align_offsets:        npt.NDArray[np.int64]                # (N+1,)

    # Flat per-point (length P_total = align_offsets[-1]).
    points_xyz:                 npt.NDArray[np.float32]        # (P_total, 3)
    points_intensity:           npt.NDArray[np.float32]        # (P_total,)
    points_timestamps_ns:       npt.NDArray[np.int64]          # (P_total,)

    metadata: LidarMetadata

    # Optional flat per-point.
    points_ring:            npt.NDArray[np.uint16]  | None = None   # (P_total,)
    points_return_index:    npt.NDArray[np.uint8]   | None = None   # (P_total,)
    points_reflectivity:    npt.NDArray[np.uint16]  | None = None   # (P_total,)
    points_ambient:         npt.NDArray[np.uint16]  | None = None   # (P_total,)
    points_validity:        npt.NDArray[np.bool_]   | None = None   # (P_total,)
    points_radial_velocity: npt.NDArray[np.float32] | None = None   # (P_total,)

    # Optional batch-level.
    ego_poses:  npt.NDArray[np.float64] | None = None    # (N, 4, 4)
    extrinsics: SensorExtrinsics       | None = None
```

Points belonging to alignment row `i` are
`points_xyz[align_offsets[i]:align_offsets[i+1]]` (and similarly for every
other `points_*` array). When a producer's natural unit is a spinning-LiDAR
sweep, `align_offsets` carries sweep boundaries. When the producer instead
queries "points near alignment instant `t_i`," `align_offsets` carries
time-window boundaries. Pipelines that ignore the per-row partitioning can
operate on the flat arrays directly.

### Required Fields (`LidarData`)

| Field | dtype | shape | unit | Required | Notes |
| --- | --- | --- | --- | --- | --- |
| `align_timestamps_ns` | `np.int64` | `(N,)` | ns | yes | Reference timestamps requested from the `SamplingGrid`; strictly increasing. |
| `sensor_timestamps_ns` | `np.int64` | `(N,)` | ns | yes | Source reference timestamps selected for each alignment row; non-decreasing; may repeat under supersampling. |
| `align_offsets` | `np.int64` | `(N+1,)` | — | yes | CSR-style window boundaries: points for alignment row `i` are `points_xyz[align_offsets[i]:align_offsets[i+1]]`. `align_offsets[0] == 0`; `align_offsets[N] == P_total`; non-decreasing. Empty rows are allowed (`align_offsets[i] == align_offsets[i+1]`). |
| `points_xyz` | `np.float32` | `(P_total, 3)` | m | yes | Point coordinates in the frame declared by `LidarMetadata.reference_frame`. |
| `points_intensity` | `np.float32` | `(P_total,)` | sensor-defined | yes | Return strength; sensor calibration determines absolute units. |
| `points_timestamps_ns` | `np.int64` | `(P_total,)` | ns | yes | Per-point capture time. Non-decreasing within each per-row window. |
| `metadata` | `LidarMetadata` | — | — | yes | Carries `motion_compensated` flag, `reference_frame` (frame of `points_xyz`), `motion_compensation_timestamp_ns`, `sensor_model`, and `ego_poses_frame` (global frame of `ego_poses`, required when `ego_poses` is non-`None`). |

### Optional Fields (`LidarData`)

| Field | dtype | shape | unit | Notes |
| --- | --- | --- | --- | --- |
| `points_ring` | `np.uint16` | `(P_total,)` | — | Laser/ring index for spinning sensors; absent for solid-state. Sized for sensors with more than 256 channels (e.g., next-generation Ouster / Hesai). |
| `points_return_index` | `np.uint8` | `(P_total,)` | — | 0 = first return, 1 = second, etc. Absent if only first returns retained. |
| `points_reflectivity` | `np.uint16` | `(P_total,)` | sensor-defined | Calibrated reflectance distinct from raw `intensity`. |
| `points_ambient` | `np.uint16` | `(P_total,)` | sensor-defined | Near-infrared ambient channel (Ouster-style sensors). |
| `points_validity` | `np.bool_` | `(P_total,)` | — | `False` marks points the source flagged invalid / blocked / filtered. |
| `points_radial_velocity` | `np.float32` | `(P_total,)` | m/s | Per-point radial velocity from FMCW / Doppler lidars. |
| `ego_poses` | `np.float64` | `(N, 4, 4)` | — | Rig-to-global-frame transform at each alignment row's reference time. The target global frame is named by `LidarMetadata.ego_poses_frame` and is independent of `LidarMetadata.reference_frame` (which describes the frame of `points_xyz`, not of the rig trajectory). Consumers interpolate to per-point times using `points_timestamps_ns`. |
| `extrinsics` | `SensorExtrinsics` | `(4, 4)` | — | Static sensor-to-rig calibration; one matrix shared across the batch. |

## Timestamp Semantics

`align_timestamps_ns` is the reference timeline requested by the
`SamplingGrid` at the sweep level. It is the timestamp downstream
`AlignedFrame` consumers use to compare rows across sensors.

`sensor_timestamps_ns` is the selected source measurement timestamp for each
alignment row — typically the start, midpoint, or end of the sweep (when
points are sweep-organized) or the query instant (when points are
window-organized). Parsers should document which convention they use and
preserve it consistently across all rows in a batch.

`LidarData` deliberately does not carry a `pts_stream` (producer-native
presentation timestamp) array analogous to `CameraData.pts_stream`. In all
common LiDAR source formats (MCAP, ROS bags, KITTI / nuScenes / Waymo file
readers, and the static-dataset paths), the source's native time domain is
already nanoseconds, so a `pts_stream` field would be a redundant copy of
`sensor_timestamps_ns`. The video-style `time_base` precision problem that
motivates `pts_stream` on `CameraData` does not arise. Re-add the field if
a future LiDAR source format uses a non-ns native time domain where
lossless re-seek would otherwise be impossible.

`points_timestamps_ns` is the per-point capture time, in absolute ns matching
the batch's `sensor_timestamps_ns` time base. Non-decreasing within each
per-row window (i.e., within
`points_timestamps_ns[align_offsets[i]:align_offsets[i+1]]`); the array as a
whole is not required to be globally non-decreasing because adjacent rows
may overlap in time. The relation between row-level `sensor_timestamps_ns[i]`
and per-point times in row `i` is sensor- and parser-specific: typically
`sensor_timestamps_ns[i]` is the sweep start or midpoint, and the per-row
window covers a ~100 ms range for spinning sensors.

If a source provides only a row-level timestamp, the parser should
synthesize `points_timestamps_ns` (uniform interpolation across the row's
duration is acceptable) and document the synthesis.

## Coordinate Frames

`LidarData` does not silently transform coordinate frames. Points in
`points_xyz` are expressed in the frame declared by
`LidarMetadata.reference_frame`. The metadata flag `motion_compensated`
declares whether all points in the batch share a single instantaneous
reference frame (`True`) or whether each point is in the sensor frame at
its own capture time (`False`). The flag applies to every point in the
batch; mixing motion-compensated and raw points in a single `LidarData` is
not supported.

Recommended defaults at the Sensor Library API boundary:

- Right-handed frames.
- SI units: meters for `xyz`, seconds (via ns timestamps) for time.
- Static sensor-to-rig calibration in `LidarData.extrinsics`.
- Time-varying rig-to-global pose (if available) in `LidarData.ego_poses`; the target global frame is named by `LidarMetadata.ego_poses_frame` and is independent of `LidarMetadata.reference_frame`.
- Frame conventions (axis direction, rotation order) follow ROS REP 103 where
  applicable ([REP 103](#ref-ros-rep-103)).

The valid combinations of `motion_compensated` and `reference_frame` are:

| `motion_compensated` | `reference_frame` | Meaning |
| --- | --- | --- |
| `False` | `"sensor"` | Raw — each point in the sensor frame at its own capture time. No ego trajectory required. Cloud is time-sheared by ego motion. |
| `False` | `"rig"` | Each point in the rig frame at its own capture time, obtained from sensor-frame coordinates via the static `extrinsics`. Cloud is time-sheared because the rig frame itself moves with the car. No ego trajectory required. |
| `False` | `"world"` / `"map"` | **Invalid** — rejected by validators. Reaching a fixed global frame requires per-point `T_world_rig(t_capture)` lookup, which is mathematically equivalent to motion compensation; producers should declare `True` and record the chosen anchor instant. |
| `True` | `"sensor"` | **Invalid** — rejected by validators. A single-instant snapshot of a moving frame is ill-defined; producers should target `"rig"` or a global frame instead. |
| `True` | `"rig"` | All points motion-compensated into the rig frame at `motion_compensation_timestamp_ns`. |
| `True` | `"world"` / `"map"` | All points motion-compensated into a global frame; `motion_compensation_timestamp_ns` records the producer's chosen anchor instant. |

## Validation Constraints

Timestamp dtype, length, and ordering constraints are covered in the field
tables. Non-timestamp fields use these constraints:

| Field | Constraint |
| --- | --- |
| all `LidarData` row-indexed arrays | Same leading length `N` as `align_timestamps_ns` (applies to `sensor_timestamps_ns` and `ego_poses`). |
| `align_offsets` | `np.int64`, shape `(N+1,)`, non-decreasing, `align_offsets[0] == 0`, `align_offsets[N] == P_total` (matches the length of every `points_*` array). Empty windows allowed. |
| all `LidarData` point-indexed arrays | Same leading length `P_total` (applies to `points_xyz`, `points_intensity`, `points_timestamps_ns`, and every present optional `points_*` field). |
| `points_xyz` | `np.float32`, shape `(P_total, 3)`, finite values (no NaN/Inf). |
| `points_intensity` | `np.float32`, shape `(P_total,)`, finite values. |
| `points_timestamps_ns` | `np.int64`, shape `(P_total,)`, non-decreasing within each per-row window `[align_offsets[i]:align_offsets[i+1]]`. The array is not required to be globally non-decreasing. |
| `points_ring` | Optional `np.uint16`, shape `(P_total,)`. |
| `points_return_index` | Optional `np.uint8`, shape `(P_total,)`. |
| `points_reflectivity` | Optional `np.uint16`, shape `(P_total,)`. |
| `points_ambient` | Optional `np.uint16`, shape `(P_total,)`. |
| `points_validity` | Optional `np.bool_`, shape `(P_total,)`. |
| `points_radial_velocity` | Optional `np.float32`, shape `(P_total,)`, finite values. |
| `ego_poses` | Optional `np.float64`, shape `(N, 4, 4)`, last row of each `(4, 4)` equals `[0, 0, 0, 1]` within tolerance. The target global frame is named by `LidarMetadata.ego_poses_frame`; it is independent of `LidarMetadata.reference_frame`. |
| `extrinsics` | Optional `SensorExtrinsics`; existing 4x4 `float64` validators apply ([SensorExtrinsics](#ref-sensor-extrinsics)). |
| `LidarMetadata` | `motion_compensated=True` requires `reference_frame != "sensor"` and a non-`None` `motion_compensation_timestamp_ns`. `motion_compensated=False` requires `reference_frame in {"sensor", "rig"}` — `"world"` / `"map"` are rejected because reaching a fixed global frame is mathematically equivalent to motion compensation and should be declared as such. A non-`None` `ego_poses` array on `LidarData` requires a non-`None` `ego_poses_frame` on `LidarMetadata`. |

Follow the existing pattern from `CameraData`: attach shared length-match
validation to the last required field so that `align_offsets`, every
required `points_*` array, and every present optional `points_*` array have
been bound by the time the validator runs; expose read-only views via
`as_readonly_view` without mutating caller-owned arrays.

Per-point `xyz` finite-value validation rejects NaN and Inf to keep the
contract simple. Sources that emit "no return" placeholders (some Velodyne
streams) should drop those points at parser time rather than encoding them as
NaN in `LidarData`.

## Related Structures

`LidarSensor` is out of scope for this note. It is expected to be a future
addition under `cosmos_curator/core/sensors/sensors/` that parses MCAP, ROS
bag, or vendor-native LiDAR streams and produces `LidarData` payloads. A
`McapLidarSensor` would be the most likely first concrete implementation,
mirroring `McapCameraSensor`.

A future `RangeImageData` sibling payload may be added as an optional field
on `LidarData` for sensors and consumers that need to preserve the
range-image topology. See [D4](#d4-range-image-representation).

A future `EgoTrajectory` `SensorData` type may absorb the `ego_poses` field
once a second sensor needs rig pose. See [D2](#d2-ego-pose-placement).

A future motion-compensation helper (transform a
`motion_compensated=False, reference_frame="sensor"` `LidarData` into a
`motion_compensated=True, reference_frame="rig"` one) belongs alongside the
data class but is not part of the data class itself.

Do not introduce a separate `RawLidarData` or `UndecodedLidarPayload` type
until concretely needed; matches the precedent from
[ImuData](#ref-imu-data).

## Implementation Status

Proposed; not yet implemented. The implementation will add:

1. `cosmos_curator/core/sensors/data/lidar_data.py`, with attrs-based
   `LidarData` and `LidarMetadata` classes matching this design note.
2. Shared validation helpers in
   `cosmos_curator/core/sensors/utils/validation.py` for any
   LiDAR-specific patterns (e.g., finite `float32` arrays, non-decreasing
   `int64` within per-row windows, CSR-style offset arrays, 4x4
   transform-batch validators) not already present.
3. Tests under `tests/cosmos_curator/core/sensors/data/test_lidar_data.py`.

## Open Follow-Up Questions

- Should the first `LidarSensor` implementation source data from MCAP, a
  vendor SDK (Ouster, Velodyne), a proprietary AV SDK, or a static dataset
  reader (KITTI, nuScenes, Waymo)? The choice affects which timestamp
  conventions and which optional fields are exercised first.
- When `points_timestamps_ns` is synthesized from a row-level timestamp, is
  uniform interpolation across the row's duration acceptable to downstream
  consumers, or do they require sensor-model-specific timing tables?
- Should `ego_poses` use a quaternion+translation representation
  (`(N, 7) float64`) instead of `(N, 4, 4) float64`? Quaternion form is
  smaller and avoids non-orthonormal `R` matrices but requires conversion
  before composition.
- Should `LidarData` carry a per-batch `frame_id` string (`"lidar_top"`,
  `"lidar_left"`, ...) for multi-LiDAR rigs, or is that pipeline-task
  metadata?
- What sentinel handling, if any, is needed for missing returns? Current
  recommendation is "drop at parser time"; revisit if downstream stages need
  to know which laser/azimuth combos failed.
- Will the first non-AV consumer (e.g., robotics, indoor scanning) require
  a different `reference_frame` vocabulary than the AV-typical
  `"sensor"`/`"rig"`/`"world"`/`"map"` set?
- What ordering does `points_return_index` denote — first-temporal,
  strongest, or producer-defined? At least one reviewed proprietary AV SDK
  distinguishes return **type** (first / last / strongest / variants
  thereof) from return **index** explicitly. Either pick one canonical
  ordering and document it, or add a per-batch field declaring which
  convention the producer used.
- Should `LidarData.extrinsics` carry the nominal (as-designed) transform,
  the calibrated (corrected) transform, or both? Some calibration systems
  (including the reviewed proprietary AV SDK) separate a nominal
  sensor-to-rig transform from a calibrated one; the current single-matrix
  slot has to pick one (likely the calibrated one when available) and loses
  the nominal fallback.
- Validity policy: keep "drop invalid points at parser time" as the default,
  or require parsers to populate the `points_validity` mask when the source
  carries per-point invalid / blockage flags? The latter preserves more
  information for blockage / weather analysis but enlarges the per-batch
  payload.
- Are empty per-row windows (`align_offsets[i] == align_offsets[i+1]`)
  semantically meaningful (e.g., a sensor-dropout row that the alignment
  grid still wants to represent), or should the parser fill them with
  synthesized placeholder points? Default recommendation: allow empty
  windows; consumers that need points-per-row guarantees should validate
  upstream.
- May per-row windows overlap in time (i.e., a point's
  `points_timestamps_ns[j]` falls inside multiple windows when comparing
  against `align_timestamps_ns`)? With CSR offsets the point belongs to
  exactly one window by construction, but a producer choosing
  `align_offsets` for "nearest sweep to each alignment timestamp" could
  legitimately route the same physical point into different rows on
  different runs. Document the producer's policy explicitly rather than
  enforcing one in the data class.
- For producers that have no natural per-row partition (e.g., a single
  "bag of points near alignment instant `t_0`" query), is the canonical
  shape `N == 1`, `align_offsets == [0, P_total]`? Or should `LidarData`
  also accept `N == 0` for empty batches?
- Should a sweep-aware helper (e.g., a lightweight `LidarSweepView`
  iterator providing zero-copy per-row slices) be added alongside
  `LidarData` for the subset of consumers that do want per-sweep access?
  This would not change `LidarData` itself.

## References

- <a id="ref-sensor-data"></a>`SensorData` protocol:
  `cosmos_curator/core/sensors/data/sensor_data.py`
- <a id="ref-aligned-frame"></a>`AlignedFrame`:
  `cosmos_curator/core/sensors/data/aligned_frame.py`
- <a id="ref-camera-data"></a>`CameraData`:
  `cosmos_curator/core/sensors/data/camera_data.py`
- <a id="ref-motion-vector-data"></a>`MotionVectorData` /
  `MotionVectorFrameData` (camera ragged side-data precedent):
  `cosmos_curator/core/sensors/data/camera_data.py`
- <a id="ref-sensor-extrinsics"></a>`SensorExtrinsics`:
  `cosmos_curator/core/sensors/data/extrinsics.py`
- <a id="ref-imu-data"></a>`ImuData` design note:
  `docs/curator/design/sensor-library-imu-data.md`
- <a id="ref-validation-helpers"></a>Sensor validation helpers:
  `cosmos_curator/core/sensors/utils/validation.py`
- <a id="ref-ros-pointcloud2"></a>ROS 2 `sensor_msgs/msg/PointCloud2`:
  <https://docs.ros.org/en/jazzy/p/sensor_msgs/msg/PointCloud2.html>
- <a id="ref-ros-rep-103"></a>ROS REP 103: Standard Units of Measure and
  Coordinate Conventions: <https://ros.org/reps/rep-0103.html>
- <a id="ref-nuscenes"></a>nuScenes data format:
  <https://www.nuscenes.org/nuscenes#data-format>
- <a id="ref-waymo"></a>Waymo Open Dataset:
  <https://github.com/waymo-research/waymo-open-dataset>
- <a id="ref-kitti"></a>KITTI Vision Benchmark Suite:
  <https://www.cvlibs.net/datasets/kitti/>
- <a id="ref-ouster-sdk"></a>Ouster SDK (LidarScan, range-image and
  point-cloud APIs): <https://github.com/ouster-lidar/ouster_sdk>
- <a id="ref-pcl"></a>Point Cloud Library (point types and PCD format):
  <https://pointclouds.org/documentation/>
- <a id="ref-openpcdet"></a>OpenPCDet (per-dataset loaders normalizing
  KITTI / nuScenes / Waymo / Lyft into a common in-memory layout):
  <https://github.com/open-mmlab/OpenPCDet>
- <a id="ref-proprietary-av-sdk"></a>Proprietary AV SDK: reviewed against
  the SDK's public types covering the decoded-packet container, per-point
  Cartesian and spherical representations, sensor properties, return-type
  enumeration, auxiliary per-point data types, invalidity-reason
  enumeration, decoded point-cloud wrapper, motion-compensation
  sub-struct, point-cloud layout mapping, and coordinate-frame
  enumeration.
