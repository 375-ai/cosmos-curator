# Monocular 3D Scene Reconstruction Design

## Scope

This document is the design for the optional 3D scene layer in the video splitting
pipeline
([cosmos_curator/pipelines/video/splitting_pipeline.py](../../../cosmos_curator/pipelines/video/splitting_pipeline.py)).
Behind `--scene3d`, each clip gains a metric 3D reconstruction that the MCAP writer emits
as two extra channels, renderable directly in Foxglove's 3D panel:

| Topic | Schema | Cardinality |
|---|---|---|
| `/scene/background` | `foxglove.PointCloud` | one coloured cloud per clip |
| `/scene/objects` | `foxglove.SceneUpdate` | one message per sampled frame |

It also promotes `/camera/camera-info` and `/tf-static` from the documented
placeholders (`fx = fy = 1.0`, identity transform) to a real estimated camera model.

It covers why the camera model is **derived** rather than required, how the two sizing
estimators combine, and the stage/function contracts.

## Motivation: no calibration exists, so derive it

Projecting a pixel to a metric world point needs `K` (intrinsics) and `R, t` (the camera's
pose above the ground). A prototype that reads those from a source recording cannot run
here: curator ingests arbitrary video, and
[mcap_schemas.py](../../../cosmos_curator/pipelines/video/read_write/mcap_schemas.py)
says outright that no calibration source exists. Fed the placeholders, every ground
projection degenerates — with `t[2] == 0` the ray-plane intersection has no solution at
all.

The way out is to invert the dependency. A **metric** depth model (Depth Anything V2
Metric) predicts absolute metres, so a single frame yields a metrically-scaled point
cloud in the camera frame with no external input. Fitting a plane to that cloud recovers
what the calibration was supposed to supply:

- the plane's distance from the origin **is** the camera height;
- the plane's normal **is** the gravity direction, giving pitch and roll.

Relative depth cannot do this — the plane fit would have arbitrary scale, so the height
would be meaningless. Choosing a metric checkpoint is what makes the whole feature
generic, and it also deletes the usual relative-depth machinery: no affine fit against a
"trusted" near-ground band, no flat-road prior, no per-scene tuning.

Yaw stays unobservable from one static view, so the map frame is gravity-aligned but not
geo-referenced: `+X` is wherever the camera points. That is a labelling convention, not a
loss — nothing downstream needs true north.

## Pipeline position

`Scene3DStage` sits after SAM3 tracking (whose per-frame boxes become the cuboids) and
before `ClipWriterStage`. That placement is forced: it needs `clip.encoded_data`, which
the clip writer drops.

```
… → SAM3BBoxStage → Scene3DStage → ClipWriterStage → McapWriterStage
      clip.sam3_frames ─┘              │                    │
                                       │                    ├─ /scene/background
                        scene3d/<uuid>.json (optional)      ├─ /scene/objects
                                                            └─ /camera/camera-info + /tf-static
```

`--scene3d` works without `--sam3`: the background cloud and calibration are still
produced, only `/scene/objects` is empty. The pipeline warns rather than erroring,
because a reconstructed backdrop is useful on its own.

## Per-clip flow

One decode, **one depth inference**, everything else pure NumPy:

1. **Decode** at `--scene3d-target-fps` via `decode_clip_at_fps`, shared with the tracking
   stage, so frames carry real presentation timestamps.
2. **Background plate** — a temporal median over ≤20 sampled frames. Anything that moves
   is erased, leaving the static scene.
3. **Depth** — one forward pass on the plate, returning metres.
4. **Calibrate** — RANSAC a ground plane over the lower half of the image; derive `K` from
   the assumed FOV and `R, t` from the plane.
5. **Background cloud** — back-project, cull, voxel-downsample to `--scene3d-max-points`,
   pack into the wire layout.
6. **Objects** — lift tracked boxes to cuboids.

### Why the plate is downscaled before the median

Frames are resized to the depth input size *first*. Medianing full 4K frames stacks
`20 × 2160 × 3840 × 3` bytes and `np.median` promotes that to float64, peaking at several
GB — the single largest memory spike in the naive version. Downscaling first makes it
~16 MB, and the plate doubles as the point cloud's colour source.

### Why object depth comes from the plate

Because movers are erased from the plate, a lookup at an object's ground-contact pixel
(bottom-centre of its box) returns the depth of the **floor behind the object** — which is
what a ground-contact point should measure — and costs nothing extra. It assumes the
camera is static within a clip; `--scene3d-object-depth per-frame` runs depth on every
sampled frame for panning or moving cameras, at N times the cost.

## Two estimators for cuboid size

A monocular view measures an object's width and height directly but can *never* see its
length. Neither estimator alone is sufficient, so `--scene3d-dimension-mode auto`
combines them:

- **Class prior** — a keyword table (`priors.py`) matched against the free-text detection
  label, longest key first so `"a white pickup truck"` resolves to `pickup truck` rather
  than the much larger `truck`. This is the only source of length.
- **Depth-derived** — the 2D box converted to metres at the object's measured range
  (`width_m ≈ (x2-x1)·Z/fx`). Needs no class knowledge, so it covers every label the table
  does not.

`auto` takes the prior when the label matches and measures otherwise; `prior` and `depth`
force one or the other.

## Heading is a property of a trajectory, not a frame

Object lifting is two-pass. Pass 1 lifts each detection's ground-contact pixel and
accumulates one trajectory per track id. Between passes, a single least-squares velocity
is fitted per track; pass 2 renders every frame using that one heading. Fitting per frame
would let the arrow flip whenever two consecutive positions jitter.

Two thresholds keep the output honest: tracks lifted in fewer than
`--scene3d-min-track-points` frames are dropped as detector ghosts, and a track that never
travels `--scene3d-min-net-displacement-m` is drawn axis-aligned with no arrow rather than
given a meaningless yaw. Arrow colour splits on the SVD principal axis of all track
velocities — an unsupervised two-colouring of opposing flows that needs no domain
knowledge and degrades to uniform below two moving tracks.

## Module contracts

Only `scene3d_stage.py` imports torch; everything else is pure NumPy and unit-tests on
CPU, mirroring the `track_funcs` split in [object tracking](object-tracking.md).

| Module | Responsibility |
|---|---|
| `calibration.py` | `Calib` (projections, cached horizon) and `estimate_calibration` |
| `lifting.py` | background plate, back-projection, voxel downsample, cloud packing |
| `object_lift.py` | tracked boxes → per-frame cuboid records |
| `priors.py` | label → dimensions/colour table |
| `detection_source.py` | `DetectionSource` protocol + the SAM3 adapter |
| `scene3d_stage.py` | data router: decode → depth → calibrate → cloud → objects |
| `scene3d_builders.py` | `Scene3DConfig` → stage |

`DetectionSource` is the seam for a future standalone detector: nothing in `object_lift`
knows what produced the boxes, and the source reports its own frame size so boxes are
rescaled onto the depth map's grid.

## Resolutions: estimate on the plate, publish against the video

Depth runs on a downscaled plate (`--scene3d-depth-long-side`, default 700), so the
camera model is *estimated* in plate pixels — which is correct, because that is the grid
the lifting maths operates on. But `/camera/camera-info` is published on the same
`frame_id` as `/camera/image-raw`, which carries the full-resolution clip, and a viewer
pairs the two. `Calib.to_payload(width, height)` therefore rescales `K` (and `P`) to the
video's resolution before emitting; the pose is resolution-independent and passes through
unchanged. Without that rescale a 1080p clip publishes a 700x394 calibration with
`fx ~= 606` next to a 1920x1080 image stream — off by the 2.7x downscale factor.

The same asymmetry applies to the two intrinsics knobs. `--scene3d-hfov-deg` is an angle
and needs no conversion, but `--scene3d-focal-px` is a pixel quantity: it is interpreted
in the **source video's** pixels (the number a user actually knows about their camera) and
scaled onto the plate before the fit.

## Failure behaviour

Failure is per-clip and never fatal. A clip that cannot be decoded or reconstructed tags
`clip.errors` and flows on with its other artefacts intact; a failure *after* the cloud is
committed only costs the cuboids. When the ground fit is rejected — too few inliers, an
implausible height, or a normal that reads as a wall rather than a floor — the stage falls
back to the CLI camera values and records `scene3d_calibration` anyway.

The payload's `source` field distinguishes the three cases, because "not estimated" alone
conflates a failure with a user's choice:

| `source` | Meaning |
|---|---|
| `ground-fit` | measured from the depth cloud |
| `overridden` | height, tilt and roll all supplied, so no fit was attempted |
| `fallback` | a fit was attempted and rejected |

Only `fallback` tags `clip.errors["scene3d_calibration"]`. Overriding *some* angles keeps
the fitted value for the rest — passing `--scene3d-camera-tilt-deg` does not silently zero
the roll of a rolled camera.

## Transport and lifetime

The packed cloud rides on `Clip.scene3d_background` as `LazyData`, for zero-copy PEP-574
transport. Its size is set by the depth plate and the sampling stride, not by
`--scene3d-max-points`: at the default 700 px plate the cloud is ~17k points (~0.5 MB),
so the 200k budget is a ceiling that only binds if the plate is enlarged. `drop_clip_intermediate_data` keeps the
three `scene3d_*` fields while `keep_mcap_payloads` is set (so they survive
`ClipWriterStage`) and releases them once the MCAP writer is done.

No gate is needed in the writer itself: channels register lazily on first message, so the
3D topics simply do not exist in a run without `--scene3d`.

Session metadata is written once, from chunk 0. The calibration records are not: chunks are
written independently and merged, so a later chunk cannot know whether chunk 0 had a
reconstruction to publish. Any chunk carrying a calibration re-emits `/camera/camera-info`
and `/tf-static`, which keeps its geometry anchored to a real map frame even when every
clip in chunk 0 failed. The transforms are static and identical, so the repetition is the
ordinary latched-static-transform pattern rather than conflicting data.

## Known limitations

- **Scale follows the assumed FOV.** `--scene3d-hfov-deg` sets the focal length and
  therefore the absolute scale; `--scene3d-focal-px` pins it when the true value is known.
- **Yaw is arbitrary** (see above).
- **Monocular.** Occluded regions are unrecoverable, and the outdoor checkpoint saturates
  at 80 m, so the far field is inferred rather than measured.
- **Lens distortion is ignored** — `D` is emitted as zeros. This is the main cause of
  far-ground warping and the obvious next improvement.
- **`background` object depth assumes a static camera within a clip.**
