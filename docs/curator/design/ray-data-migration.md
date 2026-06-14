# Ray Data Migration Plan

## Summary

This document tracks the remaining work for Ray Data to support the production video split-and-annotate, semantic
dedup, and shard-dataset workflows that currently run on Xenna-backed pipelines.

Ray Data is the target implementation path for this workflow. It is not a feature-for-feature port of Xenna. Xenna is
useful as operating history, a production fallback, a performance baseline, and a comparison source for retained
semantics.

The current Ray Data split/caption implementation lives under
[`cosmos_curator/pipelines/ray_data`](../../../cosmos_curator/pipelines/ray_data). The public entry point is the
`cosmos-curator pipeline` command group for config templates, validation, rendering, schema inspection, and preset
inspection. The focused tests under
`tests/cosmos_curator/pipelines/ray_data` are the source of truth for what is implemented today.

## Scope

This plan covers the video split -> dedup -> shard workflow. Image, examples, and direct `CuratorStage` workflows need
separate migration plans and policy decisions.

Retained behavior means behavior required by active split -> dedup -> shard workloads. Everything else is either a
Ray Data redesign choice or deferred compatibility work.

## Operating Rules

- Build against the Ray Data contract directly.
- Keep Pydantic configs, Lance-native outputs, schema-validated summaries, Ray-native observability, and a smaller CLI as
  intentional Ray Data surface.
- Use Xenna for comparison and rollback, not as a porting checklist.
- Preserve Xenna behavior only when it is retained by an active workflow or accepted as a compatibility requirement.
- Keep input normalizers at the boundary. They may convert retained source formats or orchestration inputs into Ray Data
  configs, but should not become shared Xenna/Ray Data orchestration or engine bridges.
- Treat managed-service payloads, presigned artifacts, legacy file layouts, Xenna-specific flags, stage mechanics, and
  unowned model variants as deferred unless required by a retained workflow or test harness.
- Do not remove or retire Xenna behavior as part of this plan. Retirement needs a separate project decision and human
  communication plan.

## Native Contract

| Area | Ray Data v1 target |
| --- | --- |
| Input | Versioned JSON/YAML configs validated with Pydantic and resolved from packaged defaults, packaged presets, user config, and small overrides for split, caption/filter/embedding blocks, dedup, and shard. |
| CLI | A small public surface for validation, resolved-config rendering, execution, schema inspection, and preset inspection. |
| Output | Clip media, Lance datasets for metadata, captions, scores, embeddings, tabular features, and dedup decisions; schema-validated summary JSON for run counters and operational status; retained training-dataset artifacts. |
| Workflow | Split writes clip media plus Lance metadata/caption/filter/score/embedding datasets. Dedup consumes Lance embeddings and writes dedup decisions. Shard consumes Lance metadata/captions, clip-media references, and optional dedup decisions, then downloads clip media and writes retained training outputs. |
| Runtime | Long local, Slurm, and multi-node workloads with expected restart, retry, failure-reporting, and throughput behavior. NVCF stays useful for end-to-end and performance test harnesses, but local, Slurm, and multi-node behavior are higher priority. |
| Environments | Use the merged GPU-capable `default` Pixi environment for common split, caption, embedding, shard, and orchestration dependencies. Keep specialized Pixi environments and per-stage `runtime_env` or `conda_env_name` usage where isolation remains required, including retained `legacy-transformers`, `cuml` for semantic dedup's cuML, RAFT, and NCCL stack, `seedvr`, `sam3`, and any separate `paddle-ocr` environment. |

Caption windows are internal by default and are represented in Lance or retained training outputs only when required by
the workflow contract.

## Work Queue

| Area | Target | Done when |
| --- | --- | --- |
| Entry point and config | Adopt Pydantic config models for Ray Data split, then extend the same pattern to dedup and shard. Execution consumes resolved config, with retained orchestration inputs normalized at the boundary. | Each config-backed workflow is registered with the shared pipeline CLI for template generation, validation, resolved-config rendering, schema inspection, preset inspection, and execution. |
| Xenna baseline capture | Capture dated Xenna reference runs early under `benchmarks/` for each retained workflow. | Each baseline records dataset manifest, hardware, launch environment, image or lockfile versions, git SHAs, resolved config, wall time, throughput, GPU-hours, and relevant output counters. |
| Input and ingest | Represent retained source types in the Pydantic schema and keep source-specific listing and credential behavior in input normalizers. | Schema coverage and normalizer tests cover source metadata propagation, remux outputs, invalid-video handling, existing-output detection, resume decisions, and failed-video counters. |
| Storage profile selection | Preserve explicit profile selection for retained endpoints such as input videos, input video lists, output writers, embeddings, and semantic-dedup decisions. | Readers, `StorageWriter`, and Lance storage options pass the endpoint profile explicitly instead of relying on the default profile. |
| Splitting and transcode | Preserve retained fixed-stride and TransNetV2 clip-selection semantics, stable clip IDs, and summary accounting. Use FFmpeg CPU decode for TransNetV2 in Ray Data v1. | Retained transcode modes are implemented and failed clips are accounted for explicitly. |
| Lance output and readers | Make Lance plus schema-validated summary JSON the native contract for metadata, captions, scores, embeddings, dedup decisions, and downstream tool reads. | First-party readers can read Lance before any workflow relies on Lance-only output. Legacy file layouts are tracked separately unless promoted to retained requirements. |
| Captioning | Expose retained caption controls, model instructions, sampling settings, and model choices through config and presets. | Retained caption metadata is stored in Lance, including outcome vocabulary, token accounting, quality signals, and retained window representation if required. Second-pass captioning, enhancement, and previews are ported only when an active workflow needs them. |
| Filtering and classification | Port retained clip filtering needed by active split workflows, especially LLM/VLM semantic filtering and video-type classification. | Filter-window inputs, model instructions, category allow/block behavior, rejection reasons, score-only mode, filtered-clip membership, and summary counters are preserved for Qwen semantic and classifier decisions. Motion, aesthetic, artificial-text, and other non-LLM filters are added only when retained. |
| Embeddings | Port shared frame extraction plus embedding backends for the retained model set. | Embeddings are written to Lance for dedup and downstream first-party tools. |
| Semantic dedup | Consume Lance embeddings directly, run retained clustering and duplicate extraction with appropriate GPU placement, and write schema-defined dedup decisions plus summary JSON for shard. | The retained multi-GPU KMeansMG path runs in a per-GPU RAFT actor pool with `PixiRuntimeEnv("cuml")`, orchestrated by Ray alongside the Ray Data workflow rather than expressed as a map or flat-map transform. |
| Shard dataset | Consume Lance metadata/captions, retained clip-media locations, and Ray Data dedup decisions; download clip media; pack retained training-dataset artifacts. | Schema-validated summary accounting is written, and only binning, tar sizing, T5, resume, and drop behavior needed by active training consumers is preserved. |
| Optional feature blocks | Add tracking, event captioning, multi-camera, and other optional blocks only when active workflows need them. | Adopted blocks reuse shared decode, model, and writer helpers where useful and preserve retained output and summary accounting. |
| Observability | Ship Ray Data-native profiling, tracing, row sampling, checkpointing, and comparison artifacts for the retained workflow. | Documented commands or run outputs produce per-stage timing, trace files, sampled rows or checkpoints, and a retained-output comparison report from a small fixture. Xenna stage mechanics are recreated only for active debugging needs. |
| Deployment and launch integration | Validate Ray Data execution through existing local and Slurm launch paths. | Slurm proves bare `ray.init()` attachment to the launched cluster, preserves credential environment plumbing such as `COSMOS_S3_PROFILE_PATH`, and keeps Prometheus service-discovery generation working during bring-up. NVCF Ray Data dispatch is deferred until retained NVCF test-harness requirements define the payload contract and entry point. |
| Multi-node model setup | Validate distributed model cache behavior, local NVMe copy behavior, per-node model availability, the merged default Pixi environment target, and retained specialized Pixi environments. | Multi-node runs prove model weights are present on all nodes and Ray Data actor placement scales. |
| Resilience | Add bounded I/O retries, idempotent write markers, poison-pill quarantine, and explicit failed-video/clip summaries. | Retry limits are configurable, reruns avoid duplicate outputs, repeatedly failing inputs get a quarantine artifact, and summaries distinguish skipped, failed, quarantined, and written videos or clips. |

## Transition and Cutover

Ray Data being the priority implementation path does not move production workloads by itself. Until a retained split,
dedup, or shard workflow has an explicit cutover decision, Xenna remains the production fallback for that workflow.

- New implementation work for the retained video workflow targets Ray Data first.
- Production correctness fixes land in Xenna, Ray Data, or both depending on which active path is affected. This plan
  does not create a blanket Xenna bugfix freeze.
- A workflow can move after its relevant work-queue items are complete, validation passes for the intended launch
  environment, retained output comparisons and benchmark margins are accepted, and operators have config mapping and
  runbook updates.
- Keep last-known-good Xenna entry points, configs, and compatibility artifacts available during transition.
- If Ray Data misses accepted correctness, performance, or operational criteria, route production back to Xenna while
  fixing Ray Data.

The default performance gate is Ray Data throughput at least 90% of the captured Xenna baseline and GPU-hours no more
than 110% of that baseline for the same retained outputs. Any wider margin needs an explicit workflow-owner acceptance
note. If Xenna, dependencies, or hardware drift enough to invalidate the reference, replace it with a new dated baseline
rather than comparing against an unstated moving target.

## Validation Evidence

Validation should run as features land and provide the evidence for each cutover decision:

- Unit tests for every Ray Data transform and writer shape.
- CLI and config-tooling tests for template generation, validation, render output, schema inspection, preset inspection,
  and config-backed execution for each registered workflow.
- Small local end-to-end tests for split-only, split plus caption, split plus filtering/classification, and split plus
  embedding.
- End-to-end tests for split -> dedup and split -> dedup -> shard on small fixtures.
- GPU end-to-end tests for TransNetV2 plus Qwen captioning/filtering in the merged `default` environment.
- Optional NVCF end-to-end and performance runs for test harnesses that still use that environment.
- Multi-node runs that prove model weights are present on all nodes and Ray Data actor placement can scale.
- Output comparisons against Xenna for retained split, dedup, and shard semantics.
- Mixed-profile storage tests that use distinct profiles for input, output, video-list, and dedup-decision endpoints
  where the retained workflow exposes them.
- Skip/resume and idempotency tests that pre-create retained outputs, rerun the workflow, prove skipped inputs are
  counted, and prove no duplicate clips, Lance rows, dedup decisions, or shard samples are written.
- Observability fixture tests or smoke runs that produce profiling, tracing, row-sample/checkpoint, and comparison-report
  artifacts.
- Failure-injection tests for storage read errors, write errors, FFmpeg failures, empty videos, no-scene videos, and
  caption, filtering, embedding, dedup, and shard errors.
- Performance benchmarks on the same hardware, datasets, configs, and output contract as the captured Xenna baseline.

## Remaining Design Decisions

- Exact Pydantic schemas for split/filter/dedup/shard configs, Lance datasets, dedup decisions, and summary JSON.
- Scope of retained orchestration input normalization and any NVCF-specific test-harness support.
- Retained model, caption/filter backend, transcode, and window behavior after lower-priority variants are reviewed.
- Shard output layout for active training consumers after the window audit.
- Stage replay/save replacement using Ray Data checkpointing and sampled row dumps.
- Scope and timing of any Xenna removal beyond the video split/dedup/shard workflow, including the required project
  decision and communication plan.
