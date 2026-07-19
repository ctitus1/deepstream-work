# tracking/nvmot_custom — custom NvMOT low-level library

**Placeholder.** `nvmot_custom_tracker.cpp` is a skeleton: every entry point is
present with its real signature and every body returns `NvMOTStatus_Error` with a
TODO. It implements no tracking. The build was verified once (see
[Build](#build)); the library it produces has never been loaded by `nvtracker`
and would fail at `NvMOT_Init` by design if it were.

Do not attempt this route before Route A (tuning the shipped trackers) has been
measured — see [`../README.md`](../README.md) and [`../configs/README.md`](../configs/README.md).

## How nvtracker loads a custom library

`nvtracker` is a thin GStreamer wrapper. It dlopen's the shared object named by
its `ll-lib-file` property, dlsym's a fixed set of C symbols, and drives them.
Swapping the library swaps the entire tracking algorithm while `nvtracker` keeps
doing the batching, buffer transforms, and metadata attachment.

```python
tracker.set_property("ll-lib-file", str(TRACKING_DIR / "nvmot_custom" / "libnvds_custom_tracker.so"))
tracker.set_property("ll-config-file", str(TRACKING_DIR / "configs" / "custom_tracker.yml"))
```

`ll-config-file` is handed straight through to our library as
`NvMOTConfig::customConfigFilePath` — DeepStream does not parse it. Its format is
whatever we decide; the stock library happens to use YAML.

## The API

Header (read it, it is the authority):

    /opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvdstracker.h

Reference implementation to compare behaviour against:

    /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so

### Required entry points — real signatures

All six are declared inside `extern "C" { ... }` in the header, so a `.cpp` that
includes the header and defines them gets C linkage automatically. Do not add a
second `extern "C"` wrapper and do not mangle the names.

```c
NvMOTStatus NvMOT_Query(uint16_t customConfigFilePathSize,
                        char *pCustomConfigFilePath,
                        NvMOTQuery *pQuery);

NvMOTStatus NvMOT_Init(NvMOTConfig *pConfigIn,
                       NvMOTContextHandle *pContextHandle,
                       NvMOTConfigResponse *pConfigResponse);

NvMOTStatus NvMOT_Process(NvMOTContextHandle contextHandle,
                          NvMOTProcessParams *pParams,
                          NvMOTTrackedObjBatch *pTrackedObjectsBatch);

NvMOTStatus NvMOT_RetrieveMiscData(NvMOTContextHandle contextHandle,
                                   NvMOTProcessParams *pParams,
                                   NvMOTTrackerMiscData *pTrackerMiscData);

NvMOTStatus NvMOT_RemoveStreams(NvMOTContextHandle contextHandle,
                                NvMOTStreamId streamIdMask);

void NvMOT_DeInit(NvMOTContextHandle contextHandle);
```

Note `NvMOT_DeInit` returns `void`, not a status. `NvMOT_RetrieveMiscData` is not
in the minimal five but must exist if `NvMOT_Query` advertises `supportPastFrame`,
`outputTerminatedTracks`, or `outputShadowTracks`.

`NvMOTContextHandle` is `struct NvMOTContext *` — an opaque forward declaration.
The header never defines `struct NvMOTContext`; **we** define it in our
translation unit and it holds all of our per-context state.

### Call order

1. `NvMOT_Query` — before any context exists. Reports capabilities and the input
   buffer format/size the library wants. `nvtracker` uses the answer to set up the
   buffer transforms it will perform for us, so this must be filled in honestly.
2. `NvMOT_Init` — once per `nvtracker` element. Allocates our context and returns
   the handle. Every later call passes that handle back.
3. `NvMOT_Process` — once per batched buffer, for the life of the pipeline.
4. `NvMOT_RetrieveMiscData` — after `NvMOT_Process` on the same batch, when the
   optional outputs were advertised.
5. `NvMOT_RemoveStreams` — batch mode only, when a source goes away. Only called
   when processing is quiesced.
6. `NvMOT_DeInit` — teardown. The handle is dead afterwards.

### NvMOT_Query

Fills `NvMOTQuery`. The fields that determine what `nvtracker` does for us:

- `computeConfig` — OR of `NVMOTCOMP_GPU` / `NVMOTCOMP_CPU` / `NVMOTCOMP_PVA`.
- `numTransforms` — how many transformed buffers we want per frame. `0` means we
  need no pixels at all (an IOU/Kalman-only tracker); `1` is the normal case for
  anything visual. Max is `NVMOT_MAX_TRANSFORMS` (4). If this is 0, `nvtracker`
  skips the color conversion entirely — a large saving.
- `colorFormats[0]` — required input color format (an `NvBufSurfaceColorFormat`).
- `memType` — `NvBufSurfaceMemType`, e.g. device memory for a GPU tracker.
- `maxTargetsPerStream`, `maxShadowTrackingAge`.
- `batchMode` — `NvMOTBatchMode_Batch` and/or `NvMOTBatchMode_NonBatch`. At least
  one must be supported or it is `NvMOTBatchMode_Error`.
- `supportPastFrame`, `outputTerminatedTracks`, `outputShadowTracks`,
  `maxTrajectoryBufferLength` — gate `NvMOT_RetrieveMiscData`.
- `outputReidTensor`, `reidFeatureSize`, `outputTrajectory`, `outputVisibility`,
  `outputFootLocation`, `outputConvexHull`, `maxConvexHullSize` — optional
  user-meta outputs. Leave all off for a first motion-only tracker.
- `contextHandle` — present in the struct; not used by a stateless query.

The custom config path is passed in so the query answer may depend on the config
(e.g. "config says motion-only, so `numTransforms = 0`"). Consulting it is optional.

### NvMOT_Init

`pConfigIn` gives `maxStreams`, `computeConfig`, `numTransforms` and the
per-transform batch configuration (`bufferType`, `maxWidth`, `maxHeight`,
`maxPitch`, `maxSize`), `miscConfig.gpuId`, and `customConfigFilePath` /
`customConfigFilePathSize`.

The header notes `NvMOTConfig` **must be deep-copied** if retained —
`perTransformBatchConfig` and `customConfigFilePath` are borrowed pointers.

Fill in every field of `*pConfigResponse` (`summaryStatus`, `computeStatus`,
`transformBatchStatus`, `miscConfigStatus`, `customConfigStatus`), not just the
summary. Return `NvMOTStatus_OK` and set `*pContextHandle` on success.

Gotcha: in `NvMOTMiscConfig` the logging callback is declared as a *nested
typedef*, `typedef void (*logMsg)(int, const char *, ...);`, which is a member
type, not a member variable. There is no function pointer to call. Do not write
code expecting `miscConfig.logMsg` to exist. (This also makes the header C++-only —
a typedef inside a struct is not valid C. Compile with `g++`.)

### NvMOT_Process

The core. `pParams->frameList` is an array of `pParams->numFrames`
`NvMOTFrame`s. Per frame:

- `streamID`, `seq_index` (0 .. maxStreams-1, stable for the stream's lifetime),
  `frameNum`, `srcFrameWidth`, `srcFrameHeight`, `timeStamp` / `timeStampValid`
- `doTracking` — skip the frame if false
- `reset` — reset this stream's tracking state
- `bufferList` — array of `numBuffers` `NvBufSurfaceParams *`, the transforms we
  asked for in the query
- `objectsIn` — `NvMOTObjToTrackList` of detector boxes: `detectionDone`, `list`,
  `numAllocated`, `numFilled`. Each `NvMOTObjToTrack` has `classId`, `bbox`
  (`NvMOTRect{x, y, width, height}` in float pixels), `confidence`, `doTracking`,
  and `pPreservedData`.

Output goes into `*pTrackedObjectsBatch`, which the **caller allocates**. We fill,
never allocate:

- `pTrackedObjectsBatch->list[i]` is an `NvMOTTrackedObjList` with `streamID`,
  `frameNum`, `valid`, `list`, `numAllocated`, `numFilled`.
- Write at most `numAllocated` entries and set `numFilled` to what was written.
  Overrunning `numAllocated` is the obvious way to corrupt memory here.
- Each `NvMOTTrackedObj`: `classId`, `trackingId` (uint64, the persistent id that
  becomes `NvDsObjectMeta.object_id`), `bbox`, `confidence`, `age` (track length
  in frames), and `associatedObjectIn` — a pointer back into the input
  `objectsIn` list, or `NULL` for a shadow/predicted target with no detection
  this frame. Setting `associatedObjectIn` is what lets `nvtracker` carry the
  detector's object meta forward.
- `reid`, `visibility`, `ptImgFeet`, `ptWorldFeet`, `convexHull` are only
  meaningful if the query advertised them.

Bounding boxes in and out are scaled to the resolution of the **first** input
transform buffer, not the source frame.

### NvMOT_RemoveStreams

Removes every stream where `(streamId & streamIdMask) == streamIdMask`. Batch
mode only. Free per-stream state here.

## Build

`Makefile` follows the `CUDA_VER` convention used by
`scripts/build_yolo_parser.sh` (which resolves 12.6 for DeepStream 7.1) and by
DeepStream-Yolo's own Makefile:

```bash
CUDA_VER=12.6 make -C tracking/nvmot_custom
```

Output: `libnvds_custom_tracker.so` in this directory (gitignored).

This was run once inside `deepstream-work:7.1` to confirm the skeleton compiles
against the real header and that the six entry points come out unmangled:

```
$ nm -D --defined-only libnvds_custom_tracker.so | grep ' T NvMOT'
T NvMOT_DeInit
T NvMOT_Init
T NvMOT_Process
T NvMOT_Query
T NvMOT_RemoveStreams
T NvMOT_RetrieveMiscData
```

The link flags are only sufficient because the skeleton never touches
`NvBufSurface`. Expect to add `-lnvbufsurface` / `-lnvbufsurftransform` (from
`/opt/nvidia/deepstream/deepstream/lib`) as soon as `NvMOT_Query` reports
`numTransforms > 0`.

Nothing copies the `.so` into `lib/` — `lib/` currently holds only
`libnvdsinfer_custom_impl_Yolo.so`, and `paths.py` exposes `LIB_DIR` if we later
want to install it there alongside. `paths.py` has no tracking constants yet; add
them there rather than hardcoding paths in the pipeline.

## TODO

- [ ] Decide the algorithm before writing a line of it (Kalman + Hungarian on
      IOU is the honest starting point).
- [ ] Define `struct NvMOTContext` and the per-stream state map keyed by `seq_index`.
- [ ] Pick the custom config format and parse it in `NvMOT_Init`.
- [ ] Implement `NvMOT_Query` honestly — especially `numTransforms`.
- [ ] Add the NvBufSurface link flags once the tracker reads pixels.
- [ ] Load the built `.so` through `nvtracker` and confirm the dlsym succeeds
      before writing any algorithm.
