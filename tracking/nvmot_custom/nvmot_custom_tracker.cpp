/*
 * PLACEHOLDER custom NvMOT low-level tracker library.
 *
 * Skeleton only. Every entry point below has the signature nvtracker dlsym's
 * (see /opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvdstracker.h)
 * and every body is a TODO that returns a not-implemented status. This library
 * tracks nothing. Loading it via nvtracker's ll-lib-file will fail at
 * NvMOT_Init by design, rather than producing bogus tracks.
 *
 * The entry points are declared inside extern "C" in nvdstracker.h, so these
 * definitions inherit C linkage. Do not add another extern "C" wrapper.
 *
 * Build: CUDA_VER=12.6 make   ->  libnvds_custom_tracker.so   (untested)
 */

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "nvdstracker.h"

/*
 * nvdstracker.h forward declares "struct NvMOTContext" and typedefs
 * NvMOTContextHandle to a pointer to it, but never defines it. The definition
 * is ours. Everything the tracker remembers between NvMOT_Process calls lives
 * here.
 *
 * TODO: replace these placeholder members with the real per-stream state
 * (track list, Kalman filter state, next tracking id, association scratch).
 */
struct NvMOTContext
{
    /* Deep copy of the NvMOTConfig handed to NvMOT_Init. The header states the
     * config must be deep-copied to be retained: perTransformBatchConfig and
     * customConfigFilePath are borrowed pointers owned by the caller. */
    NvMOTConfig config;
    std::vector<NvMOTPerTransformBatchConfig> batchConfig;
    std::string customConfigPath;

    /* TODO: per-stream state, keyed by NvMOTFrame::seq_index (0..maxStreams-1). */
    std::map<uint32_t, int> perStreamState;

    /* TODO: monotonically increasing source of NvMOTTrackedObj::trackingId. */
    uint64_t nextTrackingId = 0;
};

/**
 * Report capabilities and input requirements. Called before any context exists.
 *
 * nvtracker uses the answer to decide which buffer transforms to perform on our
 * behalf, so numTransforms / colorFormats[0] / memType must be honest. A
 * motion-only tracker that never looks at pixels sets numTransforms = 0 and
 * nvtracker skips the color conversion entirely.
 *
 * pCustomConfigFilePath is the ll-config-file path, provided so the answer may
 * depend on the config. Consulting it is optional.
 */
NvMOTStatus NvMOT_Query(uint16_t customConfigFilePathSize,
                        char *pCustomConfigFilePath,
                        NvMOTQuery *pQuery)
{
    (void) customConfigFilePathSize;
    (void) pCustomConfigFilePath;

    if (pQuery == nullptr)
    {
        return NvMOTStatus_Error;
    }

    /* TODO: fill pQuery, at minimum:
     *   computeConfig       NVMOTCOMP_GPU | NVMOTCOMP_CPU
     *   numTransforms       0 for motion-only, 1 for anything visual
     *   colorFormats[0]     required NvBufSurfaceColorFormat when numTransforms > 0
     *   memType             NvBufSurfaceMemType for the input buffers
     *   maxTargetsPerStream, maxShadowTrackingAge
     *   batchMode           NvMOTBatchMode_Batch and/or NvMOTBatchMode_NonBatch
     * and leave the optional outputs (outputReidTensor, outputTrajectory,
     * outputVisibility, outputFootLocation, outputConvexHull, supportPastFrame,
     * outputTerminatedTracks, outputShadowTracks) false until implemented --
     * advertising one obligates NvMOT_RetrieveMiscData to populate it.
     *
     * Zero the struct first; the caller does not guarantee it is clean. */
    memset(pQuery, 0, sizeof(*pQuery));

    return NvMOTStatus_Error;
}

/**
 * Create a tracking context for a batch of streams.
 *
 * On success: allocate the context, store a deep copy of *pConfigIn, set
 * *pContextHandle, fill every field of *pConfigResponse, return NvMOTStatus_OK.
 */
NvMOTStatus NvMOT_Init(NvMOTConfig *pConfigIn,
                       NvMOTContextHandle *pContextHandle,
                       NvMOTConfigResponse *pConfigResponse)
{
    if (pConfigIn == nullptr || pContextHandle == nullptr || pConfigResponse == nullptr)
    {
        return NvMOTStatus_Error;
    }

    /* Report every sub-status, not just the summary -- nvtracker logs them
     * individually and a half-filled response is unreadable at the call site. */
    pConfigResponse->summaryStatus = NvMOTConfigStatus_Error;
    pConfigResponse->computeStatus = NvMOTConfigStatus_Error;
    pConfigResponse->transformBatchStatus = NvMOTConfigStatus_Error;
    pConfigResponse->miscConfigStatus = NvMOTConfigStatus_Error;
    pConfigResponse->customConfigStatus = NvMOTConfigStatus_Error;
    *pContextHandle = nullptr;

    /* TODO:
     *   1. validate pConfigIn->computeConfig against what NvMOT_Query advertised
     *   2. deep copy pConfigIn->perTransformBatchConfig (numTransforms entries)
     *      and pConfigIn->customConfigFilePath (customConfigFilePathSize bytes)
     *   3. parse the custom config file
     *   4. reserve per-stream state for pConfigIn->maxStreams
     *   5. set the CUDA device from pConfigIn->miscConfig.gpuId
     *
     * Note: NvMOTMiscConfig declares logMsg as a nested typedef, not a member.
     * There is no logging callback to call.
     */

    return NvMOTStatus_Error;
}

/**
 * Track one batch.
 *
 * pParams->frameList holds pParams->numFrames NvMOTFrame entries. Detector
 * boxes arrive in frame.objectsIn; tracked output is written into the
 * caller-allocated pTrackedObjectsBatch. Boxes in and out are in the coordinate
 * space of the first input transform buffer, not the source frame.
 */
NvMOTStatus NvMOT_Process(NvMOTContextHandle contextHandle,
                          NvMOTProcessParams *pParams,
                          NvMOTTrackedObjBatch *pTrackedObjectsBatch)
{
    if (contextHandle == nullptr || pParams == nullptr || pTrackedObjectsBatch == nullptr)
    {
        return NvMOTStatus_Error;
    }

    /* TODO: for each frame in pParams->frameList:
     *   - honour frame.reset (drop this stream's state) and frame.doTracking
     *   - predict existing targets forward to frame.frameNum
     *   - associate frame.objectsIn.list[0 .. numFilled) with those predictions
     *   - update matched targets, age unmatched ones, spawn new tracks
     *   - write results into pTrackedObjectsBatch->list[i]:
     *       streamID, frameNum, valid = true
     *       list[j].trackingId    persistent id -> NvDsObjectMeta.object_id
     *       list[j].classId, bbox, confidence, age
     *       list[j].associatedObjectIn  pointer back into frame.objectsIn.list,
     *                                   or NULL for a shadow/predicted target
     *       numFilled = number written, and NEVER exceed numAllocated
     *
     * The batch and its per-stream lists are allocated by the caller. Fill them;
     * do not allocate or free them.
     */

    /* Mark every output slot invalid so a caller that ignores the error status
     * cannot mistake uninitialized memory for tracks. */
    for (uint32_t i = 0; i < pTrackedObjectsBatch->numAllocated; ++i)
    {
        pTrackedObjectsBatch->list[i].valid = false;
        pTrackedObjectsBatch->list[i].numFilled = 0;
    }
    pTrackedObjectsBatch->numFilled = 0;

    return NvMOTStatus_Error;
}

/**
 * Return past-frame / terminated-track / shadow-track history for this batch.
 *
 * Only called when NvMOT_Query advertised supportPastFrame,
 * outputTerminatedTracks or outputShadowTracks. Until one of those is turned on
 * this stays a stub.
 */
NvMOTStatus NvMOT_RetrieveMiscData(NvMOTContextHandle contextHandle,
                                   NvMOTProcessParams *pParams,
                                   NvMOTTrackerMiscData *pTrackerMiscData)
{
    (void) pParams;

    if (contextHandle == nullptr || pTrackerMiscData == nullptr)
    {
        return NvMOTStatus_Error;
    }

    /* TODO: populate pPastFrameObjBatch / pTerminatedTrackBatch /
     * pShadowTrackBatch (NvDsTargetMiscDataBatch) to match what NvMOT_Query
     * advertised. */

    return NvMOTStatus_Error;
}

/**
 * Drop every stream matching the mask: (streamId & streamIdMask) == streamIdMask.
 * Batch mode only, and only called once processing is quiesced.
 */
NvMOTStatus NvMOT_RemoveStreams(NvMOTContextHandle contextHandle,
                                NvMOTStreamId streamIdMask)
{
    (void) streamIdMask;

    if (contextHandle == nullptr)
    {
        return NvMOTStatus_Error;
    }

    /* TODO: free per-stream state for every matching stream. */

    return NvMOTStatus_Error;
}

/**
 * Retire a context. Returns void -- there is no way to report failure.
 * The handle must not be used again.
 */
void NvMOT_DeInit(NvMOTContextHandle contextHandle)
{
    /* TODO: release the deep-copied config, per-stream state, and any device
     * memory before deleting the context. */
    delete contextHandle;
}
