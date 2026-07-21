# Integration tests

Message-level checks against a **running** pipeline. Unlike `../tests.py`
(GPU/ROS-free, runs anywhere), these import `rclpy` and `cdcl_umd_msgs` and
drive the real services, so they only run inside the container:

```bash
docker compose -f ds_ros_pipeline/compose.yaml up -d ds-ros-pipeline
docker exec ds-ros-pipeline bash -lc \
  'source /opt/ros/${DS_ROS_DISTRO}/setup.bash && source $CDCL_ROS_SETUP \
   && python3 ds_ros_pipeline/integration/verify_pipes.py'
```

Both exit non-zero on failure.

Run them from the **pipeline** container, not the bridge — they need the
same ROS distro the node publishes with.

| Script | Purpose |
|---|---|
| `verify_pipes.py` | Asserts the three request pipes end to end. Exit code is the verdict. |
| `evidence.py` | Same three pipes, but prints every published field (class, bbox, flags, annotation heads) before the verdict, for eyeballing what actually went on the wire. |

## What they cover

- **detect** — one `TargetBoxArray` per batched frame on
  `/uas4/target_detections`, boxes present, `annotations` empty,
  `use_for_assessment=false`.
- **detect+assess** — same arrays with the 8 `clip_rgb_*` heads filled in.
- **capture/vlm** — one `TargetBoxArray` on `/uas4/target_detections/vlm`,
  no annotations, `use_for_assessment=true`, nothing leaked onto the batch
  topic, and the array still reaches a late (post-call) subscriber. It is
  the *only* thing that pipe publishes.
- **detection scope** — every published box is class `person` and at or
  above `detect.min_confidence`. Set `DS_MIN_CONFIDENCE` to match if the
  node was launched with a non-default threshold.
- **the detection index** — the invariant that array position *is* the
  DeepStream detection index: annotations must land position by position on
  exactly the boxes the SGIE operates on, and the array must hold distinct,
  positive-area boxes (no duplicated or permuted entries).

## Why they retry

The test clip loops through stretches with nothing in frame, so a single
empty batch is not a verdict — it is a statement about the video. Both
scripts re-run a pipe until it lands on frames that actually exercise the
path (and, for the assess pipe, on frames the SGIE will operate on) before
asserting. A genuine regression still fails, because it fails on every
attempt.
