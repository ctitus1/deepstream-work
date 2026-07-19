# Pipeline benchmarking

`benchmark_pipeline.py` measures the deployed graph end to end: sink FPS plus
per-stage latency percentiles for detect / assess / convert / osd / sink.

```bash
# maximum throughput: a local file is not clock-paced
python3 validation/benchmark/benchmark_pipeline.py \
  --stream streams/dtc-d4-trimmed.mp4 --duration 30 --json outputs/validation/bench.json

# what the live system actually sustains
python3 validation/benchmark/benchmark_pipeline.py --duration 60
python3 validation/benchmark/benchmark_pipeline.py --no-assessment
```

It builds on the package timing code rather than re-measuring it:
`deepstream_yolo.timing.TimeLog` supplies the per-frame stage timestamps (an
infinite print interval turns it into a pure recorder) and
`deepstream_yolo.assessment_runtime.AssessmentTiming` supplies the same
`detect_ms` / `assess_ms` the parser and ROS apps print in their `ASSESS` lines.
Both are reported, so a mismatch between them is itself a signal.

## Reading the output

- **Percentiles, not means.** p99 is what a downstream consumer feels. A mean
  of 8 ms with a p99 of 60 ms is a stuttering pipeline, and the mean hides it.
- **`--warmup-frames 30` (default)** drops the first frames. TensorRT context
  setup, CUDA allocation, and the decoder's first keyframe are not steady state.
- **RTSP is clock-paced.** `build_pipeline()` sets `live-source`,
  `drop-on-latency`, and leaky queues for RTSP, so FPS is capped by the sender
  and overload shows up as dropped frames rather than rising latency. Benchmark
  a local file for a throughput ceiling and RTSP for a real-world number.
- **`--display` caps FPS.** The sink only syncs to the clock when displaying;
  the default headless `fakesink` run is the throughput measurement.
- GPU utilization is sampled from `nvidia-smi` when it is present and skipped
  silently otherwise (`--no-gpu-sampling` to force it off). Sampling at 0.5 s
  is coarse: treat it as a saturation indicator, not a profile.

## The lower-level comparison: trtexec

`benchmark_pipeline.py` measures the whole graph — decode, mux, inference, OSD,
sink. To find out how much of the budget is inference alone, run the engine
directly:

```bash
trtexec --loadEngine=models/<model>_640_<W>x<H>.onnx_b1_gpu0_fp16.engine \
  --iterations=200 --avgRuns=50 --useSpinWait --noDataTransfers
```

That is the ceiling: raw engine throughput with no decode, no format
conversion, no OSD, no display. Compare the two:

- trtexec fast, pipeline slow: the cost is outside inference. Look at the
  per-stage percentiles for the guilty stage (`convert` and `osd` are the usual
  suspects, along with JPEG branches when appsinks are enabled).
- trtexec and pipeline both slow: it is the engine. Check precision
  (`network-mode=2` is FP16), batch size, and input resolution.
- `detect` latency well above trtexec's per-inference time: batching or queue
  behaviour, not the kernel.

`nvidia-smi dmon -s um` alongside a run gives a second opinion on whether the
GPU is saturated or waiting.
