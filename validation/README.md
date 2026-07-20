# Validation

Diagnostic tools for the DeepStream pipeline. None of these are part of the
runtime path; they exist to answer "is the deployed system doing what the model
and the stream say it should".

| Area | Tool | Question it answers |
| --- | --- | --- |
| `accuracy/` | `dump_detections.py`, `score_detections.py` | Does the deployed pipeline detect what the model detects? See the three-rung ladder in `accuracy/README.md`. |
| `benchmark/` | `benchmark_pipeline.py` | What FPS and per-stage latency does the deployed graph sustain? |
| `timestamps/` | `print_rtsp_timestamps.py`, `rtsp_nvinfer_timestamp_test.py` | Does stream wall-clock time reach the far end of the pipeline? |

## Timestamps

`print_rtsp_timestamps.py` is plain GStreamer with no DeepStream elements: it
answers whether the RTSP source carries `GstReferenceTimestampMeta` at all,
which requires the server to be sending RTCP sender reports.

```bash
python3 validation/timestamps/print_rtsp_timestamps.py --frames 100
python3 validation/timestamps/print_rtsp_timestamps.py --uri rtsp://127.0.0.1:8560/test --quiet
```

`rtsp_nvinfer_timestamp_test.py` bisects the DeepStream side: it probes the same
buffer after depay, parse, decode, `nvstreammux`, and `nvinfer`, and names the
first stage where wall-clock time stops being recoverable. Pre-mux stages read
the reference meta directly; post-mux stages use the shared `frame_timestamp()`
resolver, so `nvstreammux`'s own `ntp_timestamp` counts as survival.

```bash
python3 validation/timestamps/rtsp_nvinfer_timestamp_test.py --frames 200 --print-stages all
```

It uses the newest generated config from `configs/generated/`; if that directory
is empty it says to run `scripts/setup_and_export_yolo.sh` first. Pass
`--infer-config` to pin a specific one.

Both default to `paths.DEFAULT_RTSP_URL`, so start the stream first:

```bash
scripts/start_rtsp_stream.sh streams/my-video.mp4
```

## Interpreters

`dump_detections.py`, `benchmark_pipeline.py`, and both timestamp tools need
the DeepStream container interpreter (gi, pyds). `score_detections.py` needs
`.venv-yolo` instead, because pycocotools lives there — it deliberately imports
nothing from GStreamer.
