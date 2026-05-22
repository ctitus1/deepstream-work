#!/usr/bin/env python3
import argparse
import datetime as dt
import time
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import clip
from clip.model import build_model

import pyds


PROJECT_DIR = Path("/home/user/deepstream-work")
NTP_TO_UNIX_SECONDS = 2_208_988_800
NS_PER_SEC = 1_000_000_000
PERSON_CLASS_ID = 0


def ntp_ns_to_unix_ns(ntp_ns: int) -> int:
    return ntp_ns - NTP_TO_UNIX_SECONDS * NS_PER_SEC


def ns_to_utc(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9, tz=dt.timezone.utc).isoformat(timespec="milliseconds")


def seconds(ns: int) -> str:
    if ns == Gst.CLOCK_TIME_NONE or ns < 0:
        return "NONE"
    return f"{ns / 1e9:.3f}s"


def get_stream_time(buffer: Gst.Buffer):
    ref_meta = buffer.get_reference_timestamp_meta(None)
    if ref_meta is None:
        return None
    return ntp_ns_to_unix_ns(ref_meta.timestamp)


def find_yolo_config() -> Path:
    matches = sorted(
        (PROJECT_DIR / "configs").glob("config_infer_primary_yolo12x_640_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError("No yolo12x 640 config found")
    return matches[0]


class InjuryClassifier:
    def __init__(self, model_path: Path, device: str):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        clip_state = {
            k.removeprefix("clip_model."): v
            for k, v in ckpt.items()
            if k.startswith("clip_model.")
        }

        self.model = build_model(clip_state).to(device).eval()

        self.heads = nn.ModuleDict()
        head_names = sorted({k.split(".")[1] for k in ckpt if k.startswith("heads.")})

        for name in head_names:
            w = ckpt[f"heads.{name}.weight"]
            b = ckpt[f"heads.{name}.bias"]
            head = nn.Linear(w.shape[1], w.shape[0])
            head.weight.data.copy_(w)
            head.bias.data.copy_(b)
            self.heads[name] = head

        self.heads.to(device).eval()
        self.device = device

        self.preprocess = transforms.Compose([
            transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    @torch.inference_mode()
    def classify(self, crops):
        if not crops:
            return []

        x = torch.stack([self.preprocess(crop) for crop in crops]).to(self.device)

        feats = self.model.encode_image(x)
        feats = feats.float()

        results = []
        for i in range(len(crops)):
            item = {}
            for name, head in self.heads.items():
                probs = torch.softmax(head(feats[i:i + 1]), dim=1)[0]
                cls = int(torch.argmax(probs).item())
                prob = float(probs[cls].item())
                item[name] = (cls, prob)
            results.append(item)

        return results


def make_element(factory: str, name: str):
    elem = Gst.ElementFactory.make(factory, name)
    if elem is None:
        raise RuntimeError(f"Missing GStreamer element: {factory}")
    return elem


def on_rtsp_pad_added(_src, pad, depay):
    caps = (pad.get_current_caps() or pad.query_caps(None)).to_string()
    if "application/x-rtp" in caps:
        pad.link(depay.get_static_pad("sink"))


def clamp_box(rect, frame_w: int, frame_h: int):
    x1 = max(0, min(frame_w - 1, int(rect.left)))
    y1 = max(0, min(frame_h - 1, int(rect.top)))
    x2 = max(0, min(frame_w, int(rect.left + rect.width)))
    y2 = max(0, min(frame_h, int(rect.top + rect.height)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def make_output_probe(classifier: InjuryClassifier, print_empty: bool):
    def _probe(_pad, info, _data):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        recv_ns = time.time_ns()
        stream_ns = get_stream_time(buffer)

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list

        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            frame_rgba = pyds.get_nvds_buf_surface(hash(buffer), frame_meta.batch_id)
            frame_h, frame_w = frame_rgba.shape[:2]

            crops = []
            boxes = []
            confs = []

            obj_list = frame_meta.obj_meta_list
            while obj_list:
                obj = pyds.NvDsObjectMeta.cast(obj_list.data)

                if obj.class_id == PERSON_CLASS_ID:
                    box = clamp_box(obj.rect_params, frame_w, frame_h)
                    if box is not None:
                        x1, y1, x2, y2 = box
                        crop_rgb = np.array(frame_rgba[y1:y2, x1:x2, :3], copy=True, order="C")
                        crops.append(Image.fromarray(crop_rgb))
                        boxes.append(box)
                        confs.append(float(obj.confidence))

                obj_list = obj_list.next

            if crops or print_empty:
                if stream_ns is not None:
                    latency_ms = (recv_ns - stream_ns) / 1e6
                    ts = (
                        f"local={ns_to_utc(recv_ns)} "
                        f"stream={ns_to_utc(stream_ns)} "
                        f"latency={latency_ms:.1f}ms"
                    )
                else:
                    ts = f"local={ns_to_utc(recv_ns)} stream=NONE latency=NONE"

                injury_results = classifier.classify(crops)

                parts = []
                for box, det_conf, injury in zip(boxes, confs, injury_results):
                    injury_str = ",".join(
                        f"{name}=class{cls}:{prob:.2f}"
                        for name, (cls, prob) in injury.items()
                    )
                    parts.append(f"box={box} yolo={det_conf:.2f} {injury_str}")

                print(
                    f"{ts} pts={seconds(buffer.pts)} persons={len(crops)} "
                    + " | ".join(parts),
                    flush=True,
                )

            frame_list = frame_list.next

        return Gst.PadProbeReturn.OK

    return _probe


def on_message(_bus, msg, loop):
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print(f"ERROR: {err}")
        print(f"DEBUG: {dbg}")
        loop.quit()
    elif msg.type == Gst.MessageType.EOS:
        loop.quit()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="rtsp://127.0.0.1:8554/rgb")
    parser.add_argument("--latency", type=int, default=0)
    parser.add_argument("--yolo-config", default=None)
    parser.add_argument("--injury-model", default="models/injury.pt")
    parser.add_argument("--print-empty", action="store_true")
    args = parser.parse_args()

    Gst.init(None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = InjuryClassifier(PROJECT_DIR / args.injury_model, device)

    yolo_config = Path(args.yolo_config) if args.yolo_config else find_yolo_config()
    print(f"yolo config={yolo_config}", flush=True)
    print(f"injury model={PROJECT_DIR / args.injury_model}", flush=True)
    print(f"injury device={device}", flush=True)

    pipeline = Gst.Pipeline.new("rtsp-yolo-injury-timestamp-test")

    src = make_element("rtspsrc", "src")
    depay = make_element("rtph264depay", "depay")
    parse = make_element("h264parse", "parse")
    dec = make_element("nvv4l2decoder", "decoder")
    queue = make_element("queue", "queue")
    mux = make_element("nvstreammux", "mux")
    infer = make_element("nvinfer", "infer")
    convert = make_element("nvvideoconvert", "convert")
    caps = make_element("capsfilter", "caps")
    sink = make_element("fakesink", "sink")

    src.set_property("location", args.uri)
    src.set_property("latency", args.latency)
    src.set_property("drop-on-latency", True)
    src.set_property("protocols", "tcp")
    src.set_property("ntp-sync", True)
    src.set_property("add-reference-timestamp-meta", True)

    queue.set_property("leaky", 2)
    queue.set_property("max-size-buffers", 1)
    queue.set_property("max-size-bytes", 0)
    queue.set_property("max-size-time", 0)

    mux.set_property("batch-size", 1)
    mux.set_property("width", 3840)
    mux.set_property("height", 2160)
    mux.set_property("live-source", 1)
    mux.set_property("batched-push-timeout", 0)

    infer.set_property("config-file-path", str(yolo_config))
    convert.set_property("nvbuf-memory-type", 3)  # CUDA unified memory for pyds.get_nvds_buf_surface

    caps.set_property(
        "caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
    )

    sink.set_property("sync", False)
    sink.set_property("qos", False)

    for elem in (src, depay, parse, dec, queue, mux, infer, convert, caps, sink):
        pipeline.add(elem)

    src.connect("pad-added", on_rtsp_pad_added, depay)

    depay.link(parse)
    parse.link(dec)
    dec.link(queue)
    queue.get_static_pad("src").link(mux.request_pad_simple("sink_0"))
    mux.link(infer)
    infer.link(convert)
    convert.link(caps)
    caps.link(sink)

    caps.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        make_output_probe(classifier, args.print_empty),
        None,
    )

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
