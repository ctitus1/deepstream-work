#!/usr/bin/env python3
import argparse
import datetime as dt
import time
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from clip.model import build_model
import pyds


PROJECT_DIR = Path("/home/user/deepstream-work")
NTP_TO_UNIX_SECONDS = 2_208_988_800
NS_PER_SEC = 1_000_000_000
PERSON_CLASS_ID = 0


LABELS = {
    "severe_hemorrhage": {0: "hem-", 1: "hem+", 2: "hem?"},
    "respiratory_distress": {0: "resp-", 1: "resp+", 2: "resp?"},
    "trauma_head": {0: "head-", 1: "head+", 3: "head?"},
    "trauma_torso": {0: "torso-", 1: "torso+", 3: "torso?"},
    "trauma_upper_ext": {0: "upper-", 1: "upper+", 2: "upper_amp", 4: "upper?"},
    "trauma_lower_ext": {0: "lower-", 1: "lower+", 2: "lower_amp", 4: "lower?"},
    "alertness_ocular": {0: "eyes_open", 1: "eyes_closed", 2: "eyes_nt", 3: "eyes?"},
    "person_type": {0: "manikin", 1: "human", 2: "type?"},
}

# Label layout:
#   person 0: 0.79
#   human hem+ resp-
#   head+ torso-
#   upper+ lower- eyes_open
DISPLAY_ROWS = [
    ["person_type", "severe_hemorrhage", "respiratory_distress"],
    ["trauma_head", "trauma_torso"],
    ["trauma_upper_ext", "trauma_lower_ext", "alertness_ocular"],
]


def ntp_ns_to_unix_ns(ntp_ns: int) -> int:
    return ntp_ns - NTP_TO_UNIX_SECONDS * NS_PER_SEC


def ns_to_hhmmss(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9, tz=dt.timezone.utc).strftime("%H:%M:%S.%f")[:-3] + "Z"


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


def conf_color(conf: float, lo: float = 0.2, hi: float = 0.8):
    t = max(0.0, min(1.0, (conf - lo) / (hi - lo)))
    # BGR: red at low conf, green at high conf.
    return (0, int(255 * t), int(255 * (1.0 - t)))



class GstDisplay:
    def __init__(self, title: str = "assess"):
        self.title = title
        self.pipeline = None
        self.appsrc = None
        self.width = None
        self.height = None

    def _start(self, width: int, height: int):
        self.width = width
        self.height = height

        sink = "ximagesink"
        if Gst.ElementFactory.find(sink) is None:
            sink = "xvimagesink" if Gst.ElementFactory.find("xvimagesink") else "autovideosink"

        self.pipeline = Gst.parse_launch(
            'appsrc name=src is-live=true do-timestamp=true block=false format=time '
            f'caps=video/x-raw,format=BGR,width={width},height={height},framerate=0/1 ! '
            'queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! '
            'videoconvert ! '
            f'{sink} sync=false'
        )
        self.appsrc = self.pipeline.get_by_name("src")
        self.pipeline.set_state(Gst.State.PLAYING)

    def push(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if self.pipeline is None:
            self._start(w, h)
        elif w != self.width or h != self.height:
            return

        data = frame_bgr.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        self.appsrc.emit("push-buffer", buf)

    def stop(self):
        if self.appsrc is not None:
            self.appsrc.emit("end-of-stream")
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

class InjuryClassifier:
    def __init__(self, model_path: Path, device: str):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        clip_state = {
            k.removeprefix("clip_model."): v
            for k, v in ckpt.items()
            if k.startswith("clip_model.")
        }

        self.model = build_model(clip_state).to(device).eval()

        head_names = sorted({k.split(".")[1] for k in ckpt if k.startswith("heads.")})
        self.heads = nn.ModuleDict()

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
        feats = self.model.encode_image(x).float()

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


def injury_label(name: str, injury: dict) -> str:
    if name not in injury:
        return ""
    cls, _prob = injury[name]
    return LABELS.get(name, {}).get(cls, f"{name}=class{cls}")


def label_text(box_id: int, yolo_label: str, yolo_conf: float, injury: dict) -> list[str]:
    lines = [f"{yolo_label} {box_id}: {yolo_conf:.2f}"]

    for row in DISPLAY_ROWS:
        labels = [injury_label(name, injury) for name in row]
        labels = [x for x in labels if x]
        if labels:
            lines.append("  ".join(labels))

    return lines


def draw_label(img, x1, y1, lines, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.70
    thick = 1
    pad = 5
    line_h = 22

    width = max(cv2.getTextSize(line, font, scale, thick)[0][0] for line in lines) + 2 * pad
    height = len(lines) * line_h + 2 * pad

    y_top = max(0, y1 - height)
    x_right = min(img.shape[1] - 1, x1 + width)

    cv2.rectangle(img, (x1, y_top), (x_right, y_top + height), (0, 0, 0), -1)
    cv2.rectangle(img, (x1, y_top), (x_right, y_top + height), color, 1)

    y = y_top + pad + 12
    for line in lines:
        cv2.putText(img, line, (x1 + pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h


def make_probe(classifier: InjuryClassifier, args, display: GstDisplay):
    def _probe(_pad, info, _data):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        sys_ns = time.time_ns()
        frame_ns = get_stream_time(buffer)
        frame_s = ns_to_hhmmss(frame_ns) if frame_ns is not None else "NONE"
        sys_s = ns_to_hhmmss(sys_ns)
        age_s = f"{(sys_ns - frame_ns) / 1e9:.3f}s" if frame_ns is not None else "NONE"

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            frame_rgba = pyds.get_nvds_buf_surface(hash(buffer), frame_meta.batch_id)
            frame_rgb = np.array(frame_rgba[:, :, :3], copy=True, order="C")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            crops = []
            boxes = []
            confs = []

            frame_h, frame_w = frame_rgb.shape[:2]
            obj_list = frame_meta.obj_meta_list
            while obj_list:
                obj = pyds.NvDsObjectMeta.cast(obj_list.data)
                conf = float(obj.confidence)

                if obj.class_id == PERSON_CLASS_ID and conf >= args.yolo_conf_min:
                    box = clamp_box(obj.rect_params, frame_w, frame_h)
                    if box is not None:
                        x1, y1, x2, y2 = box
                        crop_rgb = np.array(frame_rgb[y1:y2, x1:x2, :3], copy=True, order="C")
                        crops.append(Image.fromarray(crop_rgb))
                        boxes.append(box)
                        confs.append(conf)

                obj_list = obj_list.next

            results = classifier.classify(crops)

            for box_id, (box, conf, injury) in enumerate(zip(boxes, confs, results)):
                x1, y1, x2, y2 = box
                color = conf_color(conf, args.conf_red, args.conf_green)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)

                lines = label_text(box_id, "person", conf, injury)
                draw_label(frame_bgr, x1, y1, lines, color)

            header = f"frame={frame_s} sys={sys_s} age={age_s} pts={seconds(buffer.pts)} n={len(crops)}"
            cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(frame_bgr, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            if args.display_width > 0:
                scale = args.display_width / frame_bgr.shape[1]
                out = cv2.resize(frame_bgr, (args.display_width, int(frame_bgr.shape[0] * scale)))
            else:
                out = frame_bgr

            display.push(out)

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
    parser.add_argument("--yolo-conf-min", type=float, default=0.20)
    parser.add_argument("--conf-red", type=float, default=0.20)
    parser.add_argument("--conf-green", type=float, default=0.80)
    parser.add_argument("--display-width", type=int, default=0)
    args = parser.parse_args()

    Gst.init(None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = InjuryClassifier(PROJECT_DIR / args.injury_model, device)

    yolo_config = Path(args.yolo_config) if args.yolo_config else find_yolo_config()
    print(f"yolo config={yolo_config}", flush=True)
    print(f"injury model={PROJECT_DIR / args.injury_model}", flush=True)
    print(f"device={device}", flush=True)

    pipeline = Gst.Pipeline.new("rtsp-yolo-injury-viewer")

    src = make_element("rtspsrc", "src")
    depay = make_element("rtph264depay", "depay")
    parse = make_element("h264parse", "parse")
    dec = make_element("nvv4l2decoder", "decoder")
    queue = make_element("queue", "pre-mux-queue")
    mux = make_element("nvstreammux", "mux")
    infer = make_element("nvinfer", "infer")
    convert = make_element("nvvideoconvert", "convert")
    caps = make_element("capsfilter", "caps")
    clip_queue = make_element("queue", "clip-queue")
    sink = make_element("fakesink", "sink")

    src.set_property("location", args.uri)
    src.set_property("latency", args.latency)
    src.set_property("drop-on-latency", True)
    src.set_property("protocols", "tcp")
    src.set_property("ntp-sync", True)
    src.set_property("add-reference-timestamp-meta", True)

    for q in (queue, clip_queue):
        q.set_property("leaky", 2)
        q.set_property("max-size-buffers", 1)
        q.set_property("max-size-bytes", 0)
        q.set_property("max-size-time", 0)

    mux.set_property("batch-size", 1)
    mux.set_property("width", 3840)
    mux.set_property("height", 2160)
    mux.set_property("live-source", 1)
    mux.set_property("batched-push-timeout", 0)

    infer.set_property("config-file-path", str(yolo_config))

    convert.set_property("nvbuf-memory-type", 3)
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))

    sink.set_property("sync", False)
    sink.set_property("qos", False)

    for elem in (src, depay, parse, dec, queue, mux, infer, convert, caps, clip_queue, sink):
        pipeline.add(elem)

    src.connect("pad-added", on_rtsp_pad_added, depay)

    depay.link(parse)
    parse.link(dec)
    dec.link(queue)
    queue.get_static_pad("src").link(mux.request_pad_simple("sink_0"))
    mux.link(infer)
    infer.link(convert)
    convert.link(caps)
    caps.link(clip_queue)
    clip_queue.link(sink)

    display = GstDisplay("assess")

    clip_queue.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER,
        make_probe(classifier, args, display),
        None,
    )

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        display.stop()


if __name__ == "__main__":
    main()
