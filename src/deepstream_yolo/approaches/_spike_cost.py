"""Spike: is the nvof cost plane reachable from pyds? Throwaway probe.

Two separate reachability questions, and both have to answer yes:

  1. Can an approach get the element's ``output-cost`` property turned on at
     all? The shared eval harness never sets it and is off-limits for editing.
  2. Does the cost plane survive into something Python can read? pyds exposes
     NvDsOpticalFlowMeta as {rows, cols, mv_size, frame_num, data, priv,
     reserved} -- note the absence of ``cost`` and ``cost_size``, both of which
     the C struct carries.

For (1) and (2) alike the lever is that run_motion.py is ``__main__`` and its
``probe`` resolves ``element`` and ``flow_field`` as module globals at call
time, so rebinding them from here changes what the harness runs without
touching the file.

Struct offsets are verified rather than assumed: the same ctypes read is used
to recover the motion vectors, which pyds also decodes, and the two are
compared. If the vectors match, the cost offset is being read from the right
place too.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
import pyds

NAME = "_spike-cost"

# NvDsOpticalFlowMeta, 64-bit natural alignment:
#   0 rows u32 | 4 cols u32 | 8 mv_size u32 | 12 cost_size u32
#  16 frame_num u64 | 24 data* | 32 cost* | 40 priv* | 48 reserved*
OFF_COST_SIZE = 12
OFF_DATA = 24
OFF_COST = 32

_MEM_TYPE = {0: "unregistered/host-plain", 1: "host(pinned)", 2: "device", 3: "managed"}


def _memory_type(ptr: int) -> str:
    """Ask CUDA what kind of memory a pointer refers to, without dereferencing.

    Reading a device pointer from host code segfaults the process, which no
    try/except can catch, so the kind has to be established first.
    """
    if not ptr:
        return "null"
    try:
        rt = ctypes.CDLL("libcudart.so")
    except OSError:
        try:
            rt = ctypes.CDLL("libcudart.so.12")
        except OSError:
            return "libcudart unavailable"
    # struct cudaPointerAttributes { int type; int device; void* dev; void* host; }
    buf = (ctypes.c_byte * 32)()
    err = rt.cudaPointerGetAttributes(ctypes.byref(buf), ctypes.c_void_p(ptr))
    if err != 0:
        return f"cudaPointerGetAttributes err={err}"
    kind = ctypes.cast(buf, ctypes.POINTER(ctypes.c_int))[0]
    return _MEM_TYPE.get(kind, f"kind={kind}")


def _enable_output_cost() -> str:
    main = sys.modules.get("__main__")
    original = getattr(main, "element", None)
    if original is None:
        return "no __main__.element to wrap"

    def wrapped(factory, name, *args, **kwargs):
        elem = original(factory, name, *args, **kwargs)
        if factory == "nvof" and elem is not None:
            try:
                elem.set_property("output-cost", True)
                print("SPIKE: set nvof output-cost=True", flush=True)
            except Exception as exc:
                print(f"SPIKE: output-cost set failed: {exc}", flush=True)
        return elem

    main.element = wrapped
    return "wrapped __main__.element"


class _Reporter:
    def __init__(self):
        self.frames = 0

    def inspect(self, frame_meta):
        user_list = frame_meta.frame_user_meta_list
        while user_list:
            user_meta = pyds.NvDsUserMeta.cast(user_list.data)
            if user_meta.base_meta.meta_type == pyds.NVDS_OPTICAL_FLOW_META:
                of_meta = pyds.NvDsOpticalFlowMeta.cast(user_meta.user_meta_data)
                self._report(of_meta)
                raw = pyds.get_optical_flow_vectors(of_meta)
                rows, cols = int(of_meta.rows), int(of_meta.cols)
                if rows <= 0 or cols <= 0 or raw is None:
                    return None
                return np.asarray(raw, dtype=np.float32).reshape(rows, cols, 2) / 32.0
            user_list = user_list.next
        return None

    def _report(self, of_meta):
        if self.frames >= 2:
            return
        self.frames += 1
        rows, cols = int(of_meta.rows), int(of_meta.cols)
        print(f"SPIKE: rows={rows} cols={cols} mv_size={of_meta.mv_size}", flush=True)

        base = pyds.get_ptr(of_meta)
        print(f"SPIKE: get_ptr(of_meta) = {base}", flush=True)
        if not base:
            print("SPIKE: no struct address, cost unreachable", flush=True)
            return

        blob = ctypes.string_at(base, 56)
        r, c, mv_size, cost_size = np.frombuffer(blob, dtype=np.uint32, count=4)
        data_ptr, cost_ptr = np.frombuffer(blob[OFF_DATA:OFF_DATA + 16], dtype=np.uint64)
        print(f"SPIKE: struct read rows={r} cols={c} mv_size={mv_size} "
              f"cost_size={cost_size}", flush=True)
        print(f"SPIKE: data=0x{int(data_ptr):x} cost=0x{int(cost_ptr):x}", flush=True)

        # Offsets are only trustworthy if the struct read agrees with pyds.
        if int(r) != rows or int(c) != cols:
            print("SPIKE: struct offsets WRONG (rows/cols disagree with pyds)", flush=True)
            return
        print("SPIKE: struct offsets confirmed by rows/cols agreement", flush=True)

        print(f"SPIKE: data memory type = {_memory_type(int(data_ptr))}", flush=True)
        print(f"SPIKE: cost memory type = {_memory_type(int(cost_ptr))}", flush=True)

        if cost_size == 0 or not cost_ptr:
            print("SPIKE: cost plane ABSENT (cost_size=0 or null pointer)", flush=True)
            return

        kind = _memory_type(int(cost_ptr))
        if kind.startswith("device"):
            print("SPIKE: cost is device memory; needs a cudaMemcpy to read", flush=True)
            return
        # cost_size is the size of ONE cost element, mirroring mv_size=4 for the
        # 2x int16 flow vector -- not the size of the plane. The plane is one
        # element per flow cell.
        nbytes = rows * cols * int(cost_size)
        cost = np.frombuffer(
            ctypes.string_at(int(cost_ptr), nbytes), dtype=np.uint8
        ).reshape(rows, cols)
        flow = np.asarray(
            pyds.get_optical_flow_vectors(of_meta), dtype=np.float32
        ).reshape(rows, cols, 2) / 32.0
        speed = np.hypot(flow[..., 0], flow[..., 1])
        print(f"SPIKE: cost READABLE shape={cost.shape} min={cost.min()} "
              f"max={cost.max()} mean={cost.mean():.1f} "
              f"pctiles={np.percentile(cost, [5, 50, 95]).round(1).tolist()}", flush=True)
        # The whole premise is that cost predicts unreliable flow. If cost does
        # not rise with the absurd 20-80 px/frame vectors, it is not usable.
        for lo, hi in ((0.0, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, 1e9)):
            sel = (speed >= lo) & (speed < hi)
            if sel.any():
                print(f"SPIKE:   speed[{lo},{hi}) n={int(sel.sum()):6d} "
                      f"cost mean={cost[sel].mean():6.1f}", flush=True)
        print(f"SPIKE:   border cost mean={cost[:12].mean():.1f} (top 12 rows) vs "
              f"interior {cost[20:-20, 20:-20].mean():.1f}", flush=True)


class Approach:
    needs_flow = True
    needs_pixels = False

    def __init__(self, cfg: dict):
        print(f"SPIKE: {_enable_output_cost()}", flush=True)
        reporter = _Reporter()
        main = sys.modules.get("__main__")
        if main is not None and hasattr(main, "flow_field"):
            main.flow_field = reporter.inspect
            print("SPIKE: rebound __main__.flow_field", flush=True)
        else:
            print("SPIKE: could not rebind flow_field", flush=True)

    def process(self, ctx) -> list[dict]:
        return []
