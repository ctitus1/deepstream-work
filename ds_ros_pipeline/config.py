"""All runtime parameters for ds_ros_pipeline (DESIGN.md Sec 4 "ROS parameters").

One frozen dataclass mirrors the parameter table exactly; ``declare_parameters``
binds it to a live rclpy node so every value is overridable via
``--ros-args -p``. Unit tests construct ``PipelineConfig()`` directly and never
touch rclpy (the import is deferred into the one function that needs it).
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SOURCE_URI = "file://streams/lorton-d4-rgb-nano.mp4"


@dataclass(frozen=True)
class PipelineConfig:
    """Every tunable in DESIGN.md Sec 4, defaults verbatim.

    Instances are plain values: no locks, safe to share read-only across all
    threads after construction. Nothing mutates a config after startup.
    """

    source_uri: str = DEFAULT_SOURCE_URI          # source.uri
    source_loop: bool = True                      # source.loop (AU-replay loop, Sec 3.1/5)
    source_max_preload_mb: int = 1024             # source.max_preload_mb guard (Sec 11 risk 6)
    batch_capacity: int = 16                      # batch.capacity (BatchItem deque cap)
    batch_engine_batch: int = 8                   # batch.engine_batch (b8 engine / mux batch-size)
    continuous_stride: int = 3                    # continuous.stride
    continuous_run_size: int = 4                  # continuous.run_size
    preview_width: int = 640                      # preview.width
    preview_height: int = 360                     # preview.height
    preview_quality: int = 75                     # preview.quality
    record_bitrate: int = 200_000_000             # record.bitrate (CBR, Sec 3.1 Branch R)
    record_output_dir: str = "outputs/ds_ros"     # record.output_dir
    record_stop_timeout: float = 5.0              # record.stop_timeout (async drain wait, Sec 7)
    snapshot_output_dir: str = "outputs/ds_ros"   # snapshot.output_dir
    detections_image_width: int = 640             # detections.image_width (source_img JPEG)
    detections_image_height: int = 368            # detections.image_height
    frame_id: str = "ds_camera"                   # frame_id for every published header


# ROS parameter name -> dataclass field name, one row per Sec 4 parameter.
PARAMETER_MAP: dict[str, str] = {
    "source.uri": "source_uri",
    "source.loop": "source_loop",
    "source.max_preload_mb": "source_max_preload_mb",
    "batch.capacity": "batch_capacity",
    "batch.engine_batch": "batch_engine_batch",
    "continuous.stride": "continuous_stride",
    "continuous.run_size": "continuous_run_size",
    "preview.width": "preview_width",
    "preview.height": "preview_height",
    "preview.quality": "preview_quality",
    "record.bitrate": "record_bitrate",
    "record.output_dir": "record_output_dir",
    "record.stop_timeout": "record_stop_timeout",
    "snapshot.output_dir": "snapshot_output_dir",
    "detections.image_width": "detections_image_width",
    "detections.image_height": "detections_image_height",
    "frame_id": "frame_id",
}


def declare_parameters(node) -> PipelineConfig:
    """Declare every Sec 4 parameter on ``node`` and return the resolved config.

    Called once from the main thread during node construction, before the
    executor spins. ``node`` is an ``rclpy.node.Node``; rclpy is imported by
    the caller, not here, so this module stays importable without ROS.
    Values already set via ``--ros-args -p name:=value`` override the
    dataclass defaults. Returns a fully populated PipelineConfig.

    ``node`` may be ``None`` (standalone mode, unit tests): the defaults are
    returned without touching any ROS machinery. Only duck-typed ``Node``
    methods are used (``has_parameter``/``declare_parameter``/
    ``get_parameter``), so no rclpy import happens here either way.
    """
    defaults = PipelineConfig()
    if node is None:
        return defaults
    values: dict[str, object] = {}
    for name, field in PARAMETER_MAP.items():
        default = getattr(defaults, field)
        if not node.has_parameter(name):
            node.declare_parameter(name, default)
        values[field] = _coerce(node.get_parameter(name).value, default)
    return PipelineConfig(**values)


def _coerce(value, default):
    """Cast a resolved parameter value to its field's type.

    rclpy already enforces the declared type, so this only normalizes the
    benign cases (int given for a float field, None for an unset value).
    """
    if value is None:
        return default
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return str(value)
