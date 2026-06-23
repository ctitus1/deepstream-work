#!/usr/bin/env python3
"""Generate README data-flow diagrams.

The diagrams are defined as small fixed-grid layouts and emitted in three forms:

- ``*.drawio``: editable diagrams.net/draw.io source.
- ``*.svg``: README-friendly vector preview.
- ``*.png``: raster preview for quick local checks.

Both diagrams use the same dark-mode styling, row spacing, group spacing, node
height, and node widths so the parser app and ROS publisher can be compared
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import atan2, cos, sin
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent

SPACING = 40
BOX_W = 240
BOX_H = 135
GROUP_INSET = SPACING
GROUP_GAP = SPACING
BOX_GAP = SPACING

HEADER_X = SPACING
HEADER_TITLE_Y = SPACING
HEADER_TITLE_HEIGHT = 34
HEADER_SUBTITLE_Y = HEADER_TITLE_Y + HEADER_TITLE_HEIGHT
HEADER_SUBTITLE_HEIGHT = 24
HEADER_BOTTOM = HEADER_SUBTITLE_Y + HEADER_SUBTITLE_HEIGHT
HEADER_TITLE_SVG_Y = HEADER_TITLE_Y + 24
HEADER_SUBTITLE_SVG_Y = HEADER_SUBTITLE_Y + 13
HEADER_TITLE_PNG_Y = HEADER_TITLE_Y
HEADER_SUBTITLE_PNG_Y = HEADER_SUBTITLE_Y

ROW_TOP = HEADER_BOTTOM + SPACING + GROUP_INSET
ROW_MIDDLE = ROW_TOP + BOX_H + SPACING
ROW_BOTTOM = ROW_MIDDLE + BOX_H + SPACING
ROW_GAP = BOX_H + SPACING
assert ROW_BOTTOM - ROW_MIDDLE == ROW_GAP

X_VIDEO = SPACING + GROUP_INSET
X_DECODE = X_VIDEO + BOX_W + GROUP_INSET + GROUP_GAP + GROUP_INSET
X_STAMP = X_DECODE + BOX_W + BOX_GAP
X_PROCESS = X_STAMP + BOX_W + BOX_GAP
X_BRANCH = X_PROCESS + BOX_W + BOX_GAP
X_TCP = X_BRANCH + BOX_W + GROUP_INSET + GROUP_GAP + GROUP_INSET
X_BUILDER = X_TCP + BOX_W + BOX_GAP
X_TOPICS = X_BUILDER + BOX_W + BOX_GAP
X_CONSUMERS = X_TOPICS + BOX_W + BOX_GAP
BRANCH_JOIN_X = X_BRANCH + BOX_W + GROUP_INSET + GROUP_GAP // 2
CANVAS_W = X_CONSUMERS + BOX_W + GROUP_INSET + SPACING
CANVAS_H = ROW_BOTTOM + BOX_H + GROUP_INSET + SPACING

COLORS = {
    "background": "#0b1120",
    "group_fill": "#0f172a",
    "group_stroke": "#334155",
    "node_fill": "#111827",
    "node_stroke": "#38bdf8",
    "text": "#e5e7eb",
    "muted_text": "#a7b0be",
    "arrow": "#94a3b8",
}

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclass(frozen=True)
class Node:
    key: str
    title: str
    lines: tuple[str, ...]
    x: int
    y: int
    w: int = BOX_W
    h: int = BOX_H


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class Port:
    side: str
    ratio: float = 0.5


@dataclass(frozen=True)
class Edge:
    key: str
    source: str
    target: str
    source_port: Port = Port("right", 0.5)
    target_port: Port = Port("left", 0.5)
    points: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Diagram:
    basename: str
    title: str
    subtitle: str
    groups: tuple[Group, ...]
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    width: int = CANVAS_W
    height: int = CANVAS_H


def group_around(key: str, label: str, x: int, y: int, w: int, h: int) -> Group:
    return Group(key, label, x - GROUP_INSET, y - GROUP_INSET, w + GROUP_INSET * 2, h + GROUP_INSET * 2)


def nodes_by_key(diagram: Diagram) -> dict[str, Node]:
    return {node.key: node for node in diagram.nodes}


def port_point(node: Node, port: Port) -> tuple[float, float]:
    if port.side == "left":
        return node.x, node.y + node.h * port.ratio
    if port.side == "right":
        return node.x + node.w, node.y + node.h * port.ratio
    if port.side == "top":
        return node.x + node.w * port.ratio, node.y
    if port.side == "bottom":
        return node.x + node.w * port.ratio, node.y + node.h
    raise ValueError(f"unknown port side: {port.side}")


def edge_points(diagram: Diagram, edge: Edge) -> list[tuple[float, float]]:
    nodes = nodes_by_key(diagram)
    source = nodes[edge.source]
    target = nodes[edge.target]
    return [port_point(source, edge.source_port), *edge.points, port_point(target, edge.target_port)]


def validate_spacing(diagram: Diagram) -> None:
    group_left = min(group.x for group in diagram.groups)
    group_top = min(group.y for group in diagram.groups)
    group_right = max(group.x + group.w for group in diagram.groups)
    group_bottom = max(group.y + group.h for group in diagram.groups)
    checks = {
        "left edge": group_left,
        "header to groups": group_top - HEADER_BOTTOM,
        "right edge": diagram.width - group_right,
        "bottom edge": diagram.height - group_bottom,
    }
    for label, actual in checks.items():
        if actual != SPACING:
            raise ValueError(f"{diagram.basename} {label} spacing is {actual}, expected {SPACING}")


def drawio_value(title: str, lines: tuple[str, ...]) -> str:
    return f"<b>{escape(title)}</b><br>" + "<br>".join(escape(line) for line in lines)


def add_geometry(parent: ET.Element, x: int, y: int, w: int, h: int) -> None:
    ET.SubElement(
        parent,
        "mxGeometry",
        {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
    )


def add_background(root: ET.Element, diagram: Diagram) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": "background",
            "value": "",
            "style": f"rounded=0;whiteSpace=wrap;html=1;fillColor={COLORS['background']};strokeColor=none;",
            "vertex": "1",
            "parent": "1",
            "connectable": "0",
        },
    )
    add_geometry(cell, 0, 0, diagram.width, diagram.height)


def add_text(root: ET.Element, cell_id: str, value: str, x: int, y: int, w: int, h: int, size: int) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "value": value,
            "style": (
                "text;html=1;strokeColor=none;fillColor=none;align=left;"
                f"verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize={size};"
                f"fontColor={COLORS['text']};fontStyle=1;"
            ),
            "vertex": "1",
            "parent": "1",
        },
    )
    add_geometry(cell, x, y, w, h)


def add_group(root: ET.Element, group: Group) -> None:
    rect = ET.SubElement(
        root,
        "mxCell",
        {
            "id": group.key,
            "value": "",
            "style": (
                "rounded=1;whiteSpace=wrap;html=1;arcSize=4;absoluteArcSize=1;"
                f"fillColor={COLORS['group_fill']};strokeColor={COLORS['group_stroke']};dashed=1;"
            ),
            "vertex": "1",
            "parent": "1",
            "connectable": "0",
        },
    )
    add_geometry(rect, group.x, group.y, group.w, group.h)

    label = ET.SubElement(
        root,
        "mxCell",
        {
            "id": f"{group.key}_label",
            "value": escape(group.label),
            "style": (
                "text;html=1;strokeColor=none;fillColor=none;align=left;"
                f"verticalAlign=bottom;whiteSpace=wrap;rounded=0;fontSize=14;"
                f"fontColor={COLORS['muted_text']};fontStyle=1;"
            ),
            "vertex": "1",
            "parent": "1",
            "connectable": "0",
        },
    )
    add_geometry(label, group.x + 16, group.y + group.h - 32, group.w - 32, 24)


def add_node(root: ET.Element, node: Node) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": node.key,
            "value": drawio_value(node.title, node.lines),
            "style": (
                "rounded=1;whiteSpace=wrap;html=1;arcSize=6;absoluteArcSize=1;"
                f"fillColor={COLORS['node_fill']};strokeColor={COLORS['node_stroke']};"
                f"fontColor={COLORS['text']};spacing=10;spacingTop=4;spacingBottom=4;shadow=0;"
            ),
            "vertex": "1",
            "parent": "1",
        },
    )
    add_geometry(cell, node.x, node.y, node.w, node.h)


def drawio_port_style(prefix: str, port: Port) -> str:
    if port.side == "left":
        x, y = 0, port.ratio
    elif port.side == "right":
        x, y = 1, port.ratio
    elif port.side == "top":
        x, y = port.ratio, 0
    elif port.side == "bottom":
        x, y = port.ratio, 1
    else:
        raise ValueError(f"unknown port side: {port.side}")
    return f"{prefix}X={x};{prefix}Y={y};{prefix}Dx=0;{prefix}Dy=0;"


def add_edge(root: ET.Element, edge: Edge) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": "edge_" + edge.key,
            "value": "",
            "style": (
                "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
                "jettySize=auto;html=1;endArrow=block;endFill=1;"
                f"strokeColor={COLORS['arrow']};fontColor={COLORS['muted_text']};"
                + drawio_port_style("exit", edge.source_port)
                + drawio_port_style("entry", edge.target_port)
            ),
            "edge": "1",
            "parent": "1",
            "source": edge.source,
            "target": edge.target,
        },
    )
    geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    if edge.points:
        points = ET.SubElement(geometry, "Array", {"as": "points"})
        for x, y in edge.points:
            ET.SubElement(points, "mxPoint", {"x": str(x), "y": str(y)})


def build_drawio(diagram: Diagram) -> str:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "agent": "Codex",
            "version": "24.7.17",
            "type": "device",
        },
    )
    page = ET.SubElement(mxfile, "diagram", {"id": diagram.basename, "name": diagram.title})
    model = ET.SubElement(
        page,
        "mxGraphModel",
        {
            "dx": "1422",
            "dy": "794",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": str(diagram.width),
            "pageHeight": str(diagram.height),
            "background": COLORS["background"],
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    add_background(root, diagram)
    add_text(root, "title", diagram.title, HEADER_X, HEADER_TITLE_Y, 700, HEADER_TITLE_HEIGHT, 24)
    add_text(root, "subtitle", diagram.subtitle, HEADER_X, HEADER_SUBTITLE_Y, 1180, HEADER_SUBTITLE_HEIGHT, 13)
    for group in diagram.groups:
        add_group(root, group)
    for node in diagram.nodes:
        add_node(root, node)
    for edge in diagram.edges:
        add_edge(root, edge)

    rough = ET.tostring(mxfile, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def svg_element(tag: str, attrs: dict[str, object], body: str = "") -> str:
    attr_text = " ".join(f'{key}="{escape(str(value))}"' for key, value in attrs.items())
    if body:
        return f"<{tag} {attr_text}>{body}</{tag}>"
    return f"<{tag} {attr_text}/>"


def svg_path(points: list[tuple[float, float]]) -> str:
    first, *rest = points
    return " ".join([f"M {first[0]:.1f} {first[1]:.1f}", *[f"L {x:.1f} {y:.1f}" for x, y in rest]])


def build_svg(diagram: Diagram) -> str:
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{diagram.width}" height="{diagram.height}" '
            f'viewBox="0 0 {diagram.width} {diagram.height}">'
        ),
        "<defs>",
        (
            '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" '
            'orient="auto" markerUnits="strokeWidth">'
        ),
        f'<path d="M2,2 L10,6 L2,10 z" fill="{COLORS["arrow"]}"/>',
        "</marker>",
        "</defs>",
        "<style>",
        ".title{font:700 24px Arial,sans-serif;fill:#e5e7eb}",
        ".subtitle{font:13px Arial,sans-serif;fill:#a7b0be}",
        ".group-label{font:700 14px Arial,sans-serif;fill:#a7b0be}",
        ".node-title{font:700 17px Arial,sans-serif;fill:#e5e7eb}",
        ".node-body{font:13px Arial,sans-serif;fill:#a7b0be}",
        "</style>",
        svg_element(
            "rect",
            {"x": 0, "y": 0, "width": diagram.width, "height": diagram.height, "fill": COLORS["background"]},
        ),
        svg_element("text", {"x": HEADER_X, "y": HEADER_TITLE_SVG_Y, "class": "title"}, diagram.title),
        svg_element("text", {"x": HEADER_X, "y": HEADER_SUBTITLE_SVG_Y, "class": "subtitle"}, diagram.subtitle),
    ]

    for group in diagram.groups:
        parts.append(
            svg_element(
                "rect",
                {
                    "x": group.x,
                    "y": group.y,
                    "width": group.w,
                    "height": group.h,
                    "rx": 10,
                    "fill": COLORS["group_fill"],
                    "stroke": COLORS["group_stroke"],
                    "stroke-width": 1.6,
                    "stroke-dasharray": "7 7",
                },
            )
        )
        parts.append(
            svg_element(
                "text",
                {"x": group.x + 16, "y": group.y + group.h - 14, "class": "group-label"},
                group.label,
            )
        )

    for node in diagram.nodes:
        parts.append(
            svg_element(
                "rect",
                {
                    "x": node.x,
                    "y": node.y,
                    "width": node.w,
                    "height": node.h,
                    "rx": 8,
                    "fill": COLORS["node_fill"],
                    "stroke": COLORS["node_stroke"],
                    "stroke-width": 1.8,
                },
            )
        )
        parts.append(svg_element("text", {"x": node.x + 15, "y": node.y + 30, "class": "node-title"}, node.title))
        y = node.y + 62
        for line in node.lines:
            parts.append(svg_element("text", {"x": node.x + 15, "y": y, "class": "node-body"}, line))
            y += 23

    for edge in diagram.edges:
        parts.append(
            svg_element(
                "path",
                {
                    "d": svg_path(edge_points(diagram, edge)),
                    "fill": "none",
                    "stroke": COLORS["arrow"],
                    "stroke-width": 2.4,
                    "stroke-linejoin": "round",
                    "stroke-linecap": "round",
                    "marker-end": "url(#arrow)",
                },
            )
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def draw_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill: str, width: int = 3) -> None:
    draw.line(points, fill=fill, width=width, joint="curve")
    (x1, y1), (x2, y2) = points[-2], points[-1]
    angle = atan2(y2 - y1, x2 - x1)
    size = 13
    wing = 0.55
    arrow = [
        (x2, y2),
        (x2 - size * cos(angle - wing), y2 - size * sin(angle - wing)),
        (x2 - size * cos(angle + wing), y2 - size * sin(angle + wing)),
    ]
    draw.polygon(arrow, fill=fill)


def render_png(diagram: Diagram, output_path: Path) -> None:
    title_font = load_font(FONT_BOLD, 24)
    subtitle_font = load_font(FONT_REGULAR, 13)
    group_font = load_font(FONT_BOLD, 14)
    node_title_font = load_font(FONT_BOLD, 17)
    node_body_font = load_font(FONT_REGULAR, 13)

    image = Image.new("RGBA", (diagram.width, diagram.height), COLORS["background"])
    draw = ImageDraw.Draw(image)

    draw.text((HEADER_X, HEADER_TITLE_PNG_Y), diagram.title, font=title_font, fill=COLORS["text"])
    draw.text((HEADER_X, HEADER_SUBTITLE_PNG_Y), diagram.subtitle, font=subtitle_font, fill=COLORS["muted_text"])

    for group in diagram.groups:
        draw.rounded_rectangle(
            (group.x, group.y, group.x + group.w, group.y + group.h),
            radius=10,
            fill=COLORS["group_fill"],
            outline=COLORS["group_stroke"],
            width=2,
        )
        draw.text((group.x + 16, group.y + group.h - 28), group.label, font=group_font, fill=COLORS["muted_text"])

    for node in diagram.nodes:
        draw.rounded_rectangle(
            (node.x, node.y, node.x + node.w, node.y + node.h),
            radius=8,
            fill=COLORS["node_fill"],
            outline=COLORS["node_stroke"],
            width=2,
        )
        draw.text((node.x + 15, node.y + 13), node.title, font=node_title_font, fill=COLORS["text"])
        y = node.y + 49
        for line in node.lines:
            draw.text((node.x + 15, y), line, font=node_body_font, fill=COLORS["muted_text"])
            y += 23

    for edge in diagram.edges:
        draw_arrow(draw, edge_points(diagram, edge), COLORS["arrow"])

    image.save(output_path)


def ros_publisher_diagram() -> Diagram:
    groups = (
        group_around("stream_group", "Video starter: scripts/start_rtsp_stream.sh", X_VIDEO, ROW_MIDDLE, BOX_W, BOX_H),
        group_around(
            "source_group",
            "DeepStream sender: src/ros_source.py + pipeline.py",
            X_DECODE,
            ROW_TOP,
            X_BRANCH + BOX_W - X_DECODE,
            ROW_BOTTOM + BOX_H - ROW_TOP,
        ),
        group_around(
            "bridge_group",
            "ROS publisher: src/ros_bridge.py",
            X_TCP,
            ROW_MIDDLE,
            X_CONSUMERS + BOX_W - X_TCP,
            BOX_H,
        ),
    )
    nodes = (
        Node("video", "Video input", ("camera stream", "or saved video"), X_VIDEO, ROW_MIDDLE),
        Node("decode", "Read frames", ("open the video", "one frame at a time"), X_DECODE, ROW_MIDDLE),
        Node("stamp", "Frame time", ("save the source time", "keep it with frame"), X_STAMP, ROW_MIDDLE),
        Node("raw", "Plain image", ("resize to 640x368", "send as JPEG"), X_BRANCH, ROW_TOP),
        Node("pgie", "Find people", ("draw person boxes", "add score + id"), X_PROCESS, ROW_MIDDLE),
        Node("detect", "Detection image", ("resize with boxes", "send box data"), X_BRANCH, ROW_MIDDLE),
        Node("sgie", "Check injuries", ("look at each person", "make injury scores"), X_PROCESS, ROW_BOTTOM),
        Node("assess", "Assessment image", ("one image/person", "send injury labels"), X_BRANCH, ROW_BOTTOM),
        Node("tcp", "Send to bridge", ("three local ports", "raw | boxes | injury"), X_TCP, ROW_MIDDLE),
        Node("builder", "Make ROS messages", ("one publisher each", "reuse frame time"), X_BUILDER, ROW_MIDDLE),
        Node(
            "topics",
            "Published topics",
            ("/uas4/image", "/uas4/target_detections", "/casualty_image/..."),
            X_TOPICS,
            ROW_MIDDLE,
        ),
        Node("consumers", "View or record", ("Foxglove", "rosbag -s mcap"), X_CONSUMERS, ROW_MIDDLE),
    )
    edges = (
        Edge("video_decode", "video", "decode"),
        Edge("decode_stamp", "decode", "stamp"),
        Edge(
            "stamp_raw",
            "stamp",
            "raw",
            Port("top", 0.5),
            Port("left", 0.5),
            ((X_STAMP + BOX_W // 2, ROW_TOP + BOX_H // 2),),
        ),
        Edge("stamp_pgie", "stamp", "pgie", Port("right", 0.5), Port("left", 0.5)),
        Edge("pgie_detect", "pgie", "detect"),
        Edge("pgie_sgie", "pgie", "sgie", Port("bottom", 0.5), Port("top", 0.5)),
        Edge("sgie_assess", "sgie", "assess"),
        Edge(
            "raw_tcp",
            "raw",
            "tcp",
            Port("right", 0.5),
            Port("left", 0.25),
            ((BRANCH_JOIN_X, ROW_TOP + BOX_H // 2), (BRANCH_JOIN_X, ROW_MIDDLE + BOX_H // 4)),
        ),
        Edge("detect_tcp", "detect", "tcp"),
        Edge(
            "assess_tcp",
            "assess",
            "tcp",
            Port("right", 0.5),
            Port("left", 0.75),
            ((BRANCH_JOIN_X, ROW_BOTTOM + BOX_H // 2), (BRANCH_JOIN_X, ROW_MIDDLE + BOX_H * 3 // 4)),
        ),
        Edge("tcp_builder", "tcp", "builder"),
        Edge("builder_topics", "builder", "topics"),
        Edge("topics_consumers", "topics", "consumers"),
    )
    return Diagram(
        "data_flow",
        "ROS Publisher Flow",
        "DeepStream turns video into images, person boxes, and injury checks; ROS publishes them for tools to view or record.",
        groups,
        nodes,
        edges,
    )


def parser_app_diagram() -> Diagram:
    row_setup = ROW_TOP
    row_runtime = row_setup + BOX_H + GROUP_INSET + GROUP_GAP + GROUP_INSET
    row_assess = row_runtime + BOX_H + BOX_GAP
    group_gap_y = row_setup + BOX_H + GROUP_INSET + GROUP_GAP // 2

    x_setup = X_DECODE
    x_call = X_STAMP
    x_attach = X_PROCESS
    x_loop = X_BRANCH
    x_assess = X_PROCESS
    x_bbox = X_PROCESS
    x_osd = X_BRANCH
    x_sink = X_TCP
    x_status = X_BRANCH

    groups = (
        group_around("stream_group", "Video starter: scripts/start_rtsp_stream.sh", X_VIDEO, row_runtime, BOX_W, BOX_H),
        group_around(
            "parser_group",
            "Parser app: src/parser_app.py",
            x_setup,
            row_setup,
            x_loop + BOX_W - x_setup,
            BOX_H,
        ),
        group_around(
            "pipeline_group",
            "Video pipeline: pipeline.py + helper probes",
            X_DECODE,
            row_runtime,
            x_sink + BOX_W - X_DECODE,
            row_assess + BOX_H - row_runtime,
        ),
    )
    nodes = (
        Node("setup", "Startup settings", ("choose models", "make configs"), x_setup, row_setup),
        Node("call", "Create pipeline", ("build_pipeline()", "video processing graph"), x_call, row_setup),
        Node("attach", "Add frame hooks", ("boxes + injuries", "timing + playback"), x_attach, row_setup),
        Node("loop", "Run the app", ("keyboard controls", "stop on errors"), x_loop, row_setup),
        Node("video", "Video input", ("camera stream", "or saved video"), X_VIDEO, row_runtime),
        Node("decode", "Read frames", ("open the video", "one frame at a time"), X_DECODE, row_runtime),
        Node("pgie", "Find people", ("draw person boxes", "add score + id"), X_STAMP, row_runtime),
        Node("bbox", "Box overlay", ("show person boxes", "and confidence"), x_bbox, row_runtime),
        Node("osd", "Draw overlays", ("combine boxes/text", "prepare display"), x_osd, row_runtime),
        Node("sink", "Show window", ("display output", "or no-window mode"), x_sink, row_runtime),
        Node("sgie", "Check injuries", ("optional model", "one person at a time"), X_STAMP, row_assess),
        Node("assess", "Injury labels", ("fresh labels", "console logs"), x_assess, row_assess),
        Node("status", "Frame choice", ("show all frames", "or fresh checks only"), x_status, row_assess),
    )
    edges = (
        Edge("setup_call", "setup", "call"),
        Edge("call_attach", "call", "attach"),
        Edge("attach_loop", "attach", "loop"),
        Edge(
            "call_decode",
            "call",
            "decode",
            Port("bottom", 0.5),
            Port("top", 0.5),
            ((x_call + BOX_W // 2, group_gap_y), (X_DECODE + BOX_W // 2, group_gap_y)),
        ),
        Edge(
            "attach_bbox",
            "attach",
            "bbox",
            Port("bottom", 0.5),
            Port("top", 0.5),
            ((x_attach + BOX_W // 2, group_gap_y), (x_bbox + BOX_W // 2, group_gap_y)),
        ),
        Edge(
            "attach_assess",
            "attach",
            "assess",
            Port("bottom", 0.75),
            Port("top", 0.75),
        ),
        Edge(
            "loop_sink",
            "loop",
            "sink",
            Port("bottom", 0.5),
            Port("top", 0.5),
            ((x_loop + BOX_W // 2, group_gap_y), (x_sink + BOX_W // 2, group_gap_y)),
        ),
        Edge("video_decode", "video", "decode"),
        Edge("decode_pgie", "decode", "pgie"),
        Edge("pgie_bbox", "pgie", "bbox"),
        Edge("bbox_osd", "bbox", "osd", Port("right", 0.25), Port("left", 0.25)),
        Edge("osd_sink", "osd", "sink"),
        Edge("pgie_sgie", "pgie", "sgie", Port("bottom", 0.5), Port("top", 0.5)),
        Edge("sgie_assess", "sgie", "assess"),
        Edge("assess_status", "assess", "status"),
        Edge(
            "status_osd",
            "status",
            "osd",
            Port("top", 0.5),
            Port("bottom", 0.5),
        ),
    )
    return Diagram(
        "parser_flow",
        "Parser App Flow",
        "The parser app opens video, finds people, optionally checks injuries, draws labels, and shows/logs the result.",
        groups,
        nodes,
        edges,
        width=x_sink + BOX_W + GROUP_INSET + SPACING,
        height=row_assess + BOX_H + GROUP_INSET + SPACING,
    )


def write_diagram(diagram: Diagram) -> None:
    validate_spacing(diagram)
    drawio_path = OUT_DIR / f"{diagram.basename}.drawio"
    svg_path = OUT_DIR / f"{diagram.basename}.svg"
    png_path = OUT_DIR / f"{diagram.basename}.png"
    drawio_path.write_text(build_drawio(diagram), encoding="utf-8")
    svg_path.write_text(build_svg(diagram), encoding="utf-8")
    render_png(diagram, png_path)
    print(f"Wrote {drawio_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {png_path}")


def main() -> int:
    for diagram in (ros_publisher_diagram(), parser_app_diagram()):
        write_diagram(diagram)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
