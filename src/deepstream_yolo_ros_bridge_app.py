#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from cdcl_umd_msgs.msg import (
    AerialDetectionSource,
    Annotation,
    CasualtyImageCompressed,
    TargetBox,
    TargetBoxArray,
)
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import BoundingBox2D

from deepstream_yolo.frame_wire import recv_frame

DEFAULT_DETECT_ENDPOINT = "0.0.0.0:5610"
DEFAULT_ASSESS_ENDPOINT = "0.0.0.0:5611"
DEFAULT_IMAGE_ENDPOINT = "0.0.0.0:5609"
INT32_MAX = 2_147_483_647


class FramePublisherNode(Node):
    def __init__(
        self,
        name: str,
        topic: str,
        endpoint: str,
        log_interval: float,
        message_kind: str,
        frame_id: str,
        system_id: int,
        platform_name: str,
        sensor_frame_id: str,
    ):
        super().__init__(name)
        self.topic = topic
        self.host, self.port = parse_endpoint(endpoint)
        self.log_interval = log_interval
        self.last_log_time = 0.0
        self.message_kind = message_kind
        self.frame_id = frame_id
        self.system_id = system_id
        self.platform_name = platform_name
        self.sensor_frame_id = sensor_frame_id
        self.seq = 0
        message_type = {
            "image": CompressedImage,
            "detect": TargetBoxArray,
            "assess": CasualtyImageCompressed,
        }[message_kind]
        self.publisher = self.create_publisher(message_type, topic, 10)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            socket.create_connection((self.host, self.port), timeout=0.2).close()
        except OSError:
            pass
        self.thread.join(timeout=2.0)

    def serve(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            server.settimeout(0.5)
            self.get_logger().debug(f"listening endpoint={self.host}:{self.port} topic={self.topic}")

            while not self.stop_event.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue

                if self.stop_event.is_set():
                    conn.close()
                    break

                self.get_logger().debug(f"connected peer={addr[0]}:{addr[1]}")
                with conn:
                    conn.settimeout(0.5)
                    while not self.stop_event.is_set():
                        try:
                            metadata, payload = recv_frame(conn)
                        except socket.timeout:
                            continue
                        except EOFError:
                            break
                        except OSError as exc:
                            self.get_logger().warning(f"socket receive failed: {exc}")
                            break
                        self.publish_frame(metadata, payload)

    def publish_frame(self, metadata: dict, payload: bytes) -> None:
        if self.message_kind == "image":
            messages = [self.compressed_image(metadata, payload)]
        elif self.message_kind == "detect":
            messages = self.target_box_array_messages(metadata, payload)
        else:
            messages = self.casualty_image_messages(metadata, payload)
        for msg in messages:
            self.publisher.publish(msg)

        if self.should_log():
            log_text = metadata.get("log_text", "metadata=missing")
            self.get_logger().debug(
                f"published topic={self.topic} messages={len(messages)} bytes={len(payload)}\n{log_text}"
            )

    def compressed_image(self, metadata: dict, payload: bytes) -> CompressedImage:
        msg = CompressedImage()
        msg.header.stamp = self.stamp(metadata)
        msg.header.frame_id = self.frame_id
        msg.format = str(metadata.get("format", "jpeg"))
        msg.data = payload
        return msg

    def target_box_array_messages(self, metadata: dict, payload: bytes) -> list[TargetBoxArray]:
        source_img = self.compressed_image(metadata, payload)
        msg = TargetBoxArray()
        msg.seq = self.seq
        self.seq += 1
        msg.header = source_img.header
        msg.system_id = self.system_id
        msg.source_img = source_img
        msg.gimbal_attitude_quaternion.w = 1.0
        msg.uav_target_boxes = [
            self.target_box(metadata, obj)
            for obj in metadata.get("objects", [])
        ]
        msg.use_for_mosaic = False
        msg.detection_source = AerialDetectionSource.DETECTION_YOLO
        return [msg]

    def casualty_image_messages(
        self,
        metadata: dict,
        payload: bytes,
    ) -> list[CasualtyImageCompressed]:
        messages = []
        for obj in metadata.get("objects", []):
            source_img = self.compressed_image(metadata, payload)
            bbox = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
            msg = CasualtyImageCompressed()
            msg.data_source_id = data_source_id(metadata)
            msg.stamp = source_img.header.stamp
            msg.image = source_img
            msg.position.header = source_img.header
            msg.annotations = self.annotations(obj.get("predictions", {}))
            msg.bbox_x = float(bbox[0])
            msg.bbox_y = float(bbox[1])
            msg.bbox_width = float(bbox[2])
            msg.bbox_height = float(bbox[3])
            msg.sensor_frame_id = self.sensor_frame_id
            msg.platform_name = self.platform_name
            msg.is_sensor_frame_moving = False
            messages.append(msg)
        return messages

    def target_box(self, metadata: dict, obj: dict) -> TargetBox:
        bbox = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
        target_bbox = BoundingBox2D()
        target_bbox.size_x = float(bbox[2])
        target_bbox.size_y = float(bbox[3])
        target_bbox.center.position.x = float(bbox[0]) + float(bbox[2]) / 2.0
        target_bbox.center.position.y = float(bbox[1]) + float(bbox[3]) / 2.0

        target_box = TargetBox()
        target_box.data_source_id = data_source_id(metadata)
        target_box.target_bbox = target_bbox
        target_box.use_for_assessment = True
        target_box.detection_source.detection_source = AerialDetectionSource.DETECTION_YOLO
        target_box.detection_class = str(obj.get("class_name", "person"))
        target_box.detection_confidence = float(obj.get("confidence", 0.0))
        return target_box

    def annotations(self, predictions: dict) -> list[Annotation]:
        annotations = []
        for name, prediction in sorted(predictions.items()):
            probabilities = prediction.get("probabilities", [])
            if not probabilities:
                probabilities = [
                    float(prediction.get("class_id", -1)),
                    float(prediction.get("confidence", 0.0)),
                ]
            annotation = Annotation()
            annotation.field_name = f"clip_rgb_{name}"
            annotation.observation = [float(value) for value in probabilities]
            annotations.append(annotation)
        return annotations

    def stamp(self, metadata: dict):
        timestamp_ns = metadata.get("timestamp_ns")
        source = metadata.get("timestamp_source")
        if isinstance(timestamp_ns, int) and source in {"ntp", "ref"}:
            msg = self.get_clock().now().to_msg()
            msg.sec = int(timestamp_ns // 1_000_000_000)
            msg.nanosec = int(timestamp_ns % 1_000_000_000)
            return msg
        return self.get_clock().now().to_msg()

    def should_log(self) -> bool:
        if self.log_interval < 0:
            return False
        if self.log_interval == 0:
            return True

        now = time.perf_counter()
        if now - self.last_log_time < self.log_interval:
            return False
        self.last_log_time = now
        return True


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    host, _, port = endpoint.rpartition(":")
    if not host or not port:
        raise ValueError(f"Expected endpoint HOST:PORT, got {endpoint!r}")
    return host, int(port)


def data_source_id(metadata: dict) -> int:
    explicit_id = metadata.get("data_source_id")
    if explicit_id is not None:
        return int_value(explicit_id, 0)

    frame_num = int_value(metadata.get("frame"), 0)
    return frame_num % INT32_MAX


def int_value(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-endpoint", default=DEFAULT_IMAGE_ENDPOINT)
    parser.add_argument("--detect-endpoint", default=DEFAULT_DETECT_ENDPOINT)
    parser.add_argument("--assess-endpoint", default=DEFAULT_ASSESS_ENDPOINT)
    parser.add_argument("--image-topic", default="/uas4/image")
    parser.add_argument("--detect-topic", default="/uas4/target_detections")
    parser.add_argument("--assess-topic", default="/casualty_image/compressed/annotated")
    parser.add_argument("--frame-id", default="deepstream_camera")
    parser.add_argument("--system-id", type=int, default=0)
    parser.add_argument("--platform-name", default="deepstream")
    parser.add_argument("--sensor-frame-id", default="deepstream_camera")
    parser.add_argument(
        "--metadata-log-interval",
        type=float,
        default=0.0,
        help="Seconds between ROS metadata logs; 0 logs every published frame, negative disables.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init(args=None)

    image_node = FramePublisherNode(
        "deepstream_image_publisher",
        args.image_topic,
        args.image_endpoint,
        args.metadata_log_interval,
        "image",
        args.frame_id,
        args.system_id,
        args.platform_name,
        args.sensor_frame_id,
    )
    detect_node = FramePublisherNode(
        "deepstream_detect_publisher",
        args.detect_topic,
        args.detect_endpoint,
        args.metadata_log_interval,
        "detect",
        args.frame_id,
        args.system_id,
        args.platform_name,
        args.sensor_frame_id,
    )
    assess_node = FramePublisherNode(
        "deepstream_assess_publisher",
        args.assess_topic,
        args.assess_endpoint,
        args.metadata_log_interval,
        "assess",
        args.frame_id,
        args.system_id,
        args.platform_name,
        args.sensor_frame_id,
    )
    executor = MultiThreadedExecutor()
    executor.add_node(image_node)
    executor.add_node(detect_node)
    executor.add_node(assess_node)

    image_node.start()
    detect_node.start()
    assess_node.start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        image_node.stop()
        detect_node.stop()
        assess_node.stop()
        executor.remove_node(image_node)
        executor.remove_node(detect_node)
        executor.remove_node(assess_node)
        image_node.destroy_node()
        detect_node.destroy_node()
        assess_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
