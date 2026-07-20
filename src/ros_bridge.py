#!/usr/bin/env python3
"""ROS Humble publisher bridge for DeepStream frame outputs.

``ros_source.py`` sends JPEG payloads plus JSON metadata over three local TCP
endpoints. This bridge creates one ``FramePublisherNode`` per endpoint/topic:
raw frames become ``CompressedImage``, detections become ``TargetBoxArray``, and
assessments become one or more ``CasualtyImageCompressed`` messages. Each frame
packet is converted to one ROS timestamp, then that exact stamp is copied to
every timestamp-bearing field in the emitted message.
"""

from __future__ import annotations

import argparse
import socket
import threading

import rclpy
from builtin_interfaces.msg import Time as RosTime
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

from deepstream_yolo.frame_wire import is_wall_clock_timestamp, parse_endpoint, recv_frame

DEFAULT_DETECT_ENDPOINT = "0.0.0.0:5610"
DEFAULT_ASSESS_ENDPOINT = "0.0.0.0:5611"
DEFAULT_IMAGE_ENDPOINT = "0.0.0.0:5609"

# name, topic argument, endpoint argument, message kind
PUBLISHERS = (
    ("deepstream_image_publisher", "image_topic", "image_endpoint", "image"),
    ("deepstream_detect_publisher", "detect_topic", "detect_endpoint", "detect"),
    ("deepstream_assess_publisher", "assess_topic", "assess_endpoint", "assess"),
)


class FramePublisherNode(Node):
    def __init__(
        self,
        name: str,
        topic: str,
        endpoint: str,
        message_kind: str,
        args: argparse.Namespace,
    ):
        super().__init__(name)
        self.topic = topic
        self.host, self.port = parse_endpoint(endpoint)
        self.message_kind = message_kind
        self.frame_id = args.frame_id
        self.system_id = args.system_id
        self.platform_name = args.platform_name
        self.sensor_frame_id = args.sensor_frame_id
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
        # One listener thread per stage keeps image, detect, and assess sockets independent.
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
        # Compute the stamp once per packet so embedded images and parent messages agree.
        stamp = self.stamp(metadata)
        if self.message_kind == "image":
            messages = [self.compressed_image(metadata, payload, stamp)]
        elif self.message_kind == "detect":
            messages = [self.target_box_array(metadata, payload, stamp)]
        else:
            messages = self.casualty_image_messages(metadata, payload, stamp)
        for msg in messages:
            self.publisher.publish(msg)

        self.get_logger().debug(
            f"published topic={self.topic} messages={len(messages)} "
            f"bytes={len(payload)}\n{metadata.get('log_text', 'metadata=missing')}"
        )

    def compressed_image(self, metadata: dict, payload: bytes, stamp) -> CompressedImage:
        msg = CompressedImage()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = self.frame_id
        msg.format = str(metadata.get("format", "jpeg"))
        msg.data = payload
        return msg

    def target_box_array(self, metadata: dict, payload: bytes, stamp) -> TargetBoxArray:
        # TargetBoxArray carries the detection image plus one bbox entry per detected person.
        source_img = self.compressed_image(metadata, payload, stamp)
        msg = TargetBoxArray()
        msg.seq = self.seq
        self.seq += 1
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = self.frame_id
        msg.system_id = self.system_id
        msg.source_img = source_img
        msg.gimbal_attitude_quaternion.w = 1.0
        msg.uav_target_boxes = [
            self.target_box(metadata, obj)
            for obj in metadata.get("objects", [])
        ]
        msg.use_for_mosaic = False
        msg.detection_source = AerialDetectionSource.DETECTION_YOLO
        return msg

    def casualty_image_messages(
        self,
        metadata: dict,
        payload: bytes,
        stamp,
    ) -> list[CasualtyImageCompressed]:
        messages = []
        for obj in metadata.get("objects", []):
            # Publish one casualty image per assessed object from the source frame.
            source_img = self.compressed_image(metadata, payload, stamp)
            bbox = obj.get("bbox", [0.0, 0.0, 0.0, 0.0])
            msg = CasualtyImageCompressed()
            msg.data_source_id = data_source_id(metadata)
            copy_stamp(msg.stamp, stamp)
            msg.image = source_img
            copy_stamp(msg.position.header.stamp, stamp)
            msg.position.header.frame_id = self.frame_id
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
            annotation = Annotation()
            annotation.field_name = f"clip_rgb_{name}"
            annotation.observation = [
                float(value) for value in prediction["probabilities"]
            ]
            annotations.append(annotation)
        return annotations

    def stamp(self, metadata: dict):
        # Source metadata is preferred, but only when it is actually wall clock.
        # A stream-relative PTS written into a ROS header dates the message to
        # 1970, which breaks TF lookups, message_filters sync and bag playback.
        # RTSP hits this routinely: rtspsrc has ntp-sync, but ntp_timestamp is
        # invalid until the first RTCP sender report, so the opening seconds of
        # every run would otherwise publish 1970 and then jump ~56 years.
        timestamp_ns = metadata.get("source_timestamp_ns")
        timestamp_source = metadata.get("source_timestamp_source")
        if timestamp_ns is None or not is_wall_clock_timestamp(timestamp_source):
            self.get_logger().debug(
                "metadata timestamp unusable "
                f"(source={timestamp_source or 'missing'}); using node clock "
                f"data_source_id={data_source_id(metadata)}"
            )
            return self.get_clock().now().to_msg()
        msg = RosTime()
        msg.sec = int(timestamp_ns // 1_000_000_000)
        msg.nanosec = int(timestamp_ns % 1_000_000_000)
        return msg


def data_source_id(metadata: dict) -> int:
    return int(metadata["data_source_id"])


def copy_stamp(target, source) -> None:
    target.sec = int(source.sec)
    target.nanosec = int(source.nanosec)


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init(args=None)

    # Each node owns one endpoint/topic pair; a shared executor handles ROS
    # callbacks. Start order does not matter: each serve thread binds its own port.
    nodes = [
        FramePublisherNode(
            name,
            getattr(args, topic_arg),
            getattr(args, endpoint_arg),
            message_kind,
            args,
        )
        for name, topic_arg, endpoint_arg, message_kind in PUBLISHERS
    ]
    executor = MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
        node.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.stop()
            executor.remove_node(node)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
