from __future__ import annotations

import json
import socket
import struct
import time

HEADER = struct.Struct("!II")

# How long the peer may stay silent part-way through a frame before the
# connection is treated as unusable.
PARTIAL_TIMEOUT = 30.0

# frame_timestamp() reports where a frame's timestamp came from. Only these two
# are Unix-epoch wall clock; "buf_pts" and "pts" are stream-relative and start
# near zero, so writing them into a ROS header dates the message to 1970.
# This lives here rather than in assessment_runtime because the ROS bridge runs
# in a container with no DeepStream, and cannot import anything that needs gi.
WALL_CLOCK_TIMESTAMP_SOURCES = frozenset({"ntp", "ref"})


def is_wall_clock_timestamp(source: str | None) -> bool:
    return source in WALL_CLOCK_TIMESTAMP_SOURCES


def send_frame(sock: socket.socket, metadata: dict, payload: bytes) -> None:
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    sock.sendall(HEADER.pack(len(header), len(payload)))
    sock.sendall(header)
    sock.sendall(payload)


def recv_exact(
    sock: socket.socket,
    size: int,
    at_frame_start: bool = False,
    partial_timeout: float = PARTIAL_TIMEOUT,
) -> bytes:
    """Read exactly ``size`` bytes.

    Receivers set a short socket timeout so they can poll a stop flag between
    frames, which is only safe at a frame boundary. Letting a timeout escape
    part-way through a frame discards the bytes already read, and the caller
    then restarts framing mid-stream: payload bytes get read as an 8-byte
    header, producing nonsense lengths that the connection never recovers from.

    A timeout therefore propagates only when ``at_frame_start`` and nothing has
    been read yet. Otherwise keep waiting, and if the peer stays silent for
    ``partial_timeout`` raise ``ConnectionError`` so the caller drops the
    connection; the sender reconnects and framing resynchronizes.
    """
    chunks: list[bytes] = []
    remaining = size
    stall_deadline: float | None = None

    while remaining:
        try:
            chunk = sock.recv(remaining)
        except socket.timeout:
            if at_frame_start and not chunks:
                raise
            now = time.monotonic()
            if stall_deadline is None:
                stall_deadline = now + partial_timeout
            elif now >= stall_deadline:
                raise ConnectionError(
                    f"peer stalled mid-frame: {remaining} of {size} bytes outstanding"
                )
            continue

        if not chunk:
            raise EOFError("socket closed")

        chunks.append(chunk)
        remaining -= len(chunk)
        stall_deadline = None

    return b"".join(chunks)


def recv_frame(
    sock: socket.socket, partial_timeout: float = PARTIAL_TIMEOUT
) -> tuple[dict, bytes]:
    header_bytes = recv_exact(
        sock, HEADER.size, at_frame_start=True, partial_timeout=partial_timeout
    )
    header_size, payload_size = HEADER.unpack(header_bytes)
    metadata_bytes = recv_exact(sock, header_size, partial_timeout=partial_timeout)
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Once the stream is misaligned nothing downstream is trustworthy;
        # force a reconnect rather than keep reading garbage.
        raise ConnectionError(f"malformed frame metadata: {exc}") from exc
    payload = recv_exact(sock, payload_size, partial_timeout=partial_timeout)
    return metadata, payload
