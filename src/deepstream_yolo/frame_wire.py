from __future__ import annotations

import json
import socket
import struct

HEADER = struct.Struct("!II")


def send_frame(sock: socket.socket, metadata: dict, payload: bytes) -> None:
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    sock.sendall(HEADER.pack(len(header), len(payload)))
    sock.sendall(header)
    sock.sendall(payload)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> tuple[dict, bytes]:
    header_size, payload_size = HEADER.unpack(recv_exact(sock, HEADER.size))
    metadata = json.loads(recv_exact(sock, header_size).decode("utf-8"))
    payload = recv_exact(sock, payload_size)
    return metadata, payload
