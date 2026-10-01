"""Inter-operator messaging: CCSDS formatting, priority routing, coalition sharing."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Set


class Priority(IntEnum):
    """Message priority levels (lower value = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class CCSDSMessage:
    """CCSDS-compliant message with primary header and optional secondary header.

    Primary header (6 bytes):
      - Version (3 bits) | Type (1 bit) | Secondary Header Flag (1 bit) | APID (11 bits)
      - Sequence Flags (2 bits) | Sequence Count (14 bits)
      - Packet Length (16 bits)
    """
    apid: int
    sequence_count: int
    priority: Priority
    payload: bytes
    secondary_header: bytes = b""
    source: str = ""
    destination: str = ""
    coalition: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not 0 <= self.apid <= 0x7FF:
            raise ValueError(f"APID must be 0-2047, got {self.apid}")
        if not 0 <= self.sequence_count <= 0x3FFF:
            raise ValueError(f"Sequence count must be 0-16383, got {self.sequence_count}")

    def to_bytes(self) -> bytes:
        """Serialize to CCSDS binary format."""
        # Encode metadata in secondary header:
        # [priority(1), src_len(2), dst_len(2), coal_len(2), src, dst, coal, ...user_sec_hdr]
        src_b = self.source.encode("utf-8")
        dst_b = self.destination.encode("utf-8")
        coal_b = self.coalition.encode("utf-8")
        sec_hdr = (
            struct.pack(">BHHH", int(self.priority), len(src_b), len(dst_b), len(coal_b))
            + src_b + dst_b + coal_b + self.secondary_header
        )

        # Word 0: version(3) | type(1) | sec_hdr_flag(1) | APID(11)
        sec_hdr_flag = 1 if sec_hdr else 0
        word0 = (0 << 13) | (0 << 12) | (sec_hdr_flag << 11) | self.apid

        # Word 1: sequence_flags(2) | sequence_count(14)
        word1 = (0b11 << 14) | self.sequence_count

        # Word 2: packet length = total bytes after primary header - 1
        packet_length = len(sec_hdr) + len(self.payload) - 1
        if packet_length < 0:
            packet_length = 0

        header = struct.pack(">HHH", word0, word1, packet_length)
        if sec_hdr:
            body = bytes([len(sec_hdr)]) + sec_hdr + self.payload
        else:
            body = self.payload
        return header + body

    @classmethod
    def from_bytes(cls, data: bytes) -> "CCSDSMessage":
        """Deserialize from CCSDS binary format."""
        if len(data) < 6:
            raise ValueError(f"CCSDS message must be at least 6 bytes, got {len(data)}")

        word0, word1, word2 = struct.unpack(">HHH", data[0:6])

        apid = word0 & 0x7FF
        sec_hdr_flag = (word0 >> 11) & 0x1
        sequence_count = word1 & 0x3FFF
        packet_length = word2

        body = data[6:]
        priority = Priority.NORMAL
        source = ""
        destination = ""
        coalition = ""
        secondary_header = b""
        if sec_hdr_flag:
            # Secondary header: first byte is length, then metadata
            if len(body) < 1:
                raise ValueError("Secondary header flag set but no data")
            sec_len = body[0]
            sec_data = body[1:1 + sec_len]
            if sec_data:
                prio_val, src_len, dst_len, coal_len = struct.unpack(">BHHH", sec_data[0:7])
                priority = Priority(prio_val)
                offset = 7
                source = sec_data[offset:offset + src_len].decode("utf-8")
                offset += src_len
                destination = sec_data[offset:offset + dst_len].decode("utf-8")
                offset += dst_len
                coalition = sec_data[offset:offset + coal_len].decode("utf-8")
                offset += coal_len
                secondary_header = sec_data[offset:]
            payload = body[1 + sec_len:]
        else:
            payload = body

        return cls(
            apid=apid,
            sequence_count=sequence_count,
            priority=priority,
            payload=payload,
            secondary_header=secondary_header,
            source=source,
            destination=destination,
            coalition=coalition,
        )


class MessageRouter:
    """Priority-based message router with per-priority FIFO queues."""

    def __init__(self):
        self._handlers: Dict[Priority, List[Callable]] = {}
        self._queues: Dict[Priority, List[CCSDSMessage]] = {
            p: [] for p in Priority
        }

    def register_handler(self, priority: Priority, handler: Callable):
        """Register a handler for a given priority level."""
        if priority not in self._handlers:
            self._handlers[priority] = []
        self._handlers[priority].append(handler)

    def route(self, message: CCSDSMessage):
        """Queue a message for routing based on its priority."""
        self._queues[message.priority].append(message)

    def process_next(self) -> Optional[CCSDSMessage]:
        """Process the next highest-priority message. Returns None if empty."""
        for priority in sorted(Priority):
            if self._queues[priority]:
                msg = self._queues[priority].pop(0)
                if priority in self._handlers:
                    for handler in self._handlers[priority]:
                        handler(msg)
                return msg
        return None


class CoalitionRegistry:
    """Registry for coalition-based message sharing."""

    def __init__(self):
        self._coalitions: Dict[str, Set[str]] = {}

    def create_coalition(self, name: str):
        """Create a new coalition."""
        if name not in self._coalitions:
            self._coalitions[name] = set()

    def add_operator(self, coalition: str, operator: str):
        """Add an operator to a coalition."""
        if coalition not in self._coalitions:
            raise KeyError(f"Coalition '{coalition}' does not exist")
        self._coalitions[coalition].add(operator)

    def remove_operator(self, coalition: str, operator: str):
        """Remove an operator from a coalition."""
        if coalition not in self._coalitions:
            raise KeyError(f"Coalition '{coalition}' does not exist")
        self._coalitions[coalition].discard(operator)

    def get_members(self, coalition: str) -> Set[str]:
        """Get all members of a coalition."""
        if coalition not in self._coalitions:
            raise KeyError(f"Coalition '{coalition}' does not exist")
        return self._coalitions[coalition].copy()

    def share_message(self, message: CCSDSMessage, coalition: str) -> List[CCSDSMessage]:
        """Share a message to all coalition members except the sender."""
        if coalition not in self._coalitions:
            raise KeyError(f"Coalition '{coalition}' does not exist")

        members = self._coalitions[coalition]
        shared = []
        for member in members:
            if member == message.source:
                continue
            copy = CCSDSMessage(
                apid=message.apid,
                sequence_count=message.sequence_count,
                priority=message.priority,
                payload=message.payload,
                secondary_header=message.secondary_header,
                source=message.source,
                destination=member,
                coalition=coalition,
                timestamp=message.timestamp,
            )
            shared.append(copy)
        return shared
