"""CCSDS protocol stack, priority routing, and coalition federation.

Implements a deeper CCSDS-compliant messaging layer with:
- Primary/secondary header packing and unpacking
- Full protocol stack encode/decode
- Priority-based message routing with per-priority FIFO queues
- Coalition federation for cross-coalition message sharing
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Set


class Priority(IntEnum):
    """Message priority levels (lower value = higher priority)."""
    EMERGENCY = 0
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class CCSDSPrimaryHeader:
    """CCSDS primary header (6 bytes).

    Layout:
      - Version (3 bits) | Type (1 bit) | Secondary Header Flag (1 bit) | APID (11 bits)
      - Sequence Flags (2 bits) | Sequence Count (14 bits)
      - Packet Length (16 bits)
    """
    apid: int = 0
    sequence_count: int = 0
    version: int = 0
    type: int = 0
    secondary_header_flag: int = 1
    sequence_flags: int = 0b11
    packet_length: int = 0

    def __post_init__(self):
        if not 0 <= self.apid <= 0x7FF:
            raise ValueError(f"APID must be 0-2047, got {self.apid}")
        if not 0 <= self.sequence_count <= 0x3FFF:
            raise ValueError(f"Sequence count must be 0-16383, got {self.sequence_count}")

    def pack(self) -> bytes:
        """Pack primary header into 6 bytes."""
        word0 = (self.version << 13) | (self.type << 12) | (self.secondary_header_flag << 11) | self.apid
        word1 = (self.sequence_flags << 14) | self.sequence_count
        return struct.pack(">HHH", word0, word1, self.packet_length)

    @classmethod
    def unpack(cls, data: bytes) -> "CCSDSPrimaryHeader":
        """Unpack 6 bytes into a primary header."""
        if len(data) < 6:
            raise ValueError(f"Primary header must be at least 6 bytes, got {len(data)}")
        word0, word1, word2 = struct.unpack(">HHH", data[0:6])
        return cls(
            version=(word0 >> 13) & 0x7,
            type=(word0 >> 12) & 0x1,
            secondary_header_flag=(word0 >> 11) & 0x1,
            apid=word0 & 0x7FF,
            sequence_flags=(word1 >> 14) & 0x3,
            sequence_count=word1 & 0x3FFF,
            packet_length=word2,
        )


@dataclass
class CCSDSSecondaryHeader:
    """CCSDS secondary header.

    Layout:
      - Timestamp (8 bytes, double)
      - Spacecraft ID (2 bytes)
      - Data Length (2 bytes)
      - Data (variable)
    """
    timestamp: float = 0.0
    spacecraft_id: int = 0
    data: bytes = b""

    def pack(self) -> bytes:
        """Pack secondary header into bytes."""
        return struct.pack(">QHH", int(self.timestamp), self.spacecraft_id, len(self.data)) + self.data

    @classmethod
    def unpack(cls, data: bytes) -> "CCSDSSecondaryHeader":
        """Unpack bytes into a secondary header."""
        if len(data) < 12:
            raise ValueError(f"Secondary header must be at least 12 bytes, got {len(data)}")
        ts, scid, dlen = struct.unpack(">QHH", data[0:12])
        return cls(timestamp=float(ts), spacecraft_id=scid, data=data[12:12 + dlen])


@dataclass
class CCSDSMessage:
    """Full CCSDS message with primary header, secondary header, and payload."""
    header: CCSDSPrimaryHeader
    secondary_header: CCSDSSecondaryHeader
    payload: bytes
    priority: Priority = Priority.NORMAL
    source: str = ""
    destination: str = ""
    coalition: str = ""
    timestamp: float = field(default_factory=time.time)


class CCSDSProtocolStack:
    """CCSDS protocol stack for encoding and decoding messages."""

    def encode(self, message: CCSDSMessage) -> bytes:
        """Encode a CCSDSMessage into bytes."""
        sec_hdr = message.secondary_header.pack()
        # Metadata: priority(1) + src_len(2) + dst_len(2) + coal_len(2) + src + dst + coal
        src_b = message.source.encode("utf-8")
        dst_b = message.destination.encode("utf-8")
        coal_b = message.coalition.encode("utf-8")
        meta = struct.pack(">BHHH", int(message.priority), len(src_b), len(dst_b), len(coal_b))
        meta += src_b + dst_b + coal_b

        # Update packet length
        body = bytes([len(sec_hdr)]) + sec_hdr + meta + message.payload
        message.header.packet_length = len(body) - 1

        return message.header.pack() + body

    def decode(self, data: bytes) -> CCSDSMessage:
        """Decode bytes into a CCSDSMessage."""
        if len(data) < 6:
            raise ValueError(f"CCSDS message must be at least 6 bytes, got {len(data)}")

        header = CCSDSPrimaryHeader.unpack(data[0:6])
        body = data[6:]

        secondary_header = CCSDSSecondaryHeader()
        priority = Priority.NORMAL
        source = ""
        destination = ""
        coalition = ""
        payload = b""

        if header.secondary_header_flag:
            if len(body) < 1:
                raise ValueError("Secondary header flag set but no data")
            sec_len = body[0]
            sec_data = body[1:1 + sec_len]
            if sec_data:
                ts, scid, dlen = struct.unpack(">QHH", sec_data[0:12])
                secondary_header = CCSDSSecondaryHeader(
                    timestamp=float(ts), spacecraft_id=scid, data=sec_data[12:12 + dlen]
                )
            meta_and_payload = body[1 + sec_len:]
        else:
            meta_and_payload = body

        # Parse metadata from the start of meta_and_payload
        if len(meta_and_payload) >= 7:
            prio_val, src_len, dst_len, coal_len = struct.unpack(">BHHH", meta_and_payload[0:7])
            priority = Priority(prio_val)
            offset = 7
            source = meta_and_payload[offset:offset + src_len].decode("utf-8")
            offset += src_len
            destination = meta_and_payload[offset:offset + dst_len].decode("utf-8")
            offset += dst_len
            coalition = meta_and_payload[offset:offset + coal_len].decode("utf-8")
            offset += coal_len
            payload = meta_and_payload[offset:]
        else:
            payload = meta_and_payload

        return CCSDSMessage(
            header=header,
            secondary_header=secondary_header,
            payload=payload,
            priority=priority,
            source=source,
            destination=destination,
            coalition=coalition,
        )


class PriorityRouter:
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


class CoalitionFederation:
    """Federation for coalition-based message sharing across allied coalitions."""

    def __init__(self):
        self._coalitions: Dict[str, Set[str]] = {}
        self._alliances: Dict[str, Set[str]] = {}

    def create_coalition(self, name: str):
        """Create a new coalition."""
        if name not in self._coalitions:
            self._coalitions[name] = set()
            self._alliances[name] = set()

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

    def federate(self, coalition_a: str, coalition_b: str):
        """Establish a federation alliance between two coalitions."""
        if coalition_a not in self._coalitions:
            raise KeyError(f"Coalition '{coalition_a}' does not exist")
        if coalition_b not in self._coalitions:
            raise KeyError(f"Coalition '{coalition_b}' does not exist")
        self._alliances[coalition_a].add(coalition_b)
        self._alliances[coalition_b].add(coalition_a)

    def unfederate(self, coalition_a: str, coalition_b: str):
        """Remove a federation alliance between two coalitions."""
        if coalition_a in self._alliances:
            self._alliances[coalition_a].discard(coalition_b)
        if coalition_b in self._alliances:
            self._alliances[coalition_b].discard(coalition_a)

    def get_allies(self, coalition: str) -> Set[str]:
        """Get all allied coalitions for a given coalition."""
        if coalition not in self._alliances:
            raise KeyError(f"Coalition '{coalition}' does not exist")
        return self._alliances[coalition].copy()

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
                header=message.header,
                secondary_header=message.secondary_header,
                payload=message.payload,
                priority=message.priority,
                source=message.source,
                destination=member,
                coalition=coalition,
                timestamp=message.timestamp,
            )
            shared.append(copy)
        return shared

    def share_to_federation(self, message: CCSDSMessage, coalition: str) -> List[CCSDSMessage]:
        """Share a message to all members of allied coalitions."""
        if coalition not in self._coalitions:
            raise KeyError(f"Coalition '{coalition}' does not exist")

        allies = self._alliances.get(coalition, set())
        shared = []
        for ally in allies:
            if ally not in self._coalitions:
                continue
            members = self._coalitions[ally]
            for member in members:
                if member == message.source:
                    continue
                copy = CCSDSMessage(
                    header=message.header,
                    secondary_header=message.secondary_header,
                    payload=message.payload,
                    priority=message.priority,
                    source=message.source,
                    destination=member,
                    coalition=ally,
                    timestamp=message.timestamp,
                )
                shared.append(copy)
        return shared
