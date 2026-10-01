"""Tests for inter-operator messaging: CCSDS formatting, priority routing, coalition sharing."""

import struct
import time
import pytest

from src.space.messaging import (
    CCSDSMessage,
    CoalitionRegistry,
    MessageRouter,
    Priority,
)


# ── CCSDS Message Formatting ──────────────────────────────────────────


class TestCCSDSMessageFormatting:
    """Tests for CCSDS primary header packing and message serialization."""

    def test_ccsds_primary_header_packing(self):
        """Primary header: version(3) | type(1) | sec_hdr_flag(1) | APID(11) = 2 bytes."""
        msg = CCSDSMessage(apid=42, sequence_count=1, priority=Priority.NORMAL, payload=b"test", secondary_header=b"\x00")
        raw = msg.to_bytes()
        # First 2 bytes: version=0, type=0, sec_hdr_flag=1, APID=42
        word0 = struct.unpack(">H", raw[0:2])[0]
        assert (word0 >> 13) & 0x7 == 0  # version
        assert (word0 >> 12) & 0x1 == 0  # type
        assert (word0 >> 11) & 0x1 == 1  # secondary header flag
        assert word0 & 0x7FF == 42  # APID

    def test_ccsds_sequence_flags_and_count(self):
        """Bytes 2-3: sequence_flags(2) | sequence_count(14)."""
        msg = CCSDSMessage(apid=1, sequence_count=100, priority=Priority.NORMAL, payload=b"x")
        raw = msg.to_bytes()
        word1 = struct.unpack(">H", raw[2:4])[0]
        assert (word1 >> 14) & 0x3 == 0b11  # sequence flags = 3 (unsegmented)
        assert word1 & 0x3FFF == 100  # sequence count

    def test_ccsds_packet_length_field(self):
        """Bytes 4-5: packet length = total bytes after primary header - 1."""
        payload = b"hello"
        msg = CCSDSMessage(apid=1, sequence_count=0, priority=Priority.NORMAL, payload=payload)
        raw = msg.to_bytes()
        word2 = struct.unpack(">H", raw[4:6])[0]
        # Metadata prefix (7 bytes) + no user sec_hdr + payload - 1
        meta_len = 7  # priority(1) + src_len(2) + dst_len(2) + coal_len(2)
        assert word2 == meta_len + len(payload) - 1

    def test_ccsds_packet_length_with_secondary_header(self):
        """Packet length includes secondary header bytes."""
        payload = b"data"
        sec_hdr = b"\x00\x01\x02"
        msg = CCSDSMessage(
            apid=1, sequence_count=0, priority=Priority.NORMAL,
            payload=payload, secondary_header=sec_hdr,
        )
        raw = msg.to_bytes()
        word2 = struct.unpack(">H", raw[4:6])[0]
        # Metadata prefix (7) + user sec_hdr (3) + payload (4) - 1 = 13
        meta_len = 7
        assert word2 == meta_len + len(sec_hdr) + len(payload) - 1

    def test_ccsds_message_roundtrip(self):
        """to_bytes → from_bytes preserves all fields."""
        original = CCSDSMessage(
            apid=100,
            sequence_count=500,
            priority=Priority.HIGH,
            payload=b"orbital data",
            secondary_header=b"\xAA\xBB",
            source="OP-1",
            destination="OP-2",
            coalition="ALPHA",
        )
        raw = original.to_bytes()
        restored = CCSDSMessage.from_bytes(raw)
        assert restored.apid == original.apid
        assert restored.sequence_count == original.sequence_count
        assert restored.priority == original.priority
        assert restored.payload == original.payload
        assert restored.secondary_header == original.secondary_header
        assert restored.source == original.source
        assert restored.destination == original.destination
        assert restored.coalition == original.coalition

    def test_ccsds_sequence_count_wraps_at_16383(self):
        """Sequence count is 14 bits; values ≥ 16384 should wrap or raise."""
        msg = CCSDSMessage(apid=1, sequence_count=16383, priority=Priority.NORMAL, payload=b"x")
        raw = msg.to_bytes()
        word1 = struct.unpack(">H", raw[2:4])[0]
        assert word1 & 0x3FFF == 16383

    def test_ccsds_invalid_apid_raises(self):
        """APID must be 0–2047 (11 bits)."""
        with pytest.raises(ValueError):
            CCSDSMessage(apid=2048, sequence_count=0, priority=Priority.NORMAL, payload=b"x")

    def test_ccsds_invalid_sequence_count_raises(self):
        """Sequence count must be 0–16383 (14 bits)."""
        with pytest.raises(ValueError):
            CCSDSMessage(apid=1, sequence_count=16384, priority=Priority.NORMAL, payload=b"x")

    def test_ccsds_from_bytes_invalid_length_raises(self):
        """from_bytes should reject data shorter than 6-byte primary header."""
        with pytest.raises(ValueError):
            CCSDSMessage.from_bytes(b"\x00\x01\x02")


# ── Priority Routing ──────────────────────────────────────────────────


class TestPriorityRouting:
    """Tests for priority-based message routing."""

    def test_priority_ordering(self):
        """CRITICAL < HIGH < NORMAL < LOW."""
        assert Priority.CRITICAL < Priority.HIGH
        assert Priority.HIGH < Priority.NORMAL
        assert Priority.NORMAL < Priority.LOW

    def test_router_dispatches_critical_before_normal(self):
        """Higher priority messages are dispatched first."""
        router = MessageRouter()
        dispatched = []
        router.register_handler(Priority.CRITICAL, lambda m: dispatched.append(m))
        router.register_handler(Priority.NORMAL, lambda m: dispatched.append(m))

        normal_msg = CCSDSMessage(apid=1, sequence_count=1, priority=Priority.NORMAL, payload=b"n")
        critical_msg = CCSDSMessage(apid=2, sequence_count=2, priority=Priority.CRITICAL, payload=b"c")

        router.route(normal_msg)
        router.route(critical_msg)

        router.process_next()
        assert dispatched[0] is critical_msg

    def test_router_dispatches_fifo_within_same_priority(self):
        """Messages of the same priority are dispatched in FIFO order."""
        router = MessageRouter()
        dispatched = []
        router.register_handler(Priority.HIGH, lambda m: dispatched.append(m))

        msg1 = CCSDSMessage(apid=1, sequence_count=1, priority=Priority.HIGH, payload=b"1")
        msg2 = CCSDSMessage(apid=2, sequence_count=2, priority=Priority.HIGH, payload=b"2")
        msg3 = CCSDSMessage(apid=3, sequence_count=3, priority=Priority.HIGH, payload=b"3")

        router.route(msg1)
        router.route(msg2)
        router.route(msg3)

        router.process_next()
        router.process_next()
        router.process_next()
        assert dispatched == [msg1, msg2, msg3]

    def test_router_calls_registered_handler(self):
        """Handler is called with the routed message."""
        router = MessageRouter()
        received = []
        router.register_handler(Priority.LOW, received.append)

        msg = CCSDSMessage(apid=1, sequence_count=1, priority=Priority.LOW, payload=b"x")
        router.route(msg)
        router.process_next()
        assert received == [msg]

    def test_router_process_next_returns_none_when_empty(self):
        """process_next returns None when no messages are queued."""
        router = MessageRouter()
        assert router.process_next() is None

    def test_router_process_next_skips_unhandled_priorities(self):
        """If no handler for a priority, process_next skips to next priority with handler."""
        router = MessageRouter()
        dispatched = []
        router.register_handler(Priority.CRITICAL, lambda m: dispatched.append(m))

        normal_msg = CCSDSMessage(apid=1, sequence_count=1, priority=Priority.NORMAL, payload=b"n")
        critical_msg = CCSDSMessage(apid=2, sequence_count=2, priority=Priority.CRITICAL, payload=b"c")

        router.route(normal_msg)
        router.route(critical_msg)

        result = router.process_next()
        assert result is critical_msg
        assert dispatched == [critical_msg]


# ── Coalition Sharing ─────────────────────────────────────────────────


class TestCoalitionSharing:
    """Tests for coalition-based message sharing."""

    def test_coalition_create_and_add_operator(self):
        """Create a coalition and add operators."""
        reg = CoalitionRegistry()
        reg.create_coalition("ALPHA")
        reg.add_operator("ALPHA", "OP-1")
        reg.add_operator("ALPHA", "OP-2")
        assert reg.get_members("ALPHA") == {"OP-1", "OP-2"}

    def test_coalition_remove_operator(self):
        """Remove an operator from a coalition."""
        reg = CoalitionRegistry()
        reg.create_coalition("ALPHA")
        reg.add_operator("ALPHA", "OP-1")
        reg.add_operator("ALPHA", "OP-2")
        reg.remove_operator("ALPHA", "OP-1")
        assert reg.get_members("ALPHA") == {"OP-2"}

    def test_coalition_share_message_to_all_members(self):
        """Sharing a message produces one copy per coalition member."""
        reg = CoalitionRegistry()
        reg.create_coalition("ALPHA")
        reg.add_operator("ALPHA", "OP-1")
        reg.add_operator("ALPHA", "OP-2")
        reg.add_operator("ALPHA", "OP-3")

        msg = CCSDSMessage(apid=1, sequence_count=1, priority=Priority.HIGH, payload=b"alert")
        shared = reg.share_message(msg, "ALPHA")
        assert len(shared) == 3
        destinations = {m.destination for m in shared}
        assert destinations == {"OP-1", "OP-2", "OP-3"}

    def test_coalition_share_excludes_sender(self):
        """Sender should not receive their own shared message."""
        reg = CoalitionRegistry()
        reg.create_coalition("ALPHA")
        reg.add_operator("ALPHA", "OP-1")
        reg.add_operator("ALPHA", "OP-2")

        msg = CCSDSMessage(
            apid=1, sequence_count=1, priority=Priority.HIGH,
            payload=b"alert", source="OP-1",
        )
        shared = reg.share_message(msg, "ALPHA")
        destinations = {m.destination for m in shared}
        assert "OP-1" not in destinations
        assert destinations == {"OP-2"}

    def test_coalition_share_empty_coalition_returns_empty(self):
        """Sharing to a coalition with no members returns empty list."""
        reg = CoalitionRegistry()
        reg.create_coalition("EMPTY")
        msg = CCSDSMessage(apid=1, sequence_count=1, priority=Priority.HIGH, payload=b"x")
        assert reg.share_message(msg, "EMPTY") == []

    def test_coalition_duplicate_operator_add_is_idempotent(self):
        """Adding the same operator twice doesn't create duplicates."""
        reg = CoalitionRegistry()
        reg.create_coalition("ALPHA")
        reg.add_operator("ALPHA", "OP-1")
        reg.add_operator("ALPHA", "OP-1")
        assert reg.get_members("ALPHA") == {"OP-1"}

    def test_coalition_get_members_nonexistent_raises(self):
        """Getting members of a nonexistent coalition raises KeyError."""
        reg = CoalitionRegistry()
        with pytest.raises(KeyError):
            reg.get_members("NOPE")

    def test_coalition_operator_in_multiple_coalitions(self):
        """An operator can belong to multiple coalitions."""
        reg = CoalitionRegistry()
        reg.create_coalition("ALPHA")
        reg.create_coalition("BETA")
        reg.add_operator("ALPHA", "OP-1")
        reg.add_operator("BETA", "OP-1")
        assert reg.get_members("ALPHA") == {"OP-1"}
        assert reg.get_members("BETA") == {"OP-1"}

    def test_coalition_share_message_preserves_payload(self):
        """Shared messages preserve the original payload."""
        reg = CoalitionRegistry()
        reg.create_coalition("ALPHA")
        reg.add_operator("ALPHA", "OP-1")
        reg.add_operator("ALPHA", "OP-2")

        msg = CCSDSMessage(apid=1, sequence_count=1, priority=Priority.HIGH, payload=b"critical data")
        shared = reg.share_message(msg, "ALPHA")
        for m in shared:
            assert m.payload == b"critical data"
            assert m.priority == Priority.HIGH
