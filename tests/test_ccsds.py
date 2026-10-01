"""Tests for CCSDS protocol stack, priority routing, and coalition federation."""

import struct
import time
import pytest

from src.space.ccsds import (
    Priority,
    CCSDSPrimaryHeader,
    CCSDSSecondaryHeader,
    CCSDSMessage,
    CCSDSProtocolStack,
    PriorityRouter,
    CoalitionFederation,
)


# ── CCSDS Primary Header ──────────────────────────────────────────────


class TestCCSDSPrimaryHeader:
    """Tests for CCSDS primary header packing and field extraction."""

    def test_primary_header_packing(self):
        """Primary header: version(3) | type(1) | sec_hdr_flag(1) | APID(11) = 2 bytes."""
        hdr = CCSDSPrimaryHeader(apid=42, sequence_count=1)
        raw = hdr.pack()
        word0 = struct.unpack(">H", raw[0:2])[0]
        assert (word0 >> 13) & 0x7 == 0  # version
        assert (word0 >> 12) & 0x1 == 0  # type
        assert (word0 >> 11) & 0x1 == 1  # secondary header flag
        assert word0 & 0x7FF == 42  # APID

    def test_primary_header_sequence_flags_and_count(self):
        """Bytes 2-3: sequence_flags(2) | sequence_count(14)."""
        hdr = CCSDSPrimaryHeader(apid=1, sequence_count=100)
        raw = hdr.pack()
        word1 = struct.unpack(">H", raw[2:4])[0]
        assert (word1 >> 14) & 0x3 == 0b11  # unsegmented
        assert word1 & 0x3FFF == 100

    def test_primary_header_packet_length(self):
        """Bytes 4-5: packet length = total bytes after primary header - 1."""
        hdr = CCSDSPrimaryHeader(apid=1, sequence_count=0, packet_length=9)
        raw = hdr.pack()
        word2 = struct.unpack(">H", raw[4:6])[0]
        assert word2 == 9

    def test_primary_header_unpack_roundtrip(self):
        """pack → unpack preserves all fields."""
        hdr = CCSDSPrimaryHeader(
            version=0, type=1, secondary_header_flag=1,
            apid=2047, sequence_flags=3, sequence_count=16383,
            packet_length=100,
        )
        raw = hdr.pack()
        restored = CCSDSPrimaryHeader.unpack(raw)
        assert restored.version == 0
        assert restored.type == 1
        assert restored.secondary_header_flag == 1
        assert restored.apid == 2047
        assert restored.sequence_flags == 3
        assert restored.sequence_count == 16383
        assert restored.packet_length == 100

    def test_primary_header_invalid_apid_raises(self):
        """APID must be 0–2047 (11 bits)."""
        with pytest.raises(ValueError):
            CCSDSPrimaryHeader(apid=2048)

    def test_primary_header_invalid_sequence_count_raises(self):
        """Sequence count must be 0–16383 (14 bits)."""
        with pytest.raises(ValueError):
            CCSDSPrimaryHeader(apid=1, sequence_count=16384)

    def test_primary_header_unpack_short_data_raises(self):
        """unpack requires at least 6 bytes."""
        with pytest.raises(ValueError):
            CCSDSPrimaryHeader.unpack(b"\x00\x01\x02")


# ── CCSDS Secondary Header ────────────────────────────────────────────


class TestCCSDSSecondaryHeader:
    """Tests for CCSDS secondary header packing."""

    def test_secondary_header_packing(self):
        """Secondary header: timestamp(8) | spacecraft_id(2) | data_len(2) | data."""
        sec = CCSDSSecondaryHeader(timestamp=1234567890.0, spacecraft_id=42, data=b"\x01\x02")
        raw = sec.pack()
        ts, scid, dlen = struct.unpack(">QHH", raw[0:12])
        assert ts == 1234567890
        assert scid == 42
        assert dlen == 2
        assert raw[12:14] == b"\x01\x02"

    def test_secondary_header_unpack_roundtrip(self):
        """pack → unpack preserves all fields."""
        sec = CCSDSSecondaryHeader(timestamp=987654321.0, spacecraft_id=99, data=b"\xAA\xBB\xCC")
        raw = sec.pack()
        restored = CCSDSSecondaryHeader.unpack(raw)
        assert restored.timestamp == 987654321.0
        assert restored.spacecraft_id == 99
        assert restored.data == b"\xAA\xBB\xCC"

    def test_secondary_header_empty_data(self):
        """Secondary header with no user data."""
        sec = CCSDSSecondaryHeader(timestamp=0.0, spacecraft_id=0, data=b"")
        raw = sec.pack()
        assert len(raw) == 12  # 8 + 2 + 2
        restored = CCSDSSecondaryHeader.unpack(raw)
        assert restored.data == b""


# ── CCSDS Protocol Stack ──────────────────────────────────────────────


class TestCCSDSProtocolStack:
    """Tests for full protocol stack encode/decode."""

    def test_stack_encode_decode_roundtrip(self):
        """Full message roundtrip through protocol stack."""
        stack = CCSDSProtocolStack()
        msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=100, sequence_count=500),
            secondary_header=CCSDSSecondaryHeader(timestamp=1234567890.0, spacecraft_id=1),
            payload=b"orbital data",
            priority=Priority.HIGH,
            source="OP-1",
            destination="OP-2",
            coalition="ALPHA",
        )
        raw = stack.encode(msg)
        restored = stack.decode(raw)
        assert restored.header.apid == 100
        assert restored.header.sequence_count == 500
        assert restored.secondary_header.timestamp == 1234567890.0
        assert restored.secondary_header.spacecraft_id == 1
        assert restored.payload == b"orbital data"
        assert restored.priority == Priority.HIGH
        assert restored.source == "OP-1"
        assert restored.destination == "OP-2"
        assert restored.coalition == "ALPHA"

    def test_stack_encode_produces_valid_ccsds(self):
        """Encoded message starts with valid CCSDS primary header."""
        stack = CCSDSProtocolStack()
        msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=0),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"test",
            priority=Priority.NORMAL,
            source="",
            destination="",
            coalition="",
        )
        raw = stack.encode(msg)
        assert len(raw) >= 6
        word0 = struct.unpack(">H", raw[0:2])[0]
        assert (word0 >> 13) & 0x7 == 0  # version
        assert word0 & 0x7FF == 1  # APID

    def test_stack_decode_invalid_length_raises(self):
        """decode rejects data shorter than 6-byte primary header."""
        stack = CCSDSProtocolStack()
        with pytest.raises(ValueError):
            stack.decode(b"\x00\x01\x02")

    def test_stack_decode_empty_payload(self):
        """Message with empty payload roundtrips correctly."""
        stack = CCSDSProtocolStack()
        msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=0),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"",
            priority=Priority.NORMAL,
            source="",
            destination="",
            coalition="",
        )
        raw = stack.encode(msg)
        restored = stack.decode(raw)
        assert restored.payload == b""

    def test_stack_message_size(self):
        """Message size = primary header + secondary header + payload."""
        stack = CCSDSProtocolStack()
        msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=0),
            secondary_header=CCSDSSecondaryHeader(data=b"\x01\x02"),
            payload=b"hello",
            priority=Priority.NORMAL,
            source="",
            destination="",
            coalition="",
        )
        raw = stack.encode(msg)
        # 6 (primary) + 1 (sec_len) + 12 (sec_hdr) + 2 (sec_data) + 7 (meta) + 5 (payload) = 33
        assert len(raw) == 33


# ── Priority Routing ──────────────────────────────────────────────────


class TestPriorityRouting:
    """Tests for priority-based message routing."""

    def test_priority_ordering(self):
        """EMERGENCY < CRITICAL < HIGH < NORMAL < LOW."""
        assert Priority.EMERGENCY < Priority.CRITICAL
        assert Priority.CRITICAL < Priority.HIGH
        assert Priority.HIGH < Priority.NORMAL
        assert Priority.NORMAL < Priority.LOW

    def test_router_dispatches_critical_before_normal(self):
        """Higher priority messages are dispatched first."""
        router = PriorityRouter()
        dispatched = []
        router.register_handler(Priority.CRITICAL, lambda m: dispatched.append(m))
        router.register_handler(Priority.NORMAL, lambda m: dispatched.append(m))

        normal_msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=1),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"n", priority=Priority.NORMAL, source="", destination="", coalition="",
        )
        critical_msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=2, sequence_count=2),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"c", priority=Priority.CRITICAL, source="", destination="", coalition="",
        )

        router.route(normal_msg)
        router.route(critical_msg)
        router.process_next()
        assert dispatched[0] is critical_msg

    def test_router_dispatches_fifo_within_same_priority(self):
        """Messages of the same priority are dispatched in FIFO order."""
        router = PriorityRouter()
        dispatched = []
        router.register_handler(Priority.HIGH, lambda m: dispatched.append(m))

        msgs = []
        for i in range(3):
            msgs.append(CCSDSMessage(
                header=CCSDSPrimaryHeader(apid=i, sequence_count=i),
                secondary_header=CCSDSSecondaryHeader(),
                payload=f"{i}".encode(), priority=Priority.HIGH,
                source="", destination="", coalition="",
            ))
            router.route(msgs[-1])

        for _ in range(3):
            router.process_next()
        assert dispatched == msgs

    def test_router_process_next_returns_none_when_empty(self):
        """process_next returns None when no messages are queued."""
        router = PriorityRouter()
        assert router.process_next() is None

    def test_router_process_next_skips_unhandled_priorities(self):
        """If no handler for a priority, process_next skips to next priority with handler."""
        router = PriorityRouter()
        dispatched = []
        router.register_handler(Priority.CRITICAL, lambda m: dispatched.append(m))

        normal_msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=1),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"n", priority=Priority.NORMAL, source="", destination="", coalition="",
        )
        critical_msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=2, sequence_count=2),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"c", priority=Priority.CRITICAL, source="", destination="", coalition="",
        )

        router.route(normal_msg)
        router.route(critical_msg)
        result = router.process_next()
        assert result is critical_msg
        assert dispatched == [critical_msg]

    def test_router_multiple_handlers_same_priority(self):
        """Multiple handlers for the same priority are all called."""
        router = PriorityRouter()
        calls = []
        router.register_handler(Priority.HIGH, lambda m: calls.append("h1"))
        router.register_handler(Priority.HIGH, lambda m: calls.append("h2"))

        msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=1),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"x", priority=Priority.HIGH, source="", destination="", coalition="",
        )
        router.route(msg)
        router.process_next()
        assert calls == ["h1", "h2"]


# ── Coalition Federation ──────────────────────────────────────────────


class TestCoalitionFederation:
    """Tests for coalition federation and cross-coalition sharing."""

    def test_coalition_create_and_add_operator(self):
        """Create a coalition and add operators."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.add_operator("ALPHA", "OP-1")
        fed.add_operator("ALPHA", "OP-2")
        assert fed.get_members("ALPHA") == {"OP-1", "OP-2"}

    def test_coalition_remove_operator(self):
        """Remove an operator from a coalition."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.add_operator("ALPHA", "OP-1")
        fed.add_operator("ALPHA", "OP-2")
        fed.remove_operator("ALPHA", "OP-1")
        assert fed.get_members("ALPHA") == {"OP-2"}

    def test_coalition_share_message_to_all_members(self):
        """Sharing a message produces one copy per coalition member."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.add_operator("ALPHA", "OP-1")
        fed.add_operator("ALPHA", "OP-2")
        fed.add_operator("ALPHA", "OP-3")

        msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=1),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"alert", priority=Priority.HIGH, source="", destination="", coalition="ALPHA",
        )
        shared = fed.share_message(msg, "ALPHA")
        assert len(shared) == 3
        destinations = {m.destination for m in shared}
        assert destinations == {"OP-1", "OP-2", "OP-3"}

    def test_coalition_share_excludes_sender(self):
        """Sender should not receive their own shared message."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.add_operator("ALPHA", "OP-1")
        fed.add_operator("ALPHA", "OP-2")

        msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=1),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"alert", priority=Priority.HIGH, source="OP-1", destination="", coalition="ALPHA",
        )
        shared = fed.share_message(msg, "ALPHA")
        destinations = {m.destination for m in shared}
        assert "OP-1" not in destinations
        assert destinations == {"OP-2"}

    def test_federation_cross_coalition_share(self):
        """Federation allows sharing messages between allied coalitions."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.create_coalition("BETA")
        fed.add_operator("ALPHA", "OP-1")
        fed.add_operator("BETA", "OP-2")
        fed.federate("ALPHA", "BETA")

        msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=1),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"intel", priority=Priority.HIGH, source="OP-1", destination="", coalition="ALPHA",
        )
        shared = fed.share_to_federation(msg, "ALPHA")
        destinations = {m.destination for m in shared}
        assert "OP-2" in destinations

    def test_federation_non_federated_coalition_excluded(self):
        """Non-federated coalitions don't receive shared messages."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.create_coalition("BETA")
        fed.create_coalition("GAMMA")
        fed.add_operator("ALPHA", "OP-1")
        fed.add_operator("BETA", "OP-2")
        fed.add_operator("GAMMA", "OP-3")
        fed.federate("ALPHA", "BETA")

        msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=1),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"intel", priority=Priority.HIGH, source="OP-1", destination="", coalition="ALPHA",
        )
        shared = fed.share_to_federation(msg, "ALPHA")
        destinations = {m.destination for m in shared}
        assert "OP-2" in destinations
        assert "OP-3" not in destinations

    def test_federation_get_allies(self):
        """Get all allied coalitions for a given coalition."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.create_coalition("BETA")
        fed.create_coalition("GAMMA")
        fed.federate("ALPHA", "BETA")
        fed.federate("ALPHA", "GAMMA")
        allies = fed.get_allies("ALPHA")
        assert allies == {"BETA", "GAMMA"}

    def test_federation_remove_alliance(self):
        """Remove a federation alliance."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.create_coalition("BETA")
        fed.federate("ALPHA", "BETA")
        fed.unfederate("ALPHA", "BETA")
        allies = fed.get_allies("ALPHA")
        assert allies == set()

    def test_coalition_duplicate_operator_add_is_idempotent(self):
        """Adding the same operator twice doesn't create duplicates."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.add_operator("ALPHA", "OP-1")
        fed.add_operator("ALPHA", "OP-1")
        assert fed.get_members("ALPHA") == {"OP-1"}

    def test_coalition_get_members_nonexistent_raises(self):
        """Getting members of a nonexistent coalition raises KeyError."""
        fed = CoalitionFederation()
        with pytest.raises(KeyError):
            fed.get_members("NOPE")

    def test_coalition_operator_in_multiple_coalitions(self):
        """An operator can belong to multiple coalitions."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.create_coalition("BETA")
        fed.add_operator("ALPHA", "OP-1")
        fed.add_operator("BETA", "OP-1")
        assert fed.get_members("ALPHA") == {"OP-1"}
        assert fed.get_members("BETA") == {"OP-1"}

    def test_federation_share_message_preserves_payload(self):
        """Federated shared messages preserve the original payload."""
        fed = CoalitionFederation()
        fed.create_coalition("ALPHA")
        fed.create_coalition("BETA")
        fed.add_operator("ALPHA", "OP-1")
        fed.add_operator("BETA", "OP-2")
        fed.federate("ALPHA", "BETA")

        msg = CCSDSMessage(
            header=CCSDSPrimaryHeader(apid=1, sequence_count=1),
            secondary_header=CCSDSSecondaryHeader(),
            payload=b"critical data", priority=Priority.HIGH,
            source="OP-1", destination="", coalition="ALPHA",
        )
        shared = fed.share_to_federation(msg, "ALPHA")
        for m in shared:
            assert m.payload == b"critical data"
            assert m.priority == Priority.HIGH
