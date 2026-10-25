"""Small, dependency-free readers for the custom recorder's wire format."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class BagFormatError(ValueError):
    """Raised when a recorder file does not match the expected protobuf layout."""


@dataclass(frozen=True)
class Field:
    number: int
    wire_type: int
    value: int | bytes


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise BagFormatError('truncated varint')
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7f) << shift
        if byte < 0x80:
            return value, offset
    raise BagFormatError('varint is longer than 10 bytes')


def parse_fields(data: bytes) -> list[Field]:
    """Parse the protobuf wire types used by the recording."""
    fields: list[Field] = []
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        number, wire_type = key >> 3, key & 7
        if number == 0:
            raise BagFormatError('protobuf field number 0 is invalid')
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise BagFormatError('truncated fixed64 field')
            value, offset = data[offset:end], end
        elif wire_type == 2:
            size, offset = _read_varint(data, offset)
            end = offset + size
            if end > len(data):
                raise BagFormatError('truncated length-delimited field')
            value, offset = data[offset:end], end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise BagFormatError('truncated fixed32 field')
            value, offset = data[offset:end], end
        else:
            raise BagFormatError(f'unsupported protobuf wire type {wire_type}')
        fields.append(Field(number, wire_type, value))
    return fields


def field_map(data: bytes) -> dict[int, int | bytes]:
    """Return singular protobuf fields; later occurrences replace earlier ones."""
    return {field.number: field.value for field in parse_fields(data)}


def repeated_bytes(data: bytes, number: int) -> list[bytes]:
    values = []
    for field in parse_fields(data):
        if field.number == number and field.wire_type == 2:
            assert isinstance(field.value, bytes)
            values.append(field.value)
    return values


def iter_record_payloads(path: Path) -> Iterator[bytes]:
    """Yield application messages from the recorder's two protobuf envelopes.

    A chunk is a concatenation of records. Each record has fields 1 (recorder
    timestamp) and 2 (an inner envelope); the inner envelope repeats that shape.
    There is intentionally no outer length prefix.
    """
    fields = parse_fields(path.read_bytes())
    # Field 1 is chunk metadata. Each repeated field 2 is one recorded message.
    payload_fields = [field for field in fields if field.number == 2]
    if not payload_fields or any(field.wire_type != 2 for field in payload_fields):
        raise BagFormatError(f'{path}: missing outer record payloads')
    for index, payload_field in enumerate(payload_fields):
        assert isinstance(payload_field.value, bytes)
        inner = parse_fields(payload_field.value)
        application = [field for field in inner if field.number == 2]
        if (len(application) != 1 or application[0].wire_type != 2):
            raise BagFormatError(
                f'{path}: unexpected inner envelope at record {index}')
        assert isinstance(application[0].value, bytes)
        yield application[0].value


def read_index(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 8:
        raise BagFormatError(f'{path}: index size is not divisible by 8')
    return [value[0] for value in struct.iter_unpack('<Q', data)]


def unpack_doubles(value: int | bytes, expected: int | None = None) -> list[float]:
    if not isinstance(value, bytes) or len(value) % 8:
        raise BagFormatError('expected a packed double field')
    count = len(value) // 8
    if expected is not None and count != expected:
        raise BagFormatError(f'expected {expected} doubles, got {count}')
    return list(struct.unpack(f'<{count}d', value))


def integer(fields: dict[int, int | bytes], number: int, default: int = 0) -> int:
    value = fields.get(number, default)
    if not isinstance(value, int):
        raise BagFormatError(f'field {number} is not an integer')
    return value


def bytes_value(fields: dict[int, int | bytes], number: int) -> bytes:
    value = fields.get(number)
    if not isinstance(value, bytes):
        raise BagFormatError(f'field {number} is not bytes')
    return value
