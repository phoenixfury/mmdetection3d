"""Bounded protobuf-wire reader for the proprietary recorder envelopes.

Only the recorder envelope is decoded by hand: application messages are parsed
with the generated protobuf classes in ``recorder_pipeline.protos``.
"""
from __future__ import annotations
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


class BagFormatError(ValueError):
    """Corrupt/malformed recorder data, protobuf wire data, or index."""


@dataclass(frozen=True)
class Field:
    number: int
    wire_type: int
    value: int | bytes


def _varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for i in range(10):
        if offset >= len(data):
            raise BagFormatError('truncated protobuf varint')
        byte = data[offset]
        offset += 1
        if i == 9 and byte > 1:
            raise BagFormatError('protobuf varint exceeds uint64')
        value |= (byte & 127) << (7 * i)
        if not byte & 128:
            return value, offset
    raise BagFormatError('protobuf varint exceeds 10 bytes')


def parse_fields(data: bytes) -> list[Field]:
    fields: list[Field] = []
    offset = 0
    while offset < len(data):
        key, offset = _varint(data, offset)
        number, wire = key >> 3, key & 7
        if number == 0 or number > (1 << 29) - 1:
            raise BagFormatError(f'invalid protobuf field number {number}')
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, offset = _varint(data, offset)
            else:
                size = 8 if wire == 1 else 4
            if size > len(data) - offset:
                raise BagFormatError('truncated protobuf bytes')
            value = data[offset:offset + size]
            offset += size
        else:
            raise BagFormatError(f'unsupported protobuf wire type {wire}')
        fields.append(Field(number, wire, value))
    return fields


def field_map(data: bytes) -> dict[int, int | bytes]:
    return {field.number: field.value for field in parse_fields(data)}


def repeated_bytes(data: bytes, number: int) -> list[bytes]:
    return [field.value for field in parse_fields(data)
            if field.number == number and field.wire_type == 2]


def integer(fields: dict[int, int | bytes], number: int, default: int = 0) -> int:
    value = fields.get(number, default)
    if not isinstance(value, int):
        raise BagFormatError(f'field {number}: expected integer')
    return value


def bytes_value(fields: dict[int, int | bytes], number: int) -> bytes:
    value = fields.get(number)
    if not isinstance(value, bytes):
        raise BagFormatError(f'field {number}: expected bytes')
    return value


def unpack_doubles(value: int | bytes, expected: int | None = None) -> list[float]:
    if not isinstance(value, bytes) or len(value) % 8:
        raise BagFormatError('expected packed double field')
    count = len(value) // 8
    if expected is not None and count != expected:
        raise BagFormatError(f'expected {expected} doubles, got {count}')
    return list(struct.unpack(f'<{count}d', value))


def _file_varint(stream: BinaryIO, first: int | None = None) -> int:
    value = 0
    for i in range(10):
        b = (bytes([first]) if i == 0 and first is not None else stream.read(1))
        if not b:
            raise BagFormatError('truncated record varint')
        byte = b[0]
        if i == 9 and byte > 1:
            raise BagFormatError('record varint exceeds uint64')
        value |= (byte & 127) << (i * 7)
        if byte < 128:
            return value
    raise BagFormatError('record varint longer than 10 bytes')


def _read_field_from_file(stream: BinaryIO, first: int) -> Field:
    key = _file_varint(stream, first)
    number, wire = key >> 3, key & 7
    if number == 0 or number > (1 << 29) - 1:
        raise BagFormatError(f'invalid recorder field number {number}')
    if wire == 0:
        return Field(number, wire, _file_varint(stream))
    if wire not in (1, 2, 5):
        raise BagFormatError(f'unsupported recorder wire type {wire}')
    size = _file_varint(stream) if wire == 2 else (8 if wire == 1 else 4)
    if size > 2 * 1024**3:
        raise BagFormatError(f'unreasonably large recorder field: {size} bytes')
    payload = stream.read(size)
    if len(payload) != size:
        raise BagFormatError('truncated recorder field payload')
    return Field(number, wire, payload)


def iter_record_payloads(path: Path) -> Iterator[bytes]:
    """Yield app payloads for repeated outer tag 2, one record at a time.

    Outer field 1 may be chunk metadata; inner field 2 is the application
    protobuf bytes. Intentionally *no* bag.TopicData schema is assumed.
    """
    count = 0
    with path.open('rb') as stream:
        while (first := stream.read(1)):
            outer = _read_field_from_file(stream, first[0])
            if outer.number == 1:
                continue
            if outer.number != 2 or outer.wire_type != 2:
                raise BagFormatError(f'{path}: unexpected outer field {outer.number}/{outer.wire_type}')
            fields = parse_fields(outer.value)
            applications = [field.value for field in fields
                            if field.number == 2 and field.wire_type == 2]
            if len(applications) != 1:
                raise BagFormatError(f'{path}: expected one application payload in record {count}')
            count += 1
            yield applications[0]
    if count == 0:
        raise BagFormatError(f'{path}: no recorder messages')


def read_index(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 8:
        raise BagFormatError(f'{path}: .idx byte count is not divisible by 8')
    return [value for (value,) in struct.iter_unpack('<Q', raw)]


def iter_topic_samples(folder: Path) -> Iterator[tuple[int, bytes, Path, int]]:
    """Yield (write_timestamp_ns, application_bytes, chunk_file, sample_index).

    Reject missing, extra and duplicate index entries rather than silently
    associating sensor payloads with the wrong timestamp.
    """
    chunks = sorted(folder.glob('*.bin'))
    if not chunks:
        raise FileNotFoundError(f'no topic chunks in {folder}')
    for chunk in chunks:
        timestamps = read_index(chunk.with_suffix('.idx'))
        count = 0
        for count, payload in enumerate(iter_record_payloads(chunk), start=1):
            if count > len(timestamps):
                raise BagFormatError(f'{chunk}: more recorder messages than .idx entries')
            yield timestamps[count - 1], payload, chunk, count - 1
        if count != len(timestamps):
            raise BagFormatError(f'{chunk}: {count} messages, {len(timestamps)} timestamps')


def iter_bag_topic_samples(folder: Path):
    """Stream typed ``bag.TopicData.samples`` without loading an entire chunk.

    The file is a single protobuf message whose repeated field 2 contains
    length-delimited ``bag.Sample`` messages. Protobuf allows unknown fields;
    they are skipped safely. The .idx association is still verified exactly.
    Import the protobuf class lazily to keep the legacy calibration parser lean.
    """
    from .protos import bag_pb2

    chunks = sorted(folder.glob('*.bin'))
    if not chunks:
        raise FileNotFoundError(f'no topic chunks in {folder}')
    for chunk in chunks:
        timestamps = read_index(chunk.with_suffix('.idx'))
        count = 0
        with chunk.open('rb') as stream:
            while first := stream.read(1):
                field = _read_field_from_file(stream, first[0])
                if field.number != 2:
                    continue
                if field.wire_type != 2:
                    raise BagFormatError(f'{chunk}: TopicData.samples is not length-delimited')
                if count >= len(timestamps):
                    raise BagFormatError(f'{chunk}: more TopicData samples than .idx entries')
                sample = bag_pb2.Sample()
                try:
                    sample.ParseFromString(field.value)
                except Exception as error:
                    raise BagFormatError(f'{chunk}: invalid bag.Sample #{count}: {error}') from error
                timestamp = timestamps[count]
                index = count
                count += 1
                yield timestamp, sample, chunk, index
        if count != len(timestamps):
            raise BagFormatError(f'{chunk}: {count} TopicData samples, {len(timestamps)} .idx entries')
