"""Raw odometry adapter using the supplied adas.common.EgoPose schema.

Never guess the wire layout of adas.common.EgoPose: serialized protobufs do
not carry field names or types. The caller supplies an exact descriptor set.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from .bag_io import BagFormatError, iter_bag_topic_samples
from .odometry import timestamp_ns, TIME_UNIT_TO_NS, FIELDS, deduplicate_samples


def load_message_class(proto_dir: Path = None, message_name: str = 'adas.common.EgoPose'):
    proto_dir = (Path(proto_dir) if proto_dir is not None else Path(__file__).resolve().parent / 'schemas').resolve()
    source = proto_dir / 'common.proto'
    if not source.is_file():
        raise FileNotFoundError(f'Raw odometry requires the real {source}; cannot safely infer its protobuf fields')
    with tempfile.TemporaryDirectory(prefix='recorder-descriptors-') as temp:
        destination = Path(temp) / 'schema.desc'
        command = ['protoc', '-I', str(proto_dir), '--include_imports',
                   '--descriptor_set_out=' + str(destination), str(source)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f'cannot compile {source} with protoc: {exc}') from exc
        descriptor_set = descriptor_pb2.FileDescriptorSet.FromString(destination.read_bytes())
    pool = descriptor_pool.DescriptorPool()
    pending = list(descriptor_set.file)
    while pending:
        remaining = []
        for item in pending:
            try:
                pool.AddSerializedFile(item.SerializeToString())
            except TypeError:
                remaining.append(item)
        if len(remaining) == len(pending):
            raise BagFormatError('unresolved dependencies in common.proto descriptor set')
        pending = remaining
    try:
        descriptor = pool.FindMessageTypeByName(message_name)
    except KeyError as error:
        raise BagFormatError(f'{message_name} not found in {source}') from error
    try:
        return message_factory.GetMessageClass(descriptor)
    except AttributeError:  # older protobuf Python runtime
        return message_factory.MessageFactory(pool).GetPrototype(descriptor)


def load_raw_odometry(folder: Path, proto_dir: Path = None, timestamp_unit: str = 'ms') -> list[dict]:
    """Decode the previously observed EgoPose.pose.position and scalar heading.

    Fail explicitly if the supplied schema does not match those fields. No
    guessed velocities or roll/pitch measurements are introduced.
    """
    if timestamp_unit not in TIME_UNIT_TO_NS:
        raise ValueError('invalid timestamp unit')
    cls = load_message_class(proto_dir)
    descriptor = cls.DESCRIPTOR
    fields = {field.name for field in descriptor.fields}
    if not {'timestamp', 'pose'}.issubset(fields):
        raise BagFormatError('EgoPose must contain timestamp and pose fields')
    pose_type = descriptor.fields_by_name['pose'].message_type
    if pose_type is None or not {'position', 'orientation'}.issubset(pose_type.fields_by_name):
        raise BagFormatError('EgoPose.pose must contain position and scalar orientation')
    orientation_field = pose_type.fields_by_name['orientation']
    if orientation_field.message_type is not None:
        raise BagFormatError('EgoPose.pose.orientation is not a scalar yaw; review common.proto')
    rows = []
    for write_time, sample, chunk, index in iter_bag_topic_samples(folder):
        pose = cls()
        try:
            pose.ParseFromString(sample.data)
        except Exception as error:
            raise BagFormatError(f'{chunk} raw sample {index}: {error}') from error
        if not pose.HasField('pose') or not pose.pose.HasField('position'):
            raise BagFormatError(f'{chunk} raw sample {index}: missing pose.position')
        if int(pose.status) not in (1, 2):
            # kNotAvailable=0 does not represent a trustworthy pose.
            continue
        position = pose.pose.position
        row = dict(measurement_ns=timestamp_ns(int(pose.timestamp), timestamp_unit),
                   message_timestamp=int(pose.timestamp), recorder_index_ns=write_time,
                   read_timestamp_ns=(sample.read_timestamp.seconds * 1_000_000_000 +
                                      sample.read_timestamp.nanos if sample.HasField('read_timestamp') else None),
                   x=float(position.x), y=float(position.y), z=float(getattr(position, 'z', 0.)),
                   roll=0., pitch=0., yaw=float(pose.pose.orientation),
                   vx=0., vy=0., vz=0., yaw_rate=0., status=int(getattr(pose, 'status', 0)))
        if not all(math.isfinite(row[key]) for key in FIELDS):
            raise BagFormatError(f'{chunk}: non-finite raw pose component at sample {index}')
        rows.append(row)
    if not rows:
        raise BagFormatError(f'{folder}: no available raw odometry poses (statuses kAvailable/kAvailableReduced)')
    return deduplicate_samples(rows, 'raw')
