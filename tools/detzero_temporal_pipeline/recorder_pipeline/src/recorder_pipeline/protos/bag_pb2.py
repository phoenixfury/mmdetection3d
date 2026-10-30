# Generated-style runtime descriptors for proto/bag.proto.
# Regenerate using: protoc -Iproto --python_out=src/recorder_pipeline/protos proto/bag.proto
from google.protobuf import timestamp_pb2 as _timestamp_pb2  # required descriptor registration
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf.internal import builder as _builder
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\tbag.proto\x12\x03bag\x1a\x1fgoogle/protobuf/timestamp.proto"y\n\x06Sample\x12-\n\ttimestamp\x18\x01 \x01(\x0b2\x1a.google.protobuf.Timestamp\x122\n\x0eread_timestamp\x18\x03 \x01(\x0b2\x1a.google.protobuf.Timestamp\x12\x0c\n\x04data\x18\x02 \x01(\x0c"X\n\tTopicData\x12-\n\ttimestamp\x18\x01 \x01(\x0b2\x1a.google.protobuf.Timestamp\x12\x1c\n\x07samples\x18\x02 \x03(\x0b2\x0b.bag.Sampleb\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'recorder_pipeline.protos.bag_pb2', globals())
