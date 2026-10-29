# Generated-style protobuf runtime module. Source: proto/lidar.proto
# Regenerate with: protoc -Iproto --python_out=src/recorder_pipeline/protos proto/lidar.proto
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf.internal import builder as _builder
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0blidar.proto\x12\x00";\n\nPointCloud\x12\x11\n\ttimestamp\x18\x01 \x01(\x04\x12\x0c\n\x04rows\x18\x02 \x01(\x05\x12\x0c\n\x04data\x18\x03 \x01(\x0c"\'\n\x0bLidarOutput\x12\x18\n\x05boxes\x18\x01 \x03(\x0b2\t.LidarBox"\xd6\x01\n\x08LidarBox\x12\x13\n\x06_class\x18\x01 \x01(\rH\x00\x88\x01\x01\x12\r\n\x05score\x18\x02 \x01(\x02\x12\t\n\x01x\x18\x03 \x01(\x02\x12\t\n\x01y\x18\x04 \x01(\x02\x12\x0e\n\x01z\x18\x05 \x01(\x02H\x01\x88\x01\x01\x12\x0e\n\x06x_size\x18\x06 \x01(\x02\x12\x0e\n\x06y_size\x18\x07 \x01(\x02\x12\x13\n\x06z_size\x18\x08 \x01(\x02H\x02\x88\x01\x01\x12\x0b\n\x03yaw\x18\t \x01(\x02\x12\x15\n\x08track_id\x18\n \x01(\x04H\x03\x88\x01\x01B\t\n\x07__classB\x04\n\x02_zB\t\n\x07_z_sizeB\x0b\n\t_track_idb\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'recorder_pipeline.protos.lidar_pb2', globals())
