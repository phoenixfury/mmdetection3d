# v0.4.2 Protobuf / Abseil build fix

Observed: `/usr/local/lib/libprotobuf.a` (Protobuf 6.30.1) generates dozens of
undefined `absl::lts_20250127::*` symbols when the executable links. This is
an incomplete or incorrectly consumed static Protobuf installation, not an
error in the point-cloud serializer.

Changes:

* Default to coherent distro Protobuf (`libprotobuf.so`, headers, and protoc).
* Reject incomplete distro toolchains with a direct installation suggestion.
* Force replacement of stale cached `/usr/local` `FindProtobuf` results.
* Add a private include overlay because GCC searches `/usr/local/include`
  before `/usr/include` even when the latter is selected with FindProtobuf.
* Handle proto3 `optional` annotations for Ubuntu protoc 3.12--3.14.
* Provide opt-in `config` mode for a custom CMake-packaged Protobuf+Abseil.
* Retain original decoders, CLI, Python tests and conversion behavior.

Target testing still required: the full PCL decoder against actual recordings
on Ubuntu 22.04. The release cannot be claimed to compile with a particular
local Protobuf installation without checking it on that machine.
