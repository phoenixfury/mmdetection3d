#pragma once

// Standalone, dependency-free XYZI serialization. Kept separate from PCL so
// both the fast C++ writer and its regression tests use identical code.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace recorder_pipeline {

inline void encode_float_le(char* destination, float value) {
  static_assert(sizeof(float) == 4, "float must be IEEE-754 float32");
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  destination[0] = static_cast<char>(bits & 0xffU);
  destination[1] = static_cast<char>((bits >> 8) & 0xffU);
  destination[2] = static_cast<char>((bits >> 16) & 0xffU);
  destination[3] = static_cast<char>((bits >> 24) & 0xffU);
}

template <typename Cloud>
void write_xyzi_payload(std::ostream& output, const Cloud& cloud) {
  constexpr std::size_t kBatchPoints = 16'384;
  constexpr std::size_t kStride = 16;
  std::vector<char> encoded(kBatchPoints * kStride);
  for (std::size_t begin = 0; begin < cloud.points.size(); begin += kBatchPoints) {
    const std::size_t count = std::min(kBatchPoints, cloud.points.size() - begin);
    for (std::size_t i = 0; i < count; ++i) {
      const auto& point = cloud.points[begin + i];
      char* record = encoded.data() + i * kStride;
      encode_float_le(record, point.x);
      encode_float_le(record + 4, point.y);
      encode_float_le(record + 8, point.z);
      encode_float_le(record + 12, 0.0f); // PointXYZ carries no intensity
    }
    output.write(encoded.data(), static_cast<std::streamsize>(count * kStride));
    if (!output) throw std::runtime_error("failed to write batched XYZI payload");
  }
}

} // namespace recorder_pipeline
