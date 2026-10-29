#include "xyzi_writer.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

struct Point { float x,y,z; };
struct Cloud { std::vector<Point> points; };

int main() {
  Cloud cloud{{{1.0f, 2.0f, 3.0f}, {-0.0f, -2.5f, 0.125f}}};
  std::ostringstream output(std::ios::out | std::ios::binary);
  recorder_pipeline::write_xyzi_payload(output, cloud);
  const std::string actual = output.str();
  assert(actual.size() == 32);
  const std::uint8_t expected[] = {
    0,0,128,63, 0,0,0,64, 0,0,64,64, 0,0,0,0,
    0,0,0,128, 0,0,32,192, 0,0,0,62, 0,0,0,0
  };
  assert(std::memcmp(actual.data(), expected, sizeof(expected)) == 0);
  // Test writing across multiple 16,384-point boundaries.
  Cloud large;
  large.points.resize(33000, {1.f, 2.f, 3.f});
  std::ostringstream many(std::ios::out | std::ios::binary);
  recorder_pipeline::write_xyzi_payload(many, large);
  assert(many.str().size() == large.points.size() * 16);
  const std::string many_bytes = many.str();
  for (std::size_t offset = 0; offset < many_bytes.size(); offset += 16) {
    assert(std::memcmp(many_bytes.data() + offset, expected, 16) == 0);
  }
}
