#define PCL_NO_PRECOMPILE

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pcl/compression/octree_pointcloud_compression.h>
#include <pcl/io/impl/octree_pointcloud_compression.hpp>
#include <pcl/point_types.h>

namespace fs = std::filesystem;

namespace {

struct Options {
  fs::path bag;
  fs::path output;
  std::size_t limit = 0;
  bool overwrite = false;
};

std::uint64_t read_varint(const std::string& data, std::size_t& pos) {
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 70 && pos < data.size(); shift += 7) {
    const auto byte = static_cast<unsigned char>(data[pos++]);
    value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) return value;
  }
  throw std::runtime_error("truncated or oversized protobuf varint");
}

struct Field {
  unsigned number;
  unsigned wire;
  std::uint64_t integer = 0;
  std::string bytes;
};

Field read_field(const std::string& data, std::size_t& pos) {
  const std::uint64_t key = read_varint(data, pos);
  Field field{static_cast<unsigned>(key >> 3), static_cast<unsigned>(key & 7)};
  if (field.number == 0) throw std::runtime_error("invalid protobuf field 0");
  if (field.wire == 0) {
    field.integer = read_varint(data, pos);
  } else if (field.wire == 1) {
    if (data.size() - pos < 8) throw std::runtime_error("truncated fixed64");
    field.bytes.assign(data, pos, 8);
    pos += 8;
  } else if (field.wire == 2) {
    const auto size = read_varint(data, pos);
    if (size > data.size() - pos) throw std::runtime_error("truncated bytes field");
    field.bytes.assign(data, pos, static_cast<std::size_t>(size));
    pos += size;
  } else if (field.wire == 5) {
    if (data.size() - pos < 4) throw std::runtime_error("truncated fixed32");
    field.bytes.assign(data, pos, 4);
    pos += 4;
  } else {
    throw std::runtime_error("unsupported protobuf wire type");
  }
  return field;
}

std::string unwrap(const std::string& outer, std::size_t& pos) {
  Field outer_payload = read_field(outer, pos);
  if (outer_payload.number == 1 && outer_payload.wire == 2) {
    outer_payload = read_field(outer, pos);  // field 1 is chunk-level metadata
  }
  if (outer_payload.number != 2 || outer_payload.wire != 2) {
    throw std::runtime_error("unexpected outer recorder envelope");
  }
  std::size_t inner_pos = 0;
  std::string application;
  while (inner_pos < outer_payload.bytes.size()) {
    const Field field = read_field(outer_payload.bytes, inner_pos);
    if (field.number == 2 && field.wire == 2) application = field.bytes;
  }
  if (application.empty()) {
    throw std::runtime_error("unexpected inner recorder envelope");
  }
  return application;
}

struct PointCloudMessage {
  std::uint64_t sensor_timestamp_ns = 0;
  std::uint64_t rows = 0;
  std::string compressed;
};

PointCloudMessage parse_point_cloud(const std::string& message) {
  PointCloudMessage result;
  std::size_t pos = 0;
  while (pos < message.size()) {
    const Field field = read_field(message, pos);
    if (field.number == 1 && field.wire == 0) result.sensor_timestamp_ns = field.integer;
    if (field.number == 2 && field.wire == 0) result.rows = field.integer;
    if (field.number == 3 && field.wire == 2) result.compressed = field.bytes;
  }
  if (result.compressed.empty()) throw std::runtime_error("point cloud has no data field");
  return result;
}

std::vector<std::uint64_t> read_index(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  stream.seekg(0, std::ios::end);
  const auto size = stream.tellg();
  if (size < 0 || size % 8 != 0) throw std::runtime_error("invalid index " + path.string());
  stream.seekg(0);
  std::vector<std::uint64_t> values(static_cast<std::size_t>(size) / 8);
  stream.read(reinterpret_cast<char*>(values.data()), size);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (auto& value : values) value = __builtin_bswap64(value);
#endif
  return values;
}

std::string read_file(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  return std::string(std::istreambuf_iterator<char>(stream), {});
}

Options parse_options(int argc, char** argv) {
  if (argc < 3) {
    throw std::runtime_error(
        "usage: decode_lidar BAG OUTPUT [--limit N] [--overwrite]");
  }
  Options options{argv[1], argv[2]};
  for (int index = 3; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--limit" && index + 1 < argc) {
      options.limit = std::stoull(argv[++index]);
    } else if (argument == "--overwrite") {
      options.overwrite = true;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const fs::path source = options.bag / "ouster__points";
    const fs::path cloud_dir = options.output / "lidar";
    fs::create_directories(cloud_dir);

    std::vector<fs::path> chunks;
    for (const auto& entry : fs::directory_iterator(source)) {
      if (entry.path().extension() == ".bin") chunks.push_back(entry.path());
    }
    std::sort(chunks.begin(), chunks.end());
    if (chunks.empty()) throw std::runtime_error("no LiDAR chunks found in " + source.string());

    std::ofstream manifest(options.output / "lidar_manifest.csv");
    manifest << "timestamp_ns,sensor_timestamp_ns,path,chunk,scan_in_chunk,points\n";
    pcl::io::OctreePointCloudCompression<pcl::PointXYZ> decoder;
    std::size_t total = 0;

    for (const fs::path& chunk : chunks) {
      const auto timestamps = read_index(chunk.parent_path() / (chunk.stem().string() + ".idx"));
      const std::string data = read_file(chunk);
      std::size_t pos = 0;
      std::size_t scan = 0;
      while (pos < data.size()) {
        if (scan >= timestamps.size()) {
          throw std::runtime_error(chunk.string() + ": more records than index entries");
        }
        const PointCloudMessage message = parse_point_cloud(unwrap(data, pos));
        std::stringstream compressed(message.compressed);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        decoder.decodePointCloud(compressed, cloud);
        if (message.rows && cloud->size() != message.rows) {
          throw std::runtime_error(chunk.string() + ": decoded point count mismatch");
        }

        const auto timestamp = timestamps[scan];
        const fs::path relative = fs::path("lidar") / (std::to_string(timestamp) + ".bin");
        const fs::path destination = options.output / relative;
        if (fs::exists(destination) && !options.overwrite) {
          throw std::runtime_error(destination.string() + " exists; use --overwrite");
        }
        std::ofstream output(destination, std::ios::binary);
        for (const auto& point : cloud->points) {
          const float xyzi[4] = {point.x, point.y, point.z, 0.0f};
          output.write(reinterpret_cast<const char*>(xyzi), sizeof(xyzi));
        }
        manifest << timestamp << ',' << message.sensor_timestamp_ns << ','
                 << relative.generic_string() << ',' << chunk.filename().string() << ','
                 << scan << ',' << cloud->size() << '\n';
        ++scan;
        ++total;
        if (options.limit && total >= options.limit) {
          std::cout << "wrote " << total << " point clouds\n";
          return 0;
        }
      }
      if (scan != timestamps.size()) {
        throw std::runtime_error(chunk.string() + ": record/index count mismatch (" +
                                 std::to_string(scan) + " vs " +
                                 std::to_string(timestamps.size()) + ")");
      }
    }
    std::cout << "wrote " << total << " point clouds\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "decode_lidar: " << error.what() << '\n';
    return 1;
  }
}
