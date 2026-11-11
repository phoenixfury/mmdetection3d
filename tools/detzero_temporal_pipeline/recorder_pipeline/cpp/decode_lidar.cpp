#define PCL_NO_PRECOMPILE

// Only the PCL-octree variant belongs here. Uncompressed PointCloud.data is
// handled by the Python raw backend with explicit point layout validation.
#include <pcl/compression/octree_pointcloud_compression.h>
#include <pcl/io/impl/octree_pointcloud_compression.hpp>
#include <pcl/point_types.h>
#include <bag.pb.h>
#include <lidar.pb.h>
#include "xyzi_writer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
namespace {
struct Options {
  fs::path bag;
  fs::path output;
  size_t limit = 0;
  bool overwrite = false;
  size_t workers = 1;
};
Options parse_options(int argc, char** argv) {
  if (argc < 3) throw std::runtime_error("usage: decode_lidar BAG OUTPUT [--limit N] [--workers N] [--overwrite]");
  Options result{argv[1], argv[2]};
  for (int i = 3; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--overwrite") result.overwrite = true;
    else if (flag == "--limit" && i + 1 < argc) {
      result.limit = std::stoull(argv[++i]);
    } else if (flag == "--workers" && i + 1 < argc) {
      result.workers = std::stoull(argv[++i]);
      if (result.workers == 0) throw std::runtime_error("--workers must be positive");
    } else throw std::runtime_error("unsupported argument: " + flag);
  }
  return result;
}
std::string read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::vector<uint64_t> read_index(const fs::path& path) {
  const auto bytes = read_file(path);
  if (bytes.size() % 8 != 0) throw std::runtime_error("invalid index: " + path.string());
  std::vector<uint64_t> values;
  values.reserve(bytes.size() / 8);
  for (size_t i = 0; i < bytes.size(); i += 8) {
    uint64_t n = 0;
    for (unsigned k = 0; k < 8; ++k)
      n |= uint64_t(static_cast<uint8_t>(bytes[i + k])) << (8 * k);
    values.push_back(n);
  }
  return values;
}
void write_xyzi(const fs::path& destination, const pcl::PointCloud<pcl::PointXYZ>& cloud,
                bool overwrite) {
  if (fs::exists(destination) && !overwrite)
    throw std::runtime_error("output already exists: " + destination.string());
  fs::path tmp = destination;
  tmp += ".tmp";
  try {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create " + tmp.string());
    recorder_pipeline::write_xyzi_payload(out, cloud);
    out.close();
    if (!out) throw std::runtime_error("write failed: " + tmp.string());
    // Ubuntu/POSIX rename atomically replaces an existing destination.
    // No delete-before-rename window (unlike the previous implementation).
    fs::rename(tmp, destination);
  } catch (...) {
    fs::remove(tmp);
    throw;
  }
}
struct DecodedRow {
  uint64_t timestamp;
  uint64_t sensor_timestamp;
  std::string chunk;
  size_t scan;
  size_t points;
};

std::vector<DecodedRow> decode_chunk(const fs::path& chunk, const fs::path& output,
                                     const std::vector<uint64_t>& timestamps,
                                     bool overwrite, size_t max_samples = 0) {
  bag::TopicData topic;
  if (!topic.ParseFromString(read_file(chunk)))
    throw std::runtime_error(chunk.string() + ": invalid bag.TopicData protobuf");
  if (static_cast<size_t>(topic.samples_size()) != timestamps.size())
    throw std::runtime_error(chunk.string() + ": record/index count mismatch");
  pcl::io::OctreePointCloudCompression<pcl::PointXYZ> decoder;
  std::vector<DecodedRow> rows;
  const size_t count = max_samples ? std::min(max_samples, timestamps.size()) : timestamps.size();
  rows.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const auto timestamp = timestamps[static_cast<size_t>(i)];
    PointCloud message;
    if (!message.ParseFromString(topic.samples(static_cast<int>(i)).data()) || message.rows() <= 0 || message.data().empty())
      throw std::runtime_error(chunk.string() + ": invalid PointCloud sample " + std::to_string(i));
    std::istringstream encoded(message.data(), std::ios::in | std::ios::binary);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    decoder.decodePointCloud(encoded, cloud);
    if (cloud->size() != static_cast<size_t>(message.rows()))
      throw std::runtime_error(chunk.string() + ": decoded point count differs from PointCloud.rows; check encoding");
    const fs::path filename = std::to_string(timestamp) + ".bin";
    write_xyzi(output / "lidar" / filename, *cloud, overwrite);
    rows.push_back({timestamp, message.timestamp(), chunk.filename().string(),
                    static_cast<size_t>(i), cloud->size()});
  }
  return rows;
}
} // namespace

int main(int argc, char** argv) {
  fs::path temp_manifest;
  try {
    const auto opt = parse_options(argc, argv);
    const fs::path source = opt.bag / "ouster__points";
    if (!fs::is_directory(source)) throw std::runtime_error("missing topic: " + source.string());
    fs::create_directories(opt.output / "lidar");
    std::vector<fs::path> chunks;
    for (const auto& file : fs::directory_iterator(source))
      if (file.is_regular_file() && file.path().extension() == ".bin")
        chunks.push_back(file.path());
    std::sort(chunks.begin(), chunks.end());
    if (chunks.empty()) throw std::runtime_error("no *.bin chunks in " + source.string());
    const fs::path manifest = opt.output / "lidar_manifest.csv";
    if (fs::exists(manifest) && !opt.overwrite)
      throw std::runtime_error("manifest exists; use --overwrite: " + manifest.string());
    temp_manifest = opt.output / "lidar_manifest.tmp.csv";
    std::ofstream csv(temp_manifest, std::ios::trunc);
    if (!csv) throw std::runtime_error("cannot write " + temp_manifest.string());
    csv << "timestamp_ns,sensor_timestamp_ns,path,chunk,scan_in_chunk,points,encoding,intensity_source\n";

    std::vector<std::vector<uint64_t>> indexes;
    std::vector<size_t> chunk_limits;
    size_t selected = 0;
    for (const auto& chunk : chunks) {
      auto index = read_index(chunk.parent_path() / (chunk.stem().string() + ".idx"));
      const size_t take = opt.limit ? std::min(index.size(), opt.limit - selected) : index.size();
      selected += take;
      indexes.push_back(std::move(index));
      chunk_limits.push_back(opt.limit ? take : 0);
      if (opt.limit && selected >= opt.limit) break;
    }
    chunks.resize(indexes.size());
    size_t total = 0;
    std::unordered_set<uint64_t> seen_timestamps;
    std::vector<std::future<std::vector<DecodedRow>>> pending;
    std::vector<DecodedRow> rows;
    for (size_t i = 0; i < chunks.size(); ++i) {
      pending.push_back(std::async(std::launch::async, decode_chunk,
                                   chunks[i], opt.output, indexes[i], opt.overwrite,
                                   chunk_limits[i]));
      if (pending.size() >= opt.workers || i + 1 == chunks.size()) {
        for (auto& job : pending) {
          auto decoded = job.get();
          rows.insert(rows.end(), decoded.begin(), decoded.end());
        }
        pending.clear();
      }
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
      return a.timestamp < b.timestamp;
    });
    for (const auto& row : rows) {
      if (!seen_timestamps.insert(row.timestamp).second)
        throw std::runtime_error("duplicate LiDAR index timestamp: " + std::to_string(row.timestamp));
      const auto filename = std::to_string(row.timestamp) + ".bin";
      csv << row.timestamp << ',' << row.sensor_timestamp << ",lidar/" << filename
          << ',' << row.chunk << ',' << row.scan << ',' << row.points
          << ",pcl,zeros\n";
      ++total;
    }
    csv.close();
    if (!csv) throw std::runtime_error("failed writing manifest");
    if (fs::exists(manifest)) fs::remove(manifest);
    fs::rename(temp_manifest, manifest);
    std::cout << "Decoded " << total << " PCL-compressed point clouds\n";
    return 0;
  } catch (const std::exception& error) {
    if (!temp_manifest.empty()) fs::remove(temp_manifest);
    std::cerr << "decode_lidar: " << error.what() << '\n';
    return 1;
  }
}
