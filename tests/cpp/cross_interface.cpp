// The C++ side of tests/python/test_cross_interface.py and test_rule_table.py: reads the
// cases that the tests write (cases.txt and raw little-endian buffers in one directory) and
// writes the values (float64) and gradients (the storage type) of two C++ interfaces over the
// same kernels:
//
//   <id>.cpp_kernel.{values,grads,status}  sdca::kernel::evaluate on the whole batch, with the
//       class weights, ignore_index, label smoothing, upstream factor and check_rows of the
//       case; status: "status invalid_rows first_invalid_row". The outputs start as bytes
//       0xAB and are written whatever the status.
//   <id>.cpp_loss.{values,grads} or .error  sdca/loss.h, one row at a time (float and double
//       cases without those options, of labels, label sets or teacher logits, in the
//       storage's precision); error: the message of the first row that throws.
//
// Usage: cross_interface DIR. A line of cases.txt:
//   id key kind k gamma temperature label_smoothing ignore ignore_index dtype n classes
//   target weight threads precision upstream check_rows
// dtype: float32, float64, float16 or bfloat16; target: labels, sets, mask, probs or teacher;
// weight: 0 or 1; threads: 0 (the default) or a count; precision: 0 (the storage's) or 2
// (float64); numbers as Python's repr (nan, inf). The inputs, shared by the cases of one key:
// <key>.scores and <key>.teacher (dtype), <key>.labels, <key>.set_labels and <key>.offsets
// (int64), <key>.mask (uint8), <key>.probs and <key>.weight (float64).

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sdca/kernel.h"
#include "sdca/loss.h"

namespace {

struct Case {
  std::string id, key;
  int kind = 0;
  std::size_t k = 1;
  double gamma = 0, temperature = 1, label_smoothing = 0;
  int ignore = 0;
  int64_t ignore_index = 0;
  std::string dtype;
  std::size_t n = 0, classes = 0;
  std::string target;
  int weight = 0;
  std::size_t threads = 0;
  int precision = 0;
  double upstream = 1;
  int check_rows = 0;
};

std::string dir;

std::vector<char> read_bytes(const std::string& name) {
  std::ifstream in(dir + "/" + name, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + name);
  return std::vector<char>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

template <typename T>
std::vector<T> read(const std::string& name) {
  const std::vector<char> bytes = read_bytes(name);
  std::vector<T> out(bytes.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
  return out;
}

template <typename T>
void write(const std::string& name, const std::vector<T>& data) {
  std::ofstream out(dir + "/" + name, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(T)));
  if (!out) throw std::runtime_error("cannot write " + name);
}

template <typename S, typename Targets>
void batch(const Case& c, const std::vector<S>& z, const Targets& t) {
  const sdca::kernel::Loss loss = {c.kind, c.k, c.gamma, c.temperature, c.label_smoothing};
  const sdca::kernel::Execution e = {c.precision, c.threads, c.check_rows != 0};
  std::vector<double> values(c.n);
  std::vector<S> grads(c.n * c.classes);
  std::memset(values.data(), 0xAB, values.size() * sizeof(double));
  std::memset(static_cast<void*>(grads.data()), 0xAB, grads.size() * sizeof(S));
  const sdca::kernel::Result r = sdca::kernel::evaluate(loss, e, c.n, c.classes, z.data(), t,
                                                        values.data(), grads.data(), c.upstream);
  write(c.id + ".cpp_kernel.values", values);
  write(c.id + ".cpp_kernel.grads", grads);
  std::ofstream status(dir + "/" + c.id + ".cpp_kernel.status");
  status << r.status << " " << r.invalid_rows << " " << r.first_invalid << "\n";
}

template <typename S>
void run_kernel(const Case& c) {
  const std::vector<S> z = read<S>(c.key + ".scores");
  std::vector<double> weight;
  if (c.weight) weight = read<double>(c.key + ".weight");
  const double* w = c.weight ? weight.data() : nullptr;
  if (c.target == "labels") {
    const std::vector<int64_t> labels = read<int64_t>(c.key + ".labels");
    const sdca::kernel::Labels t = {labels.data(), w, c.ignore_index, c.ignore != 0};
    batch(c, z, t);
  } else if (c.target == "sets") {
    const std::vector<int64_t> labels = read<int64_t>(c.key + ".set_labels");
    const std::vector<int64_t> offsets = read<int64_t>(c.key + ".offsets");
    const sdca::kernel::LabelSets t = {labels.data(), offsets.data()};
    batch(c, z, t);
  } else if (c.target == "mask") {
    const std::vector<uint8_t> mask = read<uint8_t>(c.key + ".mask");
    const sdca::kernel::Mask t = {mask.data()};
    batch(c, z, t);
  } else if (c.target == "probs") {
    const std::vector<double> p = read<double>(c.key + ".probs");
    const sdca::kernel::Probabilities<double> t = {p.data(), w};
    batch(c, z, t);
  } else if (c.target == "teacher") {
    const std::vector<S> teacher = read<S>(c.key + ".teacher");
    const sdca::kernel::Teacher<S> t = {teacher.data()};
    batch(c, z, t);
  } else {
    throw std::runtime_error(c.id + ": unknown target " + c.target);
  }
}

// sdca/loss.h, row by row: the public C++ functions of one row.
template <typename S>
void run_rows(const Case& c) {
  const std::vector<S> z = read<S>(c.key + ".scores");
  std::vector<double> values(c.n);
  std::vector<S> grads(c.n * c.classes);
  const sdca::kernel::Kind kind = static_cast<sdca::kernel::Kind>(c.kind);
  const sdca::kernel::Traits& takes = sdca::kernel::traits(c.kind);
  sdca::loss_parameters params;
  if (takes.takes_k) params.k = c.k;
  if (takes.hinge || takes.proximal) params.gamma = c.gamma;
  std::vector<int64_t> labels, offsets;
  std::vector<uint8_t> mask;
  std::vector<S> teacher;
  if (c.target == "labels") labels = read<int64_t>(c.key + ".labels");
  if (c.target == "sets") labels = read<int64_t>(c.key + ".set_labels");
  if (c.target == "sets") offsets = read<int64_t>(c.key + ".offsets");
  if (c.target == "mask") mask = read<uint8_t>(c.key + ".mask");
  if (c.target == "teacher") teacher = read<S>(c.key + ".teacher");
  for (std::size_t i = 0; i < c.n; ++i) {
    const S* row = z.data() + i * c.classes;
    S* g = grads.data() + i * c.classes;
    try {
      if (c.target == "labels") {
        values[i] = sdca::loss_and_gradient<S>(kind, c.classes, row,
                                               static_cast<std::size_t>(labels[i]), g, params);
      } else if (c.target == "sets" || c.target == "mask") {
        std::vector<std::size_t> set;
        if (c.target == "sets") {
          for (int64_t j = offsets[i]; j < offsets[i + 1]; ++j)
            set.push_back(static_cast<std::size_t>(labels[static_cast<std::size_t>(j)]));
        } else {
          for (std::size_t j = 0; j < c.classes; ++j)
            if (mask[i * c.classes + j]) set.push_back(j);
        }
        values[i] =
            sdca::loss_and_gradient<S>(kind, c.classes, row, set.data(), set.size(), g, params);
      } else if (c.kind == sdca::kernel::distillation) {
        values[i] = sdca::distillation_loss<S>(c.classes, row, teacher.data() + i * c.classes,
                                               c.temperature, g);
      } else {
        values[i] = sdca::proximal_distillation_loss<S>(
            c.classes, row, teacher.data() + i * c.classes, c.temperature, c.gamma, g);
      }
    } catch (const std::exception& error) {
      std::ofstream out(dir + "/" + c.id + ".cpp_loss.error");
      out << error.what();
      return;
    }
  }
  write(c.id + ".cpp_loss.values", values);
  write(c.id + ".cpp_loss.grads", grads);
}

template <typename S>
void run(const Case& c) {
  run_kernel<S>(c);
  const bool options = c.weight || c.ignore || c.label_smoothing != 0 || c.precision != 0 ||
                       c.upstream != 1 || c.check_rows;
  if (!options && c.target != "probs") run_rows<S>(c);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: cross_interface DIR\n";
    return 2;
  }
  dir = argv[1];
  std::ifstream manifest(dir + "/cases.txt");
  if (!manifest) {
    std::cerr << "cannot read " << dir << "/cases.txt\n";
    return 2;
  }
  std::string line;
  int failures = 0;
  while (std::getline(manifest, line)) {
    if (line.empty()) continue;
    std::istringstream in(line);
    std::vector<std::string> f;
    for (std::string word; in >> word;) f.push_back(word);
    Case c;
    try {
      if (f.size() != 18) throw std::runtime_error("malformed line: " + line);
      const auto number = [&f](std::size_t i) { return std::strtod(f[i].c_str(), nullptr); };
      const auto integer = [&f](std::size_t i) { return std::strtoll(f[i].c_str(), nullptr, 10); };
      c.id = f[0];
      c.key = f[1];
      c.kind = static_cast<int>(integer(2));
      c.k = static_cast<std::size_t>(integer(3));
      c.gamma = number(4);
      c.temperature = number(5);
      c.label_smoothing = number(6);
      c.ignore = static_cast<int>(integer(7));
      c.ignore_index = integer(8);
      c.dtype = f[9];
      c.n = static_cast<std::size_t>(integer(10));
      c.classes = static_cast<std::size_t>(integer(11));
      c.target = f[12];
      c.weight = static_cast<int>(integer(13));
      c.threads = static_cast<std::size_t>(integer(14));
      c.precision = static_cast<int>(integer(15));
      c.upstream = number(16);
      c.check_rows = static_cast<int>(integer(17));
      if (c.dtype == "float32") {
        run<float>(c);
      } else if (c.dtype == "float64") {
        run<double>(c);
      } else if (c.dtype == "float16") {
        run_kernel<sdca::half>(c);
      } else if (c.dtype == "bfloat16") {
        run_kernel<sdca::bfloat16>(c);
      } else {
        throw std::runtime_error("unknown dtype " + c.dtype);
      }
    } catch (const std::exception& error) {
      std::cerr << c.id << ": " << error.what() << "\n";
      ++failures;
    }
  }
  return failures ? 1 : 0;
}
