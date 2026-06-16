// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"
#include "mlx/primitives.h"

namespace nb = nanobind;
namespace mx = mlx::core;

namespace alayajet::quest::native {

namespace {

std::string dtype_name(mx::Dtype dtype) {
  if (dtype.val() == mx::Dtype::Val::float16) {
    return "fp16";
  }
  if (dtype.val() == mx::Dtype::Val::bfloat16) {
    return "bf16";
  }
  if (dtype.val() == mx::Dtype::Val::float32) {
    return "fp32";
  }
  throw std::invalid_argument("resident_write only supports fp16/bf16/fp32");
}

std::string metal_type(mx::Dtype dtype) {
  if (dtype.val() == mx::Dtype::Val::float16) {
    return "half";
  }
  if (dtype.val() == mx::Dtype::Val::bfloat16) {
    return "bfloat16_t";
  }
  if (dtype.val() == mx::Dtype::Val::float32) {
    return "float";
  }
  throw std::invalid_argument("resident_write only supports fp16/bf16/fp32");
}

std::string kernel_prelude() {
  return "#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "#if defined(__HAVE_BFLOAT__)\n"
         "typedef bfloat bfloat16_t;\n"
         "#endif\n";
}

class ResidentWritePrimitive : public mx::Primitive {
 public:
  explicit ResidentWritePrimitive(mx::Stream stream, int page_offset)
      : mx::Primitive(stream), page_offset_(page_offset) {}

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("ResidentWritePrimitive only supports GPU");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    const auto& frame_in = inputs[0];
    const auto& k_slice = inputs[1];
    const auto& v_slice = inputs[2];
    auto& frame_out = outputs[0];

    frame_out.copy_shared_buffer(frame_in);

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto& enc = d.get_command_encoder(s.index);

    std::string kname = "resident_kv_write_" + dtype_name(frame_out.dtype());
    auto lib = d.get_library(kname, [&]() {
      auto type_string = metal_type(frame_out.dtype());
      std::string source = kernel_prelude();
      source += "\nkernel void ";
      source += kname;
      source += "(\n    const device ";
      source += type_string;
      source += "* frame_in [[buffer(0)]],\n    const device ";
      source += type_string;
      source += "* k_slice [[buffer(1)]],\n    const device ";
      source += type_string;
      source += "* v_slice [[buffer(2)]],\n    device ";
      source += type_string;
      source += "* frame_out [[buffer(3)]],\n";
      source +=
          "    constant const int& page_offset [[buffer(4)]],\n"
          "    constant const int& page_size [[buffer(5)]],\n"
          "    constant const int& head_dim [[buffer(6)]],\n"
          "    constant const int& token_count [[buffer(7)]],\n"
          "    uint tid [[thread_position_in_grid]]) {\n"
          "  const int row = int(tid);\n"
          "  if (row >= token_count) {\n"
          "    return;\n"
          "  }\n"
          "  const int k_row = page_offset + row;\n"
          "  const int v_row = page_size + page_offset + row;\n"
          "  const int row_stride = head_dim;\n"
          "  for (int d = 0; d < row_stride; ++d) {\n"
          "    frame_out[(k_row * row_stride) + d] = k_slice[(row * row_stride) + d];\n"
          "    frame_out[(v_row * row_stride) + d] = v_slice[(row * row_stride) + d];\n"
          "  }\n"
          "}\n";
      return source;
    });
    auto kernel = d.get_kernel(kname, lib);
    int page_size = static_cast<int>(frame_out.shape(1));
    int head_dim = static_cast<int>(frame_out.shape(2));
    int token_count = static_cast<int>(k_slice.shape(0));
    int page_offset = page_offset_;

    auto group = MTL::Size::Make(std::max(1, std::min(token_count, 256)), 1, 1);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(frame_in, 0);
    enc.set_input_array(k_slice, 1);
    enc.set_input_array(v_slice, 2);
    enc.set_output_array(frame_out, 3);
    enc.set_bytes(page_offset, 4);
    enc.set_bytes(page_size, 5);
    enc.set_bytes(head_dim, 6);
    enc.set_bytes(token_count, 7);
    enc.dispatch_threads(MTL::Size::Make(token_count, 1, 1), group);
  }

  const char* name() const override {
    return "ResidentWrite";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const ResidentWritePrimitive*>(&other);
    return rhs && rhs->page_offset_ == page_offset_;
  }

 private:
  int page_offset_;
};

class ResidentWriteBatchedPrimitive : public mx::Primitive {
 public:
  ResidentWriteBatchedPrimitive(mx::Stream stream, int page_offset, int num_heads)
      : mx::Primitive(stream), page_offset_(page_offset), num_heads_(num_heads) {}

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("ResidentWriteBatchedPrimitive only supports GPU");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    for (int i = 0; i < num_heads_; ++i) {
      outputs[i].copy_shared_buffer(inputs[i]);
    }

    const auto& k_slice = inputs[num_heads_];
    const auto& v_slice = inputs[num_heads_ + 1];
    auto& frame0 = outputs[0];

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto& enc = d.get_command_encoder(s.index);

    std::string kname = "resident_kv_write_batched_" +
        dtype_name(frame0.dtype()) + "_h" + std::to_string(num_heads_);
    auto lib = d.get_library(kname, [&]() {
      auto type_string = metal_type(frame0.dtype());
      std::string source = kernel_prelude();
      source += "\nkernel void ";
      source += kname;
      source += "(\n    const device ";
      source += type_string;
      source += "* k_slice [[buffer(0)]],\n    const device ";
      source += type_string;
      source += "* v_slice [[buffer(1)]],\n";
      for (int h = 0; h < num_heads_; ++h) {
        source += "    device ";
        source += type_string;
        source += "* frame_out_";
        source += std::to_string(h);
        source += " [[buffer(";
        source += std::to_string(2 + h);
        source += ")]],\n";
      }
      int cbase = 2 + num_heads_;
      source +=
          "    constant const int& page_offset [[buffer(" + std::to_string(cbase) + ")]],\n"
          "    constant const int& page_size [[buffer(" + std::to_string(cbase + 1) + ")]],\n"
          "    constant const int& head_dim [[buffer(" + std::to_string(cbase + 2) + ")]],\n"
          "    constant const int& token_count [[buffer(" + std::to_string(cbase + 3) + ")]],\n"
          "    uint tid [[thread_position_in_grid]]) {\n"
          "  const int total = " + std::to_string(num_heads_) + " * token_count * head_dim;\n"
          "  const int idx = int(tid);\n"
          "  if (idx >= total) return;\n"
          "  const int d = idx % head_dim;\n"
          "  const int tmp = idx / head_dim;\n"
          "  const int h = tmp % " + std::to_string(num_heads_) + ";\n"
          "  const int t = tmp / " + std::to_string(num_heads_) + ";\n"
          "  const int src = ((t * " + std::to_string(num_heads_) + " + h) * head_dim) + d;\n"
          "  const int k_dst = ((page_offset + t) * head_dim) + d;\n"
          "  const int v_dst = ((page_size + page_offset + t) * head_dim) + d;\n";
      for (int h = 0; h < num_heads_; ++h) {
        source += h == 0 ? "  if" : "  else if";
        source += " (h == " + std::to_string(h) + ") {\n";
        source += "    frame_out_" + std::to_string(h) + "[k_dst] = k_slice[src];\n";
        source += "    frame_out_" + std::to_string(h) + "[v_dst] = v_slice[src];\n";
        source += "  }\n";
      }
      source += "}\n";
      return source;
    });
    auto kernel = d.get_kernel(kname, lib);
    int page_size = static_cast<int>(frame0.shape(1));
    int head_dim = static_cast<int>(frame0.shape(2));
    int token_count = static_cast<int>(k_slice.shape(0));
    int total = num_heads_ * token_count * head_dim;
    int page_offset = page_offset_;

    auto group = MTL::Size::Make(std::max(1, std::min(total, 256)), 1, 1);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(k_slice, 0);
    enc.set_input_array(v_slice, 1);
    for (int h = 0; h < num_heads_; ++h) {
      enc.set_output_array(outputs[h], 2 + h);
    }
    int cbase = 2 + num_heads_;
    enc.set_bytes(page_offset, cbase);
    enc.set_bytes(page_size, cbase + 1);
    enc.set_bytes(head_dim, cbase + 2);
    enc.set_bytes(token_count, cbase + 3);
    enc.dispatch_threads(MTL::Size::Make(total, 1, 1), group);
  }

  const char* name() const override {
    return "ResidentWriteBatched";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const ResidentWriteBatchedPrimitive*>(&other);
    return rhs && rhs->page_offset_ == page_offset_ && rhs->num_heads_ == num_heads_;
  }

 private:
  int page_offset_;
  int num_heads_;
};

class ResidentArenaWriteBatchedPrimitive : public mx::Primitive {
 public:
  explicit ResidentArenaWriteBatchedPrimitive(mx::Stream stream, int page_offset)
      : mx::Primitive(stream), page_offset_(page_offset) {}

  void eval_cpu(
      const std::vector<mx::array>&,
      std::vector<mx::array>&) override {
    throw std::runtime_error("ResidentArenaWriteBatchedPrimitive only supports GPU");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    const auto& arena_in = inputs[0];
    const auto& k_slice = inputs[1];
    const auto& v_slice = inputs[2];
    const auto& frame_ids = inputs[3];
    auto& arena_out = outputs[0];

    arena_out.copy_shared_buffer(arena_in);

    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    auto& enc = d.get_command_encoder(s.index);

    std::string kname = "resident_kv_arena_write_batched_" +
        dtype_name(arena_out.dtype());
    auto lib = d.get_library(kname, [&]() {
      auto type_string = metal_type(arena_out.dtype());
      std::string source = kernel_prelude();
      source += "\nkernel void ";
      source += kname;
      source += "(\n    const device ";
      source += type_string;
      source += "* k_slice [[buffer(0)]],\n    const device ";
      source += type_string;
      source += "* v_slice [[buffer(1)]],\n    const device int64_t* frame_ids [[buffer(2)]],\n    device ";
      source += type_string;
      source += "* arena_out [[buffer(3)]],\n";
      source +=
          "    constant const int& page_offset [[buffer(4)]],\n"
          "    constant const int& page_size [[buffer(5)]],\n"
          "    constant const int& head_dim [[buffer(6)]],\n"
          "    constant const int& num_heads [[buffer(7)]],\n"
          "    constant const int& token_count [[buffer(8)]],\n"
          "    uint tid [[thread_position_in_grid]]) {\n"
          "  const int idx = int(tid);\n"
          "  const int total = num_heads * token_count * head_dim;\n"
          "  if (idx >= total) {\n"
          "    return;\n"
          "  }\n"
          "  const int d = idx % head_dim;\n"
          "  const int tmp = idx / head_dim;\n"
          "  const int h = tmp % num_heads;\n"
          "  const int t = tmp / num_heads;\n"
          "  const int64_t frame = frame_ids[h];\n"
          "  const int64_t frame_base = frame * 2 * page_size * head_dim;\n"
          "  const int k_dst = int(frame_base + (page_offset + t) * head_dim + d);\n"
          "  const int v_dst = int(frame_base + (page_size + page_offset + t) * head_dim + d);\n"
          "  const int src = ((t * num_heads + h) * head_dim) + d;\n"
          "  arena_out[k_dst] = k_slice[src];\n"
          "  arena_out[v_dst] = v_slice[src];\n"
          "}\n";
      return source;
    });
    auto kernel = d.get_kernel(kname, lib);

    int page_size = static_cast<int>(arena_out.shape(2));
    int head_dim = static_cast<int>(arena_out.shape(3));
    int num_heads = static_cast<int>(frame_ids.shape(0));
    int token_count = static_cast<int>(k_slice.shape(0));
    int total = num_heads * token_count * head_dim;
    int page_offset = page_offset_;

    auto group = MTL::Size::Make(std::max(1, std::min(total, 256)), 1, 1);
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(k_slice, 0);
    enc.set_input_array(v_slice, 1);
    enc.set_input_array(frame_ids, 2);
    enc.set_output_array(arena_out, 3);
    enc.set_bytes(page_offset, 4);
    enc.set_bytes(page_size, 5);
    enc.set_bytes(head_dim, 6);
    enc.set_bytes(num_heads, 7);
    enc.set_bytes(token_count, 8);
    enc.dispatch_threads(MTL::Size::Make(total, 1, 1), group);
  }

  const char* name() const override {
    return "ResidentArenaWriteBatched";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* rhs = dynamic_cast<const ResidentArenaWriteBatchedPrimitive*>(&other);
    return rhs && rhs->page_offset_ == page_offset_;
  }

 private:
  int page_offset_;
};

mx::array resident_write_impl(
    const mx::array& frame,
    const mx::array& k_slice,
    const mx::array& v_slice,
    int page_offset,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument("resident_write only supports GPU");
  }

  auto out = mx::array(
      frame.shape(), frame.dtype(),
      std::make_shared<ResidentWritePrimitive>(stream, page_offset),
      {frame, k_slice, v_slice});
  return out;
}

std::vector<mx::array> resident_write_batched_impl(
    const std::vector<mx::array>& frames,
    const mx::array& k_slice,
    const mx::array& v_slice,
    int page_offset,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument("resident_write_batched only supports GPU");
  }
  if (frames.empty()) {
    throw std::invalid_argument("resident_write_batched requires frames");
  }
  int num_heads = static_cast<int>(frames.size());
  std::vector<mx::array> inputs = frames;
  inputs.push_back(k_slice);
  inputs.push_back(v_slice);
  std::vector<mx::Shape> shapes;
  std::vector<mx::Dtype> dtypes;
  for (const auto& frame : frames) {
    shapes.push_back(frame.shape());
    dtypes.push_back(frame.dtype());
  }
  auto prim = std::make_shared<ResidentWriteBatchedPrimitive>(
      stream, page_offset, num_heads);
  return mx::array::make_arrays(shapes, dtypes, prim, inputs);
}

mx::array resident_arena_write_batched_impl(
    const mx::array& arena,
    const mx::array& frame_ids,
    const mx::array& k_slice,
    const mx::array& v_slice,
    int page_offset,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  if (stream.device != mx::Device::gpu) {
    throw std::invalid_argument("resident_arena_write_batched only supports GPU");
  }
  if (frame_ids.shape(0) == 0) {
    throw std::invalid_argument("resident_arena_write_batched requires frame_ids");
  }
  std::vector<mx::array> inputs;
  inputs.push_back(arena);
  inputs.push_back(k_slice);
  inputs.push_back(v_slice);
  inputs.push_back(frame_ids);
  auto prim = std::make_shared<ResidentArenaWriteBatchedPrimitive>(stream, page_offset);
  return mx::array(arena.shape(), arena.dtype(), std::move(prim), inputs);
}

} // namespace

void init_resident_write_library() {
}

nb::tuple resident_arena_write_batched(
    nb::handle arena_h,
    nb::handle frame_ids_h,
    nb::handle k_slice_h,
    nb::handle v_slice_h,
    int page_offset) {
  auto& arena = *nb::inst_ptr<mx::array>(arena_h);
  auto& frame_ids = *nb::inst_ptr<mx::array>(frame_ids_h);
  auto& k_slice = *nb::inst_ptr<mx::array>(k_slice_h);
  auto& v_slice = *nb::inst_ptr<mx::array>(v_slice_h);
  auto out = resident_arena_write_batched_impl(
      arena, frame_ids, k_slice, v_slice, page_offset, mx::Device::gpu);
  nb::object mx_core = nb::module_::import_("mlx.core");
  nb::object arr_cls = mx_core.attr("array");
  nb::object out_arr = arr_cls(nb::int_(0));
  nb::inst_ptr<mx::array>(out_arr)->overwrite_descriptor(out);
  return nb::make_tuple(out_arr);
}

nb::tuple resident_write(
    nb::handle frame_h,
    nb::handle k_slice_h,
    nb::handle v_slice_h,
    int page_offset) {
  auto& frame = *nb::inst_ptr<mx::array>(frame_h);
  auto& k_slice = *nb::inst_ptr<mx::array>(k_slice_h);
  auto& v_slice = *nb::inst_ptr<mx::array>(v_slice_h);
  auto out = resident_write_impl(
      frame, k_slice, v_slice, page_offset, mx::Device::gpu);
  nb::object mx_core = nb::module_::import_("mlx.core");
  nb::object arr_cls = mx_core.attr("array");
  nb::object zero_arg = nb::int_(0);
  nb::object out_arr = arr_cls(zero_arg);
  nb::inst_ptr<mx::array>(out_arr)->overwrite_descriptor(out);
  return nb::make_tuple(out_arr);
}

nb::list resident_write_batched(
    nb::sequence frames_h,
    nb::handle k_slice_h,
    nb::handle v_slice_h,
    int page_offset) {
  std::vector<mx::array> frames;
  frames.reserve(nb::len(frames_h));
  for (nb::handle frame_h : frames_h) {
    frames.push_back(*nb::inst_ptr<mx::array>(frame_h));
  }
  auto& k_slice = *nb::inst_ptr<mx::array>(k_slice_h);
  auto& v_slice = *nb::inst_ptr<mx::array>(v_slice_h);
  auto outs = resident_write_batched_impl(
      frames, k_slice, v_slice, page_offset, mx::Device::gpu);
  nb::object mx_core = nb::module_::import_("mlx.core");
  nb::object arr_cls = mx_core.attr("array");
  nb::list result;
  for (size_t i = 0; i < outs.size(); ++i) {
    nb::object out_arr = arr_cls(nb::int_(0));
    nb::inst_ptr<mx::array>(out_arr)->overwrite_descriptor(outs[i]);
    result.append(out_arr);
  }
  return result;
}

NB_MODULE(_quest_native, m) {
  m.def("resident_write", &resident_write,
        nb::arg("frame"),
        nb::arg("k_slice"),
        nb::arg("v_slice"),
        nb::arg("page_offset"));
  m.def("resident_write_batched", &resident_write_batched,
        nb::arg("frames"),
        nb::arg("k_slice"),
        nb::arg("v_slice"),
        nb::arg("page_offset"));
  m.def("resident_arena_write_batched", &resident_arena_write_batched,
        nb::arg("arena"),
        nb::arg("frame_ids"),
        nb::arg("k_slice"),
        nb::arg("v_slice"),
        nb::arg("page_offset"));
  m.def("init_resident_write_library", &init_resident_write_library);
}

} // namespace alayajet::quest::native
