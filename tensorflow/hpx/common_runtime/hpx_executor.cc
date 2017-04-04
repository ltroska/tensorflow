#include "tensorflow/hpx/common_runtime/hpx_executor.h"

#include <hpx/util/unwrapped.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/lcos/when_any.hpp>
#include <hpx/include/run_as.hpp>

#include <mutex>

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow
{
namespace
{

  // 1-D, 0 element tensor.
  static const Tensor* const kEmptyTensor = new Tensor;

  bool IsInitializationOp(const Node* node)
  {
    return node->op_def().allows_uninitialized_input();
  }

  // Sets the timeline_label field of *node_stats, using data from *node.
  // Returns true iff the node is a transfer node.
  // TODO(tucker): merge with the DetailText function in session.cc
  // in a common location.
  bool SetTimelineLabel(const Node* node, NodeExecStats* node_stats)
  {
    bool is_transfer_node = false;
    string memory;
    for (auto& all : node_stats->memory()) {
      int64 tot = all.total_bytes();
      if (tot >= 0.1 * 1048576.0) {
        int64 peak = all.peak_bytes();
        if (peak > 0) {
          memory = strings::StrCat(memory,
                                   "[",
                                   all.allocator_name(),
                                   strings::Printf(" %.1fMB %.1fMB] ",
                                                   tot / 1048576.0,
                                                   peak / 1048576.0));
        } else {
          memory =
              strings::StrCat(memory,
                              "[",
                              all.allocator_name(),
                              strings::Printf(" %.1fMB] ", tot / 1048576.0));
        }
      }
    }
    const NodeDef& def = node->def();
    string text = "";
    if (IsSend(node)) {
      string tensor_name;
      TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
      string recv_device;
      TF_CHECK_OK(GetNodeAttr(def, "recv_device", &recv_device));
      text = strings::StrCat(memory,
                             def.name(),
                             " = ",
                             def.op(),
                             "(",
                             tensor_name,
                             " @",
                             recv_device);
      is_transfer_node = true;
    } else if (IsRecv(node)) {
      string tensor_name;
      TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
      string send_device;
      TF_CHECK_OK(GetNodeAttr(def, "send_device", &send_device));
      text = strings::StrCat(memory,
                             def.name(),
                             " = ",
                             def.op(),
                             "(",
                             tensor_name,
                             " @",
                             send_device);
      is_transfer_node = true;
    } else {
      text = strings::StrCat(
          memory,
          def.name(),
          " = ",
          def.op(),
          "(",
          str_util::Join(
              std::vector<StringPiece>(def.input().begin(), def.input().end()),
              ", "),
          ")");
    }
    node_stats->set_timeline_label(text);
    return is_transfer_node;
  }

  // Helper routines for collecting step stats.
  namespace nodestats
  {
    inline int64 NowInUsec()
    {
      return Env::Default()->NowMicros();
    }

    void SetScheduled(NodeExecStats* nt, int64 t)
    {
      nt->set_scheduled_micros(t);
    }

    void SetAllStart(NodeExecStats* nt)
    {
      nt->set_all_start_micros(NowInUsec());
    }

    void SetOpStart(NodeExecStats* nt)
    {
      DCHECK_NE(nt->all_start_micros(), 0);
      nt->set_op_start_rel_micros(NowInUsec() - nt->all_start_micros());
    }

    void SetOpEnd(NodeExecStats* nt)
    {
      DCHECK_NE(nt->all_start_micros(), 0);
      nt->set_op_end_rel_micros(NowInUsec() - nt->all_start_micros());
    }

    void SetAllEnd(NodeExecStats* nt)
    {
      DCHECK_NE(nt->all_start_micros(), 0);
      nt->set_all_end_rel_micros(NowInUsec() - nt->all_start_micros());
    }

    void SetOutput(NodeExecStats* nt, int slot, const Tensor* v)
    {
      DCHECK(v);
      NodeOutput* no = nt->add_output();
      no->set_slot(slot);
      v->FillDescription(no->mutable_tensor_description());
    }

    void SetMemory(NodeExecStats* nt, OpKernelContext* ctx)
    {
      for (const auto& allocator_pair : ctx->wrapped_allocators()) {
        AllocatorMemoryUsed* memory = nt->add_memory();
        // retrieving the sizes from the wrapped allocator removes the
        // executor's reference to it, so allocator_pair.second must not
        // be dereferenced again after this statement
        auto sizes = allocator_pair.second->GetSizesAndUnRef();
        memory->set_allocator_name(allocator_pair.first->Name());
        memory->set_total_bytes(std::get<0>(sizes));
        if (allocator_pair.first->TracksAllocationSizes()) {
          memory->set_peak_bytes(std::get<1>(sizes));
          memory->set_live_bytes(std::get<2>(sizes));
        }
      }
      auto* ms = nt->mutable_memory_stats();
      ms->set_host_temp_memory_size(ctx->host_temp_memory_size());
      ms->set_device_temp_memory_size(ctx->device_temp_memory_size());
      for (const auto& alloc_id : ctx->host_persistent_alloc_ids()) {
        ms->mutable_host_persistent_tensor_alloc_ids()->Add(alloc_id);
      }
      for (const auto& alloc_id : ctx->device_persistent_alloc_ids()) {
        ms->mutable_device_persistent_tensor_alloc_ids()->Add(alloc_id);
      }
      ms->set_host_persistent_memory_size(
          ctx->host_persistent_memory_allocated());
      ms->set_device_persistent_memory_size(
          ctx->device_persistent_memory_allocated());
    }

    void SetReferencedTensors(NodeExecStats* nt,
                              const TensorReferenceVector& tensors)
    {
      // be careful not to increment the reference count on any tensor
      // while recording the information
      for (size_t i = 0; i < tensors.size(); ++i) {
        AllocationDescription* description = nt->add_referenced_tensor();
        tensors.at(i).FillDescription(description);
      }
    }

  } // namespace nodestats

  template <class T> unsigned numDigits(T number)
  {
    unsigned digits = 0;
    if (number <= 0)
      digits = 1;
    while (number) {
      number /= 10;
      digits++;
    }
    return digits;
  }

  class HPXExecutorImpl;
  class GraphView;

  struct EdgeInfo
  {
    int dst_id;
    int output_slot : 31;
    // true if this is the last info for output_slot in the EdgeInfo list.
    bool is_last : 1;
    int input_slot;
  };

  struct NodeItem
  {
    NodeItem()
    {
    }

    // A graph node.
    const Node* node = nullptr;

    // The kernel for this node.
    OpKernel* kernel = nullptr;

    bool kernel_is_expensive : 1; // True iff kernel->IsExpensive()
    bool kernel_is_async : 1;     // True iff kernel->AsAsync() != nullptr
    bool is_merge : 1;            // True iff IsMerge(node)
    bool is_enter : 1;            // True iff IsEnter(node)
    bool is_exit : 1;             // True iff IsExit(node)
    bool is_control_trigger : 1;  // True iff IsControlTrigger(node)
    bool is_sink : 1;             // True iff IsSink(node)
    // True iff IsEnter(node) || IsExit(node) || IsNextIteration(node)
    bool is_enter_exit_or_next_iter : 1;

    // Cached values of node->num_inputs() and node->num_outputs(), to
    // avoid levels of indirection.
    int num_inputs;
    int num_outputs;

    // ExecutorImpl::tensors_[input_start] is the 1st positional input
    // for this node.
    int input_start = 0;

    // Number of output edges.
    int num_output_edges;

    const EdgeInfo* output_edge_list() const
    {
      return output_edge_base();
    }

    // ith output edge.
    const EdgeInfo& output_edge(int i) const
    {
      DCHECK_GE(i, 0);
      DCHECK_LT(i, num_output_edges);
      return output_edge_base()[i];
    }

    DataType input_type(int i) const
    {
      DCHECK_LT(i, num_inputs);
      return static_cast<DataType>(input_type_base()[i]);
    }
    DataType output_type(int i) const
    {
      DCHECK_LT(i, num_outputs);
      return static_cast<DataType>(output_type_base()[i]);
    }

    // Return array of per-output allocator attributes.
    const AllocatorAttributes* output_attrs() const
    {
      return output_attr_base();
    }

private:
    friend class GraphView;

    // Variable length section starts immediately after *this
    // (uint8 is enough for DataType).
    //   EdgeInfo            out_edges[num_out_edges];
    //   AllocatorAttributes output_attr[num_outputs];
    //   uint8               input_type[num_inputs];
    //   uint8               output_type[num_outputs];

    // Return pointer to variable length section.
    char* var() const
    {
      return const_cast<char*>(reinterpret_cast<const char*>(this) +
                               sizeof(NodeItem));
    }

    EdgeInfo* output_edge_base() const
    {
      return reinterpret_cast<EdgeInfo*>(var());
    }
    AllocatorAttributes* output_attr_base() const
    {
      return reinterpret_cast<AllocatorAttributes*>(
          var() + sizeof(EdgeInfo) * num_output_edges);
    }
    uint8* input_type_base() const
    {
      return reinterpret_cast<uint8*>(
          var() + sizeof(EdgeInfo) * num_output_edges +
          sizeof(AllocatorAttributes) * num_outputs);
    }
    uint8* output_type_base() const
    {
      return reinterpret_cast<uint8*>(
          var() + sizeof(EdgeInfo) * num_output_edges +
          sizeof(AllocatorAttributes) * num_outputs +
          sizeof(uint8) * num_inputs);
    }

    TF_DISALLOW_COPY_AND_ASSIGN(NodeItem);
  };

  typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
  typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
  typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

  // Immutable view of a Graph organized for efficient execution.
  class GraphView
  {
public:
    GraphView() : space_(nullptr)
    {
    }
    ~GraphView();

    void Initialize(const Graph* g);
    Status SetAllocAttrs(const Graph* g, const Device* device);

    NodeItem* node(int id) const
    {
      DCHECK_GE(id, 0);
      DCHECK_LT(id, num_nodes_);
      uint32 offset = node_offsets_[id];
      return ((offset == kuint32max) ?
                  nullptr :
                  reinterpret_cast<NodeItem*>(space_ + node_offsets_[id]));
    }

private:
    char* InitializeNode(char* ptr, const Node* n);
    size_t NodeItemBytes(const Node* n);

    int32 num_nodes_ = 0;
    uint32* node_offsets_ = nullptr; // array of size "graph_.num_node_ids()"
    // node_offsets_[id] holds the byte offset for node w/ "id" in space_

    char* space_; // NodeItem objects are allocated here

    TF_DISALLOW_COPY_AND_ASSIGN(GraphView);
  };

  class HPXExecutorImpl : public Executor
  {
public:
    HPXExecutorImpl(const LocalExecutorParams& p, const Graph* g)
        : params_(p)
        , graph_(g)
        , gview_()
    {
      CHECK(p.create_kernel != nullptr);
      CHECK(p.delete_kernel != nullptr);
    }

    ~HPXExecutorImpl() override
    {
      for (int i = 0; i < graph_->num_node_ids(); i++) {
        NodeItem* item = gview_.node(i);
        if (item != nullptr) {
          params_.delete_kernel(item->kernel);
        }
      }

      delete graph_;
    }

    Status Initialize();

    // Process all Nodes in the current graph, attempting to infer the
    // memory allocation attributes to be used wherever they may allocate
    // a tensor buffer.
    Status SetAllocAttrs();

    void RunAsync(const Args& args, DoneCallback done) override;

private:
    friend class HPXExecutorState;

    // Owned.
    LocalExecutorParams params_;
    const Graph* graph_;
    GraphView gview_;

    // A cached value of params_
    bool device_record_tensor_accesses_ = false;

    std::vector<AllocatorAttributes> output_attrs_;

    TF_DISALLOW_COPY_AND_ASSIGN(HPXExecutorImpl);
  };

  // Infer memory allocation attributes of a node n's output,
  // based on its use node dst.  Note that dst might not be directly
  // connected to n by a single edge, but might be a downstream
  // consumer of n's output by reference.  *attr is updated with any
  // necessary attributes.
  Status InferAllocAttr(const Node* n,
                        const Node* dst,
                        const DeviceNameUtils::ParsedName& local_dev_name,
                        AllocatorAttributes* attr);

  GraphView::~GraphView()
  {
    static_assert(std::is_trivially_destructible<AllocatorAttributes>::value,
                  "Update code if AllocatorAttributes gains a destructor");
    static_assert(std::is_trivially_destructible<EdgeInfo>::value,
                  "Update code if EdgeInfo gains a destructor");
    for (int i = 0; i < num_nodes_; i++) {
      NodeItem* n = node(i);
      if (n != nullptr) {
        n->NodeItem::~NodeItem();
        // Memory for "n" itself is held in space_ & gets cleaned up below
      }
    }
    delete[] node_offsets_;
    delete[] space_;
  }

  size_t GraphView::NodeItemBytes(const Node* n)
  {
    const int num_output_edges = n->out_edges().size();
    const int num_inputs = n->num_inputs();
    const int num_outputs = n->num_outputs();

    // Compute number of bytes needed for NodeItem and variable length data.
    // We do not subtract sizeof(var) since num_inputs/num_outputs might
    // both be zero.
    const size_t raw_bytes =
        sizeof(NodeItem)                            // Fixed
        + num_output_edges * sizeof(EdgeInfo)       // output_edges[...]
        + num_outputs * sizeof(AllocatorAttributes) // output_attr[...]
        + num_inputs * sizeof(uint8)                // input_type[num_inputs]
        + num_outputs * sizeof(uint8);              // output_type[num_outputs]
    static constexpr size_t kItemAlignment = sizeof(NodeItem*);
    static_assert(kItemAlignment % alignof(NodeItem) == 0,
                  "NodeItem must be aligned with kItemAlignment");
    static_assert(kItemAlignment % alignof(EdgeInfo) == 0,
                  "EdgeInfo must be aligned with kItemAlignment");
    static_assert(kItemAlignment % alignof(AllocatorAttributes) == 0,
                  "AllocatorAttributes must be aligned with kItemAlignment");
    static_assert(sizeof(NodeItem) % alignof(EdgeInfo) == 0,
                  "NodeItem must be aligned with EdgeInfo");
    static_assert(sizeof(NodeItem) % alignof(AllocatorAttributes) == 0,
                  "NodeItem must be aligned with AllocatorAttributes");
    static_assert(sizeof(EdgeInfo) % alignof(AllocatorAttributes) == 0,
                  "EdgeInfo must be aligned with AllocatorAttributes");
    const size_t bytes =
        ((raw_bytes + kItemAlignment - 1) / kItemAlignment) * kItemAlignment;
    return bytes;
  }

  char* GraphView::InitializeNode(char* ptr, const Node* n)
  {
    const int id = n->id();
    CHECK(node_offsets_[id] == kuint32max); // Initial value in constructor

    const size_t bytes = NodeItemBytes(n);
    constexpr size_t kItemAlignment = sizeof(NodeItem*);
    CHECK_EQ(reinterpret_cast<uintptr_t>(ptr) % kItemAlignment, 0);
    NodeItem* item = reinterpret_cast<NodeItem*>(ptr);

    // We store a 32-bit offset relative to the beginning of space_, so that we
    // only need an array of 32-bit values to map from node id to the NodeItem*,
    // (versus 64 bits on most machines if we just stored an array of NodeItem*
    // pointers). Casting to int64 is needed on 32bit CPU to avoid comparing
    // values as "int" vs "size_t" in CHECK_LE.
    CHECK_LE(static_cast<int64>(ptr - space_), kuint32max);
    const uint32 offset = ptr - space_;
    node_offsets_[id] = offset;
    ptr += bytes;

    const int num_output_edges = n->out_edges().size();
    const int num_inputs = n->num_inputs();
    const int num_outputs = n->num_outputs();

    new (item) NodeItem();
    item->num_inputs = num_inputs;
    item->num_outputs = num_outputs;
    item->num_output_edges = num_output_edges;

    // Fill output edges.
    // Keep track of the last EdgeInfo in the EdngeInfo array that references
    // a given output slot.  For all but the last, we need to do a copy of the
    // Tensor when propagating results downstream in the graph, but for the
    // last one, we can just do a move of the Tensor object to propagate it.
    gtl::InlinedVector<EdgeInfo*, 4> last_indices(num_outputs, nullptr);
    EdgeInfo* dst_edge = item->output_edge_base();
    for (auto e : n->out_edges()) {
      dst_edge->dst_id = e->dst()->id();
      CHECK_LE(e->src_output(), ((int32)0x3FFFFFFF)); // Must fit in 31 bits
      dst_edge->output_slot = e->src_output();
      dst_edge->is_last = false;
      const int output_slot = dst_edge->output_slot;
      if (output_slot >= 0) {
        last_indices[output_slot] = dst_edge;
      }
      dst_edge->input_slot = e->dst_input();
      dst_edge++;
    }
    for (EdgeInfo* edge_info : last_indices) {
      if (edge_info != nullptr) {
        edge_info->is_last = true;
      }
    }

    AllocatorAttributes* output_attrs = item->output_attr_base();
    for (int i = 0; i < num_outputs; i++) {
      new (&output_attrs[i]) AllocatorAttributes();
    }

    DCHECK_LT(DataType_MAX, 255); // Must fit in uint8
    uint8* input_types = item->input_type_base();
    for (int i = 0; i < num_inputs; i++) {
      input_types[i] = static_cast<uint8>(n->input_type(i));
      DCHECK_EQ(item->input_type(i), n->input_type(i));
    }

    uint8* output_types = item->output_type_base();
    for (int i = 0; i < num_outputs; i++) {
      output_types[i] = static_cast<uint8>(n->output_type(i));
      DCHECK_EQ(item->output_type(i), n->output_type(i));
    }
    return ptr;
  }

  void GraphView::Initialize(const Graph* g)
  {
    CHECK(node_offsets_ == nullptr);
    const int num_nodes = g->num_node_ids();
    num_nodes_ = num_nodes;
    size_t total_bytes = 0;
    for (const Node* n : g->nodes()) {
      total_bytes += NodeItemBytes(n);
    }

    node_offsets_ = new uint32[num_nodes];
    for (int i = 0; i < num_nodes; i++) {
      node_offsets_[i] = kuint32max;
    }

    space_ = new char[total_bytes]; // NodeItem objects are allocated here
    char* ptr = space_;
    for (const Node* n : g->nodes()) {
      ptr = InitializeNode(ptr, n);
    }
    CHECK_EQ(ptr, space_ + total_bytes);
  }

  Status HPXExecutorImpl::Initialize()
  {
    gview_.Initialize(graph_);

    // Cache this value so we make this virtual function call once, rather
    // that O(# steps * # nodes per step) times.
    device_record_tensor_accesses_ =
        params_.device->RequiresRecordingAccessedTensors();

    // Preprocess every node in the graph to create an instance of op
    // kernel for each node.
    for (const Node* n : graph_->nodes()) {
      const int id = n->id();

      NodeItem* item = gview_.node(id);
      item->node = n;

      /* item->num_inputs = n->num_inputs();
       item->num_outputs = n->num_outputs();*/

      /*    for (int i = 0; i < std::min(4, item->num_inputs); i++) {
            item->inlined_input_type[i] = n->input_type(i);
          }
          for (int i = 0; i < std::min(4, item->num_outputs); i++) {
            item->inlined_output_type[i] = n->output_type(i);
          }*/

      Status s = params_.create_kernel(n->def(), &item->kernel);
      if (!s.ok()) {
        item->kernel = nullptr;
        s = AttachDef(s, n->def());
        LOG(ERROR) << "Executor failed to create kernel. " << s;
        return s;
      }
      CHECK(item->kernel);
      item->kernel_is_expensive = item->kernel->IsExpensive();
      item->kernel_is_async = (item->kernel->AsAsync() != nullptr);
      item->is_merge = IsMerge(n);
      item->is_enter = IsEnter(n);
      item->is_exit = IsExit(n);
      item->is_control_trigger = IsControlTrigger(n);
      item->is_sink = IsSink(n);
      item->is_enter_exit_or_next_iter =
          (IsEnter(n) || IsExit(n) || IsNextIteration(n));
    }

    return gview_.SetAllocAttrs(graph_, params_.device);
  }

  Status GraphView::SetAllocAttrs(const Graph* g, const Device* device)
  {
    Status s;
    DeviceNameUtils::ParsedName local_dev_name = device->parsed_name();

    for (const Node* n : g->nodes()) {
      NodeItem* item = node(n->id());
      AllocatorAttributes* attrs = item->output_attr_base();

      // Examine the out edges of each node looking for special use
      // cases that may affect memory allocation attributes.
      for (auto e : n->out_edges()) {
        if (!e->IsControlEdge()) {
          AllocatorAttributes attr;
          s = InferAllocAttr(n, e->dst(), local_dev_name, &attr);
          if (!s.ok())
            return s;
          if (attr.value != 0) {
            attrs[e->src_output()].Merge(attr);
          }
        }
      }

      for (int out = 0; out < n->num_outputs(); out++) {
        const OpKernel* op_kernel = item->kernel;
        DCHECK_LT(out, op_kernel->output_memory_types().size());
        bool on_host = op_kernel->output_memory_types()[out] == HOST_MEMORY;
        if (on_host) {
          AllocatorAttributes h;
          h.set_on_host(on_host);
          attrs[out].Merge(h);
        }
      }
    }
    return s;
  }

  Status InferAllocAttr(const Node* n,
                        const Node* dst,
                        const DeviceNameUtils::ParsedName& local_dev_name,
                        AllocatorAttributes* attr)
  {
    Status s;
    // Note that it's possible for *n to be a Recv and *dst to be a Send,
    // so these two cases are not mutually exclusive.
    if (IsRecv(n)) {
      string src_name;
      s = GetNodeAttr(n->def(), "send_device", &src_name);
      if (!s.ok())
        return s;
      DeviceNameUtils::ParsedName parsed_src_name;
      if (!DeviceNameUtils::ParseFullName(src_name, &parsed_src_name)) {
        s = errors::Internal(
            "Bad send_device attr '", src_name, "' in node ", n->name());
        return s;
      }
      if (!DeviceNameUtils::IsSameAddressSpace(parsed_src_name,
                                               local_dev_name)) {
        // Value is going to be the sink of an RPC.
        attr->set_nic_compatible(true);
        VLOG(2) << "node " << n->name() << " is the sink of an RPC in";
      } else if ((local_dev_name.type == "CPU" || n->IsHostRecv()) &&
                 parsed_src_name.type != "CPU") {
        // Value is going to be the sink of a local DMA from GPU to CPU (or
        // other
        // types of accelerators).
        attr->set_gpu_compatible(true);
        VLOG(2) << "node " << n->name() << " is the sink of a gpu->cpu copy";
      } else {
        VLOG(2) << "default alloc case local type " << local_dev_name.type
                << " remote type " << parsed_src_name.type;
      }
    }
    if (IsSend(dst)) {
      string dst_name;
      s = GetNodeAttr(dst->def(), "recv_device", &dst_name);
      if (!s.ok())
        return s;
      DeviceNameUtils::ParsedName parsed_dst_name;
      if (!DeviceNameUtils::ParseFullName(dst_name, &parsed_dst_name)) {
        s = errors::Internal(
            "Bad recv_device attr '", dst_name, "' in node ", n->name());
        return s;
      }
      if (!DeviceNameUtils::IsSameAddressSpace(parsed_dst_name,
                                               local_dev_name)) {
        // Value is going to be the source of an RPC.
        attr->set_nic_compatible(true);
        VLOG(2) << "node " << n->name() << " is the source of an RPC out";
      } else if ((local_dev_name.type == "CPU" || dst->IsHostSend()) &&
                 parsed_dst_name.type != "CPU") {
        // Value is going to be the source of a local DMA from CPU to GPU (or
        // other types of accelerators).
        // Note that this does not cover the case where the allocation of the
        // output tensor is not generated by the src: n.
        attr->set_gpu_compatible(true);
        VLOG(2) << "node " << n->name() << " is the source of a cpu->gpu copy";
      } else {
        VLOG(2) << "default alloc case local type " << local_dev_name.type
                << " remote type " << parsed_dst_name.type;
      }
    }
    return s;
  }

  // The state associated with one invocation of HPXExecutorImpl::Run.
  // HPXExecutorState dispatches nodes when they become ready and keeps
  // track of how many predecessors of a node have not done (pending_).
  class HPXExecutorState
  {
public:
    HPXExecutorState(const Executor::Args& args, HPXExecutorImpl* impl);
    ~HPXExecutorState();

    void RunAsync(Executor::DoneCallback done);

private:
    // Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
    // TODO(yuanbyu): A better way to do "has_value"?
    struct Entry
    {
      Entry()
      {
      }
      Entry(const Entry& other)
          : ref(other.ref)
          , ref_mu(other.ref_mu)
          , has_value(other.has_value)
          , is_dead(other.is_dead)
          , val_field_is_set(other.val_field_is_set)
          , alloc_attr(other.alloc_attr)
          , device_context(other.device_context)
      {
        if (val_field_is_set) {
          val.Init(*other.val);
        }
      }
      ~Entry()
      {
        if (val_field_is_set)
          val.Destroy();
      }

      Entry& operator=(const Entry& other)
      {
        if (val_field_is_set) {
          val.Destroy();
        }
        ref = other.ref;
        ref_mu = other.ref_mu;
        has_value = other.has_value;
        is_dead = other.is_dead;
        val_field_is_set = other.val_field_is_set;
        alloc_attr = other.alloc_attr;
        device_context = other.device_context;
        if (val_field_is_set) {
          val.Init(*other.val);
        }
        return *this;
      }

      // Clears the <val> field.
      void ClearVal()
      {
        if (val_field_is_set) {
          val.Destroy();
          val_field_is_set = false;
        }
      }

      // A tensor value, if val_field_is_set.
      ManualConstructor<Tensor> val;

      Tensor* ref = nullptr;   // A tensor reference.
      mutex* ref_mu = nullptr; // mutex for *ref if ref is not nullptr.

      // Whether the value exists, either in <val> or <ref>.
      bool has_value = false;

      bool is_dead = false;

      bool val_field_is_set = false;

      // The attributes of the allocator that creates the tensor.
      AllocatorAttributes alloc_attr;

      // Every entry carries an optional DeviceContext containing
      // Device-specific information about how the Tensor was produced.
      DeviceContext* device_context = nullptr;
    };

    // Contains a value for [node->id()] for the device context assigned by the
    // device at the beginning of a step.
    DeviceContextMap device_context_map_;

    std::unordered_map<std::string, hpx::lcos::local::promise<Entry> >
    promise_map_;
    std::unordered_map<std::string, hpx::lcos::shared_future<Entry> >
    future_map_;
    std::unordered_map<std::string, bool> is_set_map_;

    std::mutex finish_mutex_;
    //  std::condition_variable maybe_finished;
    bool was_finished;
    std::recursive_mutex map_mutex_;

    typedef gtl::InlinedVector<Entry, 4> EntryVector;

    const bool vlog_; // true if VLOG_IS_ON(1). Used to check vlog cheaply.

    // true if LogMemory::IsEnabled(). Used to check memory enabled cheaply.
    const bool log_memory_;

    int64 step_id_;
    // Not owned.
    Rendezvous* rendezvous_;
    SessionState* session_state_;
    TensorStore* tensor_store_;
    // Step-local container.
    ScopedStepContainer* step_container_;
    StepStatsCollector* stats_collector_;
    // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper
    // instead of a pointer?  (avoids having to delete).
    checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
    FunctionCallFrame* call_frame_;
    const HPXExecutorImpl* impl_;
    CancellationManager* cancellation_manager_;
    Executor::Args::Runner runner_;
    bool sync_on_finish_;

    std::vector<std::vector<const Node*> > iteration_nodes_;
    std::vector<bool> is_loop_node_;

    // Owned.

    // A flag that is set on error after the frame state has been
    // dumped for diagnostic purposes.
    bool dumped_on_error_ = false;

    // Invoked when the execution finishes.
    Executor::DoneCallback done_cb_;

    int64 num_outstanding_ops_;
    int64 num_computed_ops_;

    int64 num_nodes_;
    int64 num_node_ids_;

    mutex mu_;
    Status status_ GUARDED_BY(mu_);

    class AsyncState;

    void SignalNodeDone();

    // Process a ready node in current thread.
    void Process(const Node* node,
                 EntryVector inputs,
                 int64 iter_id,
                 int64 base_iter,
                 std::string const& key_prefix,
                 bool is_dead,
                 int64 scheduled_usec);

    // Before invoking item->kernel, fills in its "inputs".
    Status PrepareInputs(const NodeItem& item,
                         Entry* first_input,
                         TensorValueVec* inputs,
                         DeviceContextVec* input_device_contexts,
                         AllocatorAttributeVec* input_alloc_attrs,
                         bool* is_input_dead);

    // After item->kernel computation is done, processes its outputs.
    Status ProcessOutputs(const NodeItem& item,
                          OpKernelContext* ctx,
                          EntryVector* outputs,
                          NodeExecStats* stats);

    // After processing the outputs, propagates the outputs to their dsts.
    void PropagateOutputs(const NodeItem* item,
                          const EntryVector& outputs,
                          int64 iter_id,
                          const int64 base_iter,
                          std::string key_prefix,
                          const bool is_dead);

    Status BuildControlFlow(const Graph* g,
                            std::vector<std::vector<const Node*> >& loops);

    // Schedule all the expensive nodes in 'ready', and put all the inexpensive
    // nodes in 'ready' into 'inline_ready'.
    hpx::future<void> Schedule(const Node* node,
                               const int64 iter_id,
                               const int64 base_iter,
                               std::string const& key_prefix);

    //  const Tensor* GetTensorValueForDump(const Entry& input);

    // Clean up when this executor is done.
    void Finish();
  };

  struct HPXExecutorState::AsyncState
  {
    AsyncState(const OpKernelContext::Params& p,
               bool const _is_dead,
               const NodeItem* _item,
               Entry* _first_input,
               NodeExecStats* _stats)
        : saved_inputs(*p.inputs)
        , saved_input_device_contexts(*p.input_device_contexts)
        , saved_input_alloc_attrs(*p.input_alloc_attrs)
        , params(p)
        , is_dead(_is_dead)
        , item(_item)
        , first_input(_first_input)
        ,
        // ParamsButClearingEigenGPUDevice does equivalent of
        //   params.eigen_gpu_device = nullptr;
        ctx(ParamsButClearingEigenGPUDevice(&params), item->num_outputs)
        , stats(_stats)
    {
      params.inputs = &saved_inputs;
      params.input_device_contexts = &saved_input_device_contexts;
      params.input_alloc_attrs = &saved_input_alloc_attrs;
    }

    TensorValueVec saved_inputs;
    DeviceContextVec saved_input_device_contexts;
    AllocatorAttributeVec saved_input_alloc_attrs;
    OpKernelContext::Params params;
    bool is_dead;
    const NodeItem* item;
    Entry* first_input;
    OpKernelContext ctx;
    NodeExecStats* stats;

private:
    OpKernelContext::Params*
    ParamsButClearingEigenGPUDevice(OpKernelContext::Params* p)
    {
      // Ensure OpKernelContext constructor will make a new eigen GPU device if
      // necessary.
      p->eigen_gpu_device = nullptr; // Force allocation
      return p;
    }
  };

  HPXExecutorState::HPXExecutorState(const Executor::Args& args,
                                     HPXExecutorImpl* impl)
      : was_finished(false)
      , vlog_(VLOG_IS_ON(1))
      , log_memory_(LogMemory::IsEnabled())
      , step_id_(args.step_id)
      , rendezvous_(args.rendezvous)
      , session_state_(args.session_state)
      , tensor_store_(args.tensor_store)
      , step_container_(args.step_container)
      , stats_collector_(args.stats_collector)
      , slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper)
      , call_frame_(args.call_frame)
      , impl_(impl)
      , cancellation_manager_(args.cancellation_manager)
      , runner_(args.runner)
      , sync_on_finish_(args.sync_on_finish)
      , num_outstanding_ops_(0)
      , num_computed_ops_(0)
  {
    promise_map_.clear();
    future_map_.clear();
  }

  HPXExecutorState::~HPXExecutorState()
  {
    for (auto it : device_context_map_) {
      it->Unref();
    }
    delete slice_reader_cache_;
  }

  Status HPXExecutorState::BuildControlFlow(
      const Graph* g,
      std::vector<std::vector<const Node*> >& loops)
  {
    loops.resize(num_node_ids_);
    is_loop_node_.resize(num_node_ids_);
    std::vector<Node*> parent_nodes;
    parent_nodes.resize(num_node_ids_);
    std::vector<bool> visited;
    visited.resize(num_node_ids_);

    std::deque<Node*> ready;

    // Initialize with the root nodes.
    for (Node* n : g->nodes()) {
      if (n->in_edges().empty()) {
        visited[n->id()] = true;
        ready.push_back(n);
      }
    }

    std::vector<const Node*>* vec;

    while (!ready.empty()) {
      Node* curr_node = ready.front();
      int curr_id = curr_node->id();
      ready.pop_front();

      Node* parent = nullptr;
      if (IsEnter(curr_node)) {
        // Enter a child frame.
        parent = curr_node;
      } else if (IsExit(curr_node)) {
        // Exit to the parent frame.
        parent = parent_nodes[curr_id];
        parent = parent_nodes[parent->id()];
      } else {
        parent = parent_nodes[curr_id];
      }
      if (parent != nullptr)
        vec = &loops[parent->id()];

      for (const Edge* out_edge : curr_node->out_edges()) {
        Node* out = out_edge->dst();
        int out_id = out->id();

        // Add to ready queue if not visited.
        bool is_visited = visited[out_id];
        if (!is_visited) {
          ready.push_back(out);
          visited[out_id] = true;

          if (parent != nullptr) {
            is_loop_node_[out_id] = true;
            vec->push_back(out);
          }

          // Process the node 'out'.
          parent_nodes[out_id] = parent;
        }
      }
    }

    return Status::OK();
  }

  void HPXExecutorState::RunAsync(Executor::DoneCallback done)
  {
    const Graph* graph = impl_->graph_;

    // Ask the device to fill in the device context map.
    Device* device = impl_->params_.device;
    Status fill_status = device->FillContextMap(graph, &device_context_map_);
    if (!fill_status.ok()) {
      done(fill_status);
      return;
    }

    num_node_ids_ = graph->num_node_ids();
    num_nodes_ = graph->num_nodes();

    BuildControlFlow(graph, iteration_nodes_);

    done_cb_ = done;

    constexpr int64 iter_id = 0;
    constexpr int64 base_iter = 0;

    for (const Node* n : graph->nodes())
      if (!is_loop_node_[n->id()])
        Schedule(n, iter_id, base_iter, "");
  }

  void HPXExecutorState::SignalNodeDone()
  {
    std::unique_lock<std::mutex> lk(finish_mutex_);

    num_outstanding_ops_--;
    num_computed_ops_++;

    if (!was_finished) {
      if (num_outstanding_ops_ == 0 && num_computed_ops_ >= num_nodes_) {
        was_finished = true;
        Finish();
      }
    }
  }

  void HPXExecutorState::Process(const Node* node,
                                 EntryVector input_tensors,
                                 int64 iter_id,
                                 int64 base_iter,
                                 std::string const& key_prefix,
                                 bool is_dead,
                                 int64 scheduled_usec)
  {
    const GraphView& gview = impl_->gview_;

    if (!is_dead)
      for (auto& i : input_tensors)
        if (i.is_dead) {
          is_dead = true;
          break;
        }

    // Parameters passed to OpKernel::Compute.
    TensorValueVec inputs;
    DeviceContextVec input_device_contexts;
    AllocatorAttributeVec input_alloc_attrs;

    OpKernelContext::Params params;
    params.step_id = step_id_;
    Device* device = impl_->params_.device;
    params.frame_iter = FrameAndIter(0, iter_id);
    params.device = device;
    params.log_memory = log_memory_;
    params.record_tensor_accesses = impl_->device_record_tensor_accesses_;
    params.rendezvous = rendezvous_;
    params.session_state = session_state_;
    params.tensor_store = tensor_store_;
    params.cancellation_manager = cancellation_manager_;
    params.call_frame = call_frame_;
    params.function_library = impl_->params_.function_library;
    params.resource_manager = device->resource_manager();
    params.step_container = step_container_;
    params.slice_reader_cache = slice_reader_cache_;
    params.inputs = &inputs;
    params.input_device_contexts = &input_device_contexts;
    params.input_alloc_attrs = &input_alloc_attrs;
    params.runner = &runner_;

    Status s;
    NodeExecStats* stats = nullptr;
    EntryVector outputs;

    const int id = node->id();
    const NodeItem& item = *gview.node(id);

    // Set the device_context for this node id, if it exists.
    if (id < static_cast<int>(device_context_map_.size())) {
      params.op_device_context = device_context_map_[id];
    }

    params.track_allocations = false;
    stats = nullptr;
    if (stats_collector_ && !is_dead) {
      // track allocations if and only if we are collecting statistics
      params.track_allocations = true;
      stats = new NodeExecStats;
      stats->set_node_name(node->name());
      nodestats::SetScheduled(stats, scheduled_usec);
      nodestats::SetAllStart(stats);
    }

    // if beginning of loop, schedule all loop nodes
    if (IsEnter(node)) {
      base_iter = iter_id;

      for (const Node* n : iteration_nodes_[node->id()])
        if (!IsExit(n))
          Schedule(n,
                   iter_id,
                   base_iter,
                   key_prefix + std::to_string(iter_id) + ";");
    }

    // reschedule loop nodes for next iteration
    if (is_loop_node_[node->id()] && !IsExit(node) && !is_dead)
      Schedule(node, iter_id + 1, base_iter, key_prefix);

    if (vlog_) {
      VLOG(1) << "Process node: " << id << " step " << params.step_id << " "
              << SummarizeNodeDef(node->def());
    }

    Entry* first_input = input_tensors.data();

    outputs.clear();

    TensorReferenceVector accessed_tensors;
    DeviceContext* device_context = nullptr;

    bool launch_async = item.kernel_is_async;

    if (!is_dead || IsTransferNode(node)) {
      // Prepares inputs.
      bool is_input_dead = false;
      if (first_input) {
        s = PrepareInputs(item,
                          first_input,
                          &inputs,
                          &input_device_contexts,
                          &input_alloc_attrs,
                          &is_input_dead);

        if (!s.ok()) {
          // Clear inputs.
          int num_inputs = item.num_inputs;
          for (int i = 0; i < num_inputs; ++i) {
            (first_input + i)->ClearVal();
          }
        }
      }

      // Set up compute params.
      OpKernel* op_kernel = item.kernel;
      params.op_kernel = op_kernel;
      params.frame_iter = FrameAndIter(0, iter_id);
      params.is_input_dead = is_input_dead;
      params.output_attr_array = item.output_attrs();
      
  auto then = std::chrono::system_clock::now();

      if (launch_async) {
        AsyncOpKernel* async = item.kernel->AsAsync();
        DCHECK(async != nullptr);
        AsyncState* state =
            new AsyncState(params, is_dead, &item, first_input, stats);

        auto done = [this, state, iter_id, base_iter, key_prefix, then]() mutable {
          Device* device = impl_->params_.device;

          // Shorthands
          NodeExecStats* stats = state->stats;
          Entry* first_input = state->first_input;
          auto node = state->item->node;

          if (vlog_) {
            VLOG(2) << this
                    << " Async kernel done: " << SummarizeNodeDef(node->def());
          }

          if (stats)
            nodestats::SetOpEnd(stats);
          
          EntryVector outputs;
          Status s = ProcessOutputs(*state->item, &state->ctx, &outputs, stats);
          if (stats)
            nodestats::SetMemory(stats, &state->ctx);

          const int num_inputs = state->item->num_inputs;
          for (int i = 0; i < num_inputs; ++i) {
            (first_input + i)->ClearVal();
          }

          if (s.ok()) {
            PropagateOutputs(state->item,
                             std::move(outputs),
                             iter_id,
                             base_iter,
                             key_prefix,
                             state->is_dead);
          }
          outputs.clear();

          if (s.ok() && impl_->device_record_tensor_accesses_) {
            // Get the list of all tensors accessed during the execution
            TensorReferenceVector accessed;
            state->ctx.retrieve_accessed_tensors(&accessed);
            if (stats)
              nodestats::SetReferencedTensors(stats, accessed);
            // callee takes ownership of the vector
            device->ConsumeListOfAccessedTensors(state->ctx.op_device_context(),
                                                 accessed);
          }

          delete state;

          SignalNodeDone();
        };

        if (stats)
          nodestats::SetOpStart(stats);

        device->ComputeAsync(CHECK_NOTNULL(async), &state->ctx, done);
      } else {
        OpKernelContext ctx(&params, item.num_outputs);
        if (stats)
          nodestats::SetOpStart(stats);

        auto f = [device, op_kernel, &ctx](){
          device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
        };

        hpx::threads::run_as_os_thread(std::move(f)).get();

        if (stats)
          nodestats::SetOpEnd(stats);
          
        Status s = ProcessOutputs(item, &ctx, &outputs, stats);
        if (s.ok() && impl_->device_record_tensor_accesses_) {
          // Get the list of all tensors accessed during the execution
          ctx.retrieve_accessed_tensors(&accessed_tensors);
          device_context = ctx.op_device_context();
        }
        if (stats)
          nodestats::SetMemory(stats, &ctx);
      }
    } else {
      outputs.resize(node->num_outputs());
    }

    if (!launch_async) {
      const int num_inputs = item.num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->ClearVal();
      }

      if (s.ok()) {
        PropagateOutputs(
            &item, std::move(outputs), iter_id, base_iter, key_prefix, is_dead);
      }
      outputs.clear();

      if (!accessed_tensors.empty()) {
        if (stats)
          nodestats::SetReferencedTensors(stats, accessed_tensors);
        // device_context is set above in synchronous computes
        device->ConsumeListOfAccessedTensors(device_context, accessed_tensors);
      }
      if (stats) {
        scheduled_usec = nodestats::NowInUsec();
      }

      SignalNodeDone();
    }
  }

  Status
  HPXExecutorState::PrepareInputs(const NodeItem& item,
                                  Entry* first_input,
                                  TensorValueVec* inputs,
                                  DeviceContextVec* input_device_contexts,
                                  AllocatorAttributeVec* input_alloc_attrs,
                                  bool* is_input_dead)
  {
    const Node* node = item.node;

    inputs->clear();
    inputs->resize(item.num_inputs);
    input_device_contexts->clear();
    input_device_contexts->resize(item.num_inputs);
    input_alloc_attrs->clear();
    input_alloc_attrs->resize(item.num_inputs);

    *is_input_dead = false;

    bool is_merge = item.is_merge;
    for (int i = 0; i < item.num_inputs; ++i) {
      const bool expect_ref = IsRefType(item.input_type(i));
      Entry* entry = first_input + i;
      (*input_device_contexts)[i] = entry->device_context;
      (*input_alloc_attrs)[i] = entry->alloc_attr;

      // i-th input.
      TensorValue* inp = &(*inputs)[i];

      // Only merge and transfer nodes can have no-value inputs.
      if (!entry->has_value) {
        if (!is_merge) {
          DCHECK(IsTransferNode(node));
          DCHECK(!entry->val_field_is_set);
          entry->has_value = true;
          entry->val_field_is_set = true;
          entry->val.Init(*kEmptyTensor);
          inp->tensor = entry->val.get();
          *is_input_dead = true;
        }
        continue;
      }
      if (entry->ref == nullptr) {
        if (expect_ref) {
          return AttachDef(
              errors::InvalidArgument(i, "-th input expects a ref type"),
              item.kernel->def());
        }
        inp->tensor = entry->val.get();
      } else {
        if (!entry->ref->IsInitialized() && !IsInitializationOp(item.node)) {
          return AttachDef(errors::FailedPrecondition(
                               "Attempting to use uninitialized value ",
                               item.kernel->def().input(i)),
                           item.kernel->def());
        }
        if (expect_ref) {
          inp->mutex_if_ref = entry->ref_mu;
          inp->tensor = entry->ref;
        } else {
          // Automatically deref the tensor ref when the op expects a
          // tensor but is given a ref to a tensor.  Need to deref it
          // under the mutex.
          {
            mutex_lock l(*(entry->ref_mu));
            DCHECK(!entry->val_field_is_set);
            entry->val.Init(*entry->ref);
            entry->val_field_is_set = true;
          }
          entry->ref = nullptr;
          entry->ref_mu = nullptr;

          inp->tensor = entry->val.get();
        }
      }
    }
    return Status::OK();
  }

  Status HPXExecutorState::ProcessOutputs(const NodeItem& item,
                                          OpKernelContext* ctx,
                                          EntryVector* outputs,
                                          NodeExecStats* stats)
  {
    const Node* node = item.node;
    DCHECK_EQ(0, outputs->size());
    outputs->resize(item.num_outputs);

    Status s = ctx->status();
    if (!s.ok()) {
      s = AttachDef(s, item.kernel->def());
      // TODO(misard) Replace with a finer-grain enabling flag once we
      // add better optional debugging support.
      if (vlog_ && VLOG_IS_ON(1)) {
        LOG(WARNING) << this << " Compute status: " << s;
      }
      return s;
    }

    // Get the device_context for this node id, if it exists.
    DeviceContext* device_context = nullptr;
    if (node->id() < device_context_map_.size()) {
      device_context = device_context_map_[node->id()];
    }

    // Experimental: debugger (tfdb) access to intermediate node completion.
    if (item.num_outputs == 0 && impl_->params_.node_outputs_cb != nullptr) {
      // If the node has no output, invoke the callback with output slot set to
      // -1, signifying that this is a no-output node.
      s.Update(impl_->params_.node_outputs_cb(
          item.node->name(), -1, nullptr, false, ctx));
    }

    for (int i = 0; i < item.num_outputs; ++i) {
      TensorValue val = ctx->release_output(i);
      if (*ctx->is_output_dead() || val.tensor == nullptr) {
        // Unless it's a Switch or a Recv, the node must produce a
        // tensor value at i-th output.
        if (!IsSwitch(node) && !IsRecv(node)) {
          s.Update(errors::Internal("Missing ",
                                    i,
                                    "-th output from ",
                                    SummarizeNodeDef(node->def())));
        }
      } else {
        Entry* out = &((*outputs)[i]);

        // Set the device context of the output entry.
        out->device_context = device_context;

        // Set the allocator attributes of the output entry.
        out->alloc_attr = ctx->output_alloc_attr(i);

        // Sanity check of output tensor types.
        DataType dtype = val->dtype();
        if (val.is_ref())
          dtype = MakeRefType(dtype);
        if (dtype == item.output_type(i)) {
          if (stats && val.tensor->IsInitialized()) {
            nodestats::SetOutput(stats, i, val.tensor);
          }
          if (val.is_ref()) {
            out->has_value = true;
            out->ref = val.tensor;
            out->ref_mu = val.mutex_if_ref;
            if (log_memory_) {
              Tensor to_log;
              {
                // Dereference the tensor under the lock.
                mutex_lock l(*out->ref_mu);
                to_log = *out->ref;
              }
              LogMemory::RecordTensorOutput(
                  ctx->op_kernel().name(), ctx->step_id(), i, to_log);
            }

            // Experimental: debugger (tfdb) access to intermediate node
            // outputs.
            if (impl_->params_.node_outputs_cb != nullptr) {
              s.Update(impl_->params_.node_outputs_cb(
                  item.node->name(), i, out->ref, true, ctx));
            }
          } else {
            // NOTE that std::move is used here, so val.tensor goes to
            // uninitialized state (val.tensor->IsInitialized return false).
            DCHECK(!out->val_field_is_set);
            out->has_value = true;
            out->val_field_is_set = true;
            out->val.Init(std::move(*val.tensor));
            if (log_memory_) {
              LogMemory::RecordTensorOutput(
                  ctx->op_kernel().name(), ctx->step_id(), i, *out->val);
            }

            // Experimental: debugger access to intermediate node outputs.
            if (impl_->params_.node_outputs_cb != nullptr) {
              s.Update(impl_->params_.node_outputs_cb(
                  item.node->name(), i, out->val.get(), false, ctx));
            }
          }
        } else {
          s.Update(errors::Internal("Output ",
                                    i,
                                    " of type ",
                                    DataTypeString(dtype),
                                    " does not match declared output type ",
                                    DataTypeString(item.output_type(i)),
                                    " for node ",
                                    SummarizeNodeDef(node->def())));
        }
      }
      if (!val.is_ref()) {
        // If OpKernelContext returns outputs via pass-by-value, we
        // don't need this trouble.
        delete val.tensor;
      }
    }
    return s;
  }

  void HPXExecutorState::PropagateOutputs(const NodeItem* item,
                                          const EntryVector& outputs,
                                          int64 iter_id,
                                          const int64 base_iter,
                                          std::string key_prefix,
                                          const bool is_dead)
  {
    const Node* node = item->node;
    // each (nested) loop appends iteration of parent to the prefix
    if (IsEnter(node))
      key_prefix += std::to_string(iter_id) + ";";

    if (IsNextIteration(node))
      ++iter_id;

    // on exit of the (nested) loop, remove the last part of the prefix to send
    // data to parent loop
    if (IsExit(node)) {
      iter_id = base_iter;

      if (key_prefix.size() == numDigits(base_iter) + 1)
        key_prefix = "";
      else
        key_prefix =
            key_prefix.substr(0, key_prefix.size() - numDigits(iter_id) - 1);
    }

    auto const& gview = impl_->gview_;

    unsigned src_id = node->id();

    std::lock_guard<std::recursive_mutex> lk(map_mutex_);

    const int num_output_edges = item->num_output_edges;
    const EdgeInfo* edges = item->output_edge_list();
    for (int idx = 0; idx < num_output_edges; ++idx) {
      const EdgeInfo& e = edges[idx];
      unsigned dst_id = e.dst_id;

      std::string key = key_prefix + std::to_string(step_id_) + ";" +
                        std::to_string(iter_id) + ";" + std::to_string(src_id) +
                        ";" + std::to_string(dst_id);
      
      unsigned src_slot = e.output_slot;
      unsigned dst_slot = e.input_slot;
      const Node* dst = gview.node(dst_id)->node;
      bool is_control_edge = (src_slot == Graph::kControlSlot);

      if (!is_control_edge) {

        key += ";" + std::to_string(src_slot) + ";" + std::to_string(dst_slot);
      }

      Entry val;

      if (is_set_map_.count(key) == 0) {
        is_set_map_[key] = true;
        // if node is dead or has no valid output and destination is a loop
        // node,
        // the destination node is dead.
        if ((is_dead ||
             (!is_control_edge && !outputs[src_slot].has_value)) &&
            is_loop_node_[dst_id]) {
          // need to schedule exit node for cleanup of potentially nested loops
          if (IsExit(dst) && is_dead)
            Schedule(dst, iter_id, base_iter, key_prefix);

          if (!is_control_edge && outputs[src_slot].has_value)
          {
            if (e.is_last)
              val = std::move(outputs[src_slot]);
            else
              val = outputs[src_slot];
              
          }
          val.is_dead = true;
        } else if (!is_dead) {
          if (!is_control_edge) {
            // schedule exit of the loop
            if (IsExit(dst) && outputs[src_slot].has_value)
              Schedule(dst, iter_id, base_iter, key_prefix);

            if (e.is_last)
              val = std::move(outputs[src_slot]);
            else
              val = outputs[src_slot];
          }
        }

        promise_map_[key].set_value(std::move(val));
      }
    }
  }

  hpx::future<void> HPXExecutorState::Schedule(const Node* node,
                                               const int64 iter_id,
                                               const int64 base_iter,
                                               std::string const& key_prefix)
  {
    {
      std::lock_guard<std::mutex> lk(finish_mutex_);
      num_outstanding_ops_++;
    }

    int64 scheduled_usec = 0;
    if (stats_collector_) {
      scheduled_usec = nodestats::NowInUsec();
    }

    std::vector<hpx::lcos::shared_future<Entry> > input_futures(
        node->num_inputs());

    std::vector<hpx::lcos::shared_future<Entry> > control_futures;
    control_futures.reserve(node->in_edges().size() - node->num_inputs());

    unsigned dst_id = node->id();

    for (const Edge* e : node->in_edges()) {
      if (IsSource(e->src())) {
        control_futures.push_back(hpx::make_ready_future(Entry()));
        continue;
      }
      unsigned src_id = e->src()->id();

      std::string key = key_prefix + std::to_string(step_id_) + ";" +
                        std::to_string(iter_id) + ";" + std::to_string(src_id) +
                        ";" + std::to_string(dst_id);

      std::lock_guard<std::recursive_mutex> lk(map_mutex_);
      if (!e->IsControlEdge()) {
        unsigned src_slot = e->src_output();
        unsigned dst_slot = e->dst_input();

        key += ";" + std::to_string(src_slot) + ";" + std::to_string(dst_slot);

        if (future_map_.count(key) == 0)
          future_map_[key] = std::move(promise_map_[key].get_future().share());
        input_futures[dst_slot] = future_map_[key];
      } else {
        if (future_map_.count(key) == 0)
          future_map_[key] = std::move(promise_map_[key].get_future().share());
        control_futures.emplace_back(future_map_[key]);
      }
    }

    auto when_all_control_futures = hpx::when_all(control_futures);

    if (!IsMerge(node)) {
      auto when_all_input_futures = hpx::when_all(input_futures);

      return hpx::when_all(when_all_control_futures, when_all_input_futures)
          .then(hpx::launch::async,
                hpx::util::unwrapped([this,
                                      node,
                                      iter_id,
                                      base_iter,
                                      key_prefix,
                                      scheduled_usec](hpx::util::tuple<
                    hpx::future<std::vector<hpx::shared_future<Entry> > >,
                    hpx::future<std::vector<hpx::shared_future<Entry> > > >
                                                          data) {
                  auto inputs_futures = hpx::util::get<1>(data).get();

                  EntryVector inputs;
                  inputs.reserve(inputs_futures.size());

                  for (auto& f : inputs_futures)
                    inputs.push_back(std::move(f.get()));

                  bool is_dead = false;
                  auto control_futures = hpx::util::get<0>(data).get();
                  for (auto& f : control_futures)
                    if (f.get().is_dead) {
                      is_dead = true;
                      break;
                    }

                  Process(node,
                          std::move(inputs),
                          iter_id,
                          base_iter,
                          key_prefix,
                          is_dead,
                          scheduled_usec);
                }));
    } else {
      auto when_any_input_futures = hpx::when_any(input_futures);

      return hpx::when_all(when_all_control_futures, when_any_input_futures)
          .then(
               hpx::launch::async,
               hpx::util::unwrapped(
                   [this, node, iter_id, base_iter, key_prefix, scheduled_usec](
                       hpx::util::tuple<
                           hpx::future<
                               std::vector<hpx::lcos::shared_future<Entry> > >,
                           hpx::future<hpx::when_any_result<std::vector<
                               hpx::lcos::shared_future<Entry> > > > >
                           data) {
                     EntryVector inputs(node->num_inputs());

                     auto result = hpx::util::get<1>(data).get();

                     inputs[result.index] = result.futures[result.index].get();

                     bool is_dead = false;
                     auto control_futures = hpx::util::get<0>(data).get();
                     for (auto& f : control_futures)
                       if (f.get().is_dead) {
                         is_dead = true;
                         break;
                       }

                     Process(node,
                             std::move(inputs),
                             iter_id,
                             base_iter,
                             key_prefix,
                             is_dead,
                             scheduled_usec);
                   }));
    }
  }
  /*
  const Tensor* HPXExecutorState::GetTensorValueForDump(const Entry& input) {
    if (!input.has_value) {
      return kEmptyTensor;
    } else if (input.ref == nullptr) {
      return input.val.get();
    } else {
      return input.ref;
    }
  }*/

  void HPXExecutorState::Finish()
  {
    mu_.lock();

    auto status = status_;
    auto done_cb = std::move(done_cb_);
    auto runner = std::move(runner_);
    mu_.unlock();
    if (sync_on_finish_ && status.ok()) {
      // Block until the device has finished all queued operations. For
      // devices like GPUs that continue to execute Ops after their Compute
      // methods have completed, this ensures that control is not returned to
      // the user until the step (and its side-effects) has actually completed.
      status = impl_->params_.device->Sync();
    }
    delete this;
    CHECK(done_cb != nullptr);

    runner([=]() { done_cb(status); });
  }

  void HPXExecutorImpl::RunAsync(const Args& args, DoneCallback done)
  {
    auto state = new HPXExecutorState(args, this);

    state->RunAsync(std::move(done));
  }
} // end namespace

Status NewLocalHPXExecutor(const LocalExecutorParams& params,
                           const Graph* graph,
                           Executor** executor)
{
  HPXExecutorImpl* impl = new HPXExecutorImpl(params, graph);
  Status s = impl->Initialize();
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}

} // end namespace tensorflow
