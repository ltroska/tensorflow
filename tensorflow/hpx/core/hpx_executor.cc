/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/hpx/core/hpx_executor.h"

#include <hpx/util/unwrapped.hpp>
#include <hpx/lcos/when_all.hpp>
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


namespace tensorflow {
namespace {

// 1-D, 0 element tensor.
static const Tensor* const kEmptyTensor = new Tensor;

bool IsInitializationOp(const Node* node) {
  return node->op_def().allows_uninitialized_input();
}

// Sets the timeline_label field of *node_stats, using data from *node.
// Returns true iff the node is a transfer node.
// TODO(tucker): merge with the DetailText function in session.cc
// in a common location.
bool SetTimelineLabel(const Node* node, NodeExecStats* node_stats) {
  bool is_transfer_node = false;
  string memory;
  for (auto& all : node_stats->memory()) {
    int64 tot = all.total_bytes();
    if (tot >= 0.1 * 1048576.0) {
      int64 peak = all.peak_bytes();
      if (peak > 0) {
        memory =
            strings::StrCat(memory, "[", all.allocator_name(),
                            strings::Printf(" %.1fMB %.1fMB] ", tot / 1048576.0,
                                            peak / 1048576.0));
      } else {
        memory = strings::StrCat(memory, "[", all.allocator_name(),
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
    text = strings::StrCat(memory, def.name(), " = ", def.op(), "(",
                           tensor_name, " @", recv_device);
    is_transfer_node = true;
  } else if (IsRecv(node)) {
    string tensor_name;
    TF_CHECK_OK(GetNodeAttr(def, "tensor_name", &tensor_name));
    string send_device;
    TF_CHECK_OK(GetNodeAttr(def, "send_device", &send_device));
    text = strings::StrCat(memory, def.name(), " = ", def.op(), "(",
                           tensor_name, " @", send_device);
    is_transfer_node = true;
  } else {
    text = strings::StrCat(
        memory, def.name(), " = ", def.op(), "(",
        str_util::Join(
            std::vector<StringPiece>(def.input().begin(), def.input().end()),
            ", "),
        ")");
  }
  node_stats->set_timeline_label(text);
  return is_transfer_node;
}

// Helper routines for collecting step stats.
namespace nodestats {
inline int64 NowInUsec() { return Env::Default()->NowMicros(); }

void SetScheduled(NodeExecStats* nt, int64 t) { nt->set_scheduled_micros(t); }

void SetAllStart(NodeExecStats* nt) { nt->set_all_start_micros(NowInUsec()); }

void SetOpStart(NodeExecStats* nt) {
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_op_start_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetOpEnd(NodeExecStats* nt) {
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_op_end_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetAllEnd(NodeExecStats* nt) {
  DCHECK_NE(nt->all_start_micros(), 0);
  nt->set_all_end_rel_micros(NowInUsec() - nt->all_start_micros());
}

void SetOutput(NodeExecStats* nt, int slot, const Tensor* v) {
  DCHECK(v);
  NodeOutput* no = nt->add_output();
  no->set_slot(slot);
  v->FillDescription(no->mutable_tensor_description());
}

void SetMemory(NodeExecStats* nt, OpKernelContext* ctx) {
  for (const auto& allocator_pair : ctx->wrapped_allocators()) {
    AllocatorMemoryUsed* memory = nt->add_memory();
    // retrieving the sizes from the wrapped allocator removes the
    // executor's reference to it, so allocator_pair.second must not
    // be dereferenced again after this statement
    auto sizes = allocator_pair.second->GetSizesAndUnRef();
    memory->set_allocator_name(allocator_pair.first->Name());
    int tb = sizes.first;
    memory->set_total_bytes(tb);
    if (allocator_pair.first->TracksAllocationSizes()) {
      memory->set_peak_bytes(sizes.second);
    }
  }
}

void SetReferencedTensors(NodeExecStats* nt,
                          const TensorReferenceVector& tensors) {
  // be careful not to increment the reference count on any tensor
  // while recording the information
  for (size_t i = 0; i < tensors.size(); ++i) {
    AllocationDescription* description = nt->add_referenced_tensor();
    tensors.at(i).FillDescription(description);
  }
}

}  // namespace nodestats

struct NodeItem {
  // A graph node.
  const Node* node = nullptr;

  // The kernel for this node.
  OpKernel* kernel = nullptr;

  bool kernel_is_expensive = false;  // True iff kernel->IsExpensive()
  bool kernel_is_async = false;      // True iff kernel->AsAsync() != nullptr
  bool is_merge = false;             // True iff IsMerge(node)

  // Cached values of node->num_inputs() and node->num_outputs(), to
  // avoid levels of indirection.
  int num_inputs;
  int num_outputs;

  // HPXExecutorImpl::output_attrs_[output_attr_start] is the 1st
  // positional attribute for the 0th output of this node.
  int output_attr_start = 0;

  DataType input_type(int i) const {
    DCHECK_LT(i, num_inputs);
    return (i < 4) ? inlined_input_type[i] : node->input_type(i);
  }
  DataType output_type(int i) const {
    DCHECK_LT(i, num_outputs);
    return (i < 4) ? inlined_output_type[i] : node->output_type(i);
  }
  // Cache first 4 input and output types to reduce levels of indirection
  DataType inlined_input_type[4];
  DataType inlined_output_type[4];
};

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

class HPXExecutorImpl : public Executor {
 public:
  HPXExecutorImpl(const LocalExecutorParams& p, const Graph* g)
      : params_(p), graph_(g) {
    CHECK(p.create_kernel != nullptr);
    CHECK(p.delete_kernel != nullptr);
  }

  ~HPXExecutorImpl() override {
    for (int i = 0; i < graph_->num_node_ids(); i++) {
      params_.delete_kernel(nodes_[i].kernel);
    }

    delete[] nodes_;
    delete graph_;
  }

  Status Initialize();

  // Infer memory allocation attributes of a node n's output,
  // based on its use node dst.  Note that dst might not be directly
  // connected to n by a single edge, but might be a downstream
  // consumer of n's output by reference.  *attr is updated with any
  // necessary attributes.
  Status InferAllocAttr(const Node* n, const Node* dst,
                        const DeviceNameUtils::ParsedName& local_dev_name,
                        AllocatorAttributes* attr);

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
  NodeItem* nodes_ = nullptr;     // array of size "graph_.num_node_ids()"
  int total_output_tensors_ = 0;  // == sum(nodes_[*].num_outputs())
  int total_input_tensors_ = 0;  // == sum(nodes_[*].num_outputs())

  // A cached value of params_
  bool device_record_tensor_accesses_ = false;

  std::vector<AllocatorAttributes> output_attrs_;
  
  TF_DISALLOW_COPY_AND_ASSIGN(HPXExecutorImpl);
};

Status HPXExecutorImpl::Initialize() {
  const int num_nodes = graph_->num_node_ids();
  delete[] nodes_;
  nodes_ = new NodeItem[num_nodes];

  total_output_tensors_ = 0;
  total_input_tensors_ = 0;

  // Cache this value so we make this virtual function call once, rather
  // that O(# steps * # nodes per step) times.
  device_record_tensor_accesses_ =
      params_.device->RequiresRecordingAccessedTensors();

  // Preprocess every node in the graph to create an instance of op
  // kernel for each node.
  for (const Node* n : graph_->nodes()) {
    const int id = n->id();

    NodeItem* item = &nodes_[id];
    item->node = n;
    item->num_inputs = n->num_inputs();
    item->num_outputs = n->num_outputs();

    for (int i = 0; i < std::min(4, item->num_inputs); i++) {
      item->inlined_input_type[i] = n->input_type(i);
    }
    for (int i = 0; i < std::min(4, item->num_outputs); i++) {
      item->inlined_output_type[i] = n->output_type(i);
    }

    total_input_tensors_ += n->num_inputs();

    item->output_attr_start = total_output_tensors_;
    total_output_tensors_ += n->num_outputs();

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
  }
  
  return SetAllocAttrs();
}

Status HPXExecutorImpl::SetAllocAttrs() {
  Status s;
  Device* device = params_.device;
  DeviceNameUtils::ParsedName local_dev_name = device->parsed_name();

  output_attrs_.resize(total_output_tensors_);
  for (const Node* n : graph_->nodes()) {
    NodeItem* item = &nodes_[n->id()];
    const int base_index = item->output_attr_start;
    // Examine the out edges of each node looking for special use
    // cases that may affect memory allocation attributes.
    for (auto e : n->out_edges()) {
      const int index = e->src_output();
      AllocatorAttributes attr;
      s = InferAllocAttr(n, e->dst(), local_dev_name, &attr);
      if (!s.ok()) return s;
      if (attr.value != 0) {
        if (!e->IsControlEdge()) {
          output_attrs_[base_index + index].Merge(attr);
        }
      }
    }

    for (int out = 0; out < n->num_outputs(); out++) {
      OpKernel* op_kernel = item->kernel;
      DCHECK_LT(out, op_kernel->output_memory_types().size());
      bool on_host = op_kernel->output_memory_types()[out] == HOST_MEMORY;
      AllocatorAttributes h;
      h.set_on_host(on_host);
      output_attrs_[base_index + out].Merge(h);
    }
  }
  return s;
}

Status HPXExecutorImpl::InferAllocAttr(
    const Node* n, const Node* dst,
    const DeviceNameUtils::ParsedName& local_dev_name,
    AllocatorAttributes* attr) {
  Status s;
  // Note that it's possible for *n to be a Recv and *dst to be a Send,
  // so these two cases are not mutually exclusive.
  if (IsRecv(n)) {
    string src_name;
    s = GetNodeAttr(n->def(), "send_device", &src_name);
    if (!s.ok()) return s;
    DeviceNameUtils::ParsedName parsed_src_name;
    if (!DeviceNameUtils::ParseFullName(src_name, &parsed_src_name)) {
      s = errors::Internal("Bad send_device attr '", src_name, "' in node ",
                           n->name());
      return s;
    }
    if (!DeviceNameUtils::IsSameAddressSpace(parsed_src_name, local_dev_name)) {
      // Value is going to be the sink of an RPC.
      attr->set_nic_compatible(true);
      VLOG(2) << "node " << n->name() << " is the sink of an RPC in";
    } else if ((local_dev_name.type == "CPU" || n->IsHostRecv()) &&
               parsed_src_name.type != "CPU") {
      // Value is going to be the sink of a local DMA from GPU to CPU (or other
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
    if (!s.ok()) return s;
    DeviceNameUtils::ParsedName parsed_dst_name;
    if (!DeviceNameUtils::ParseFullName(dst_name, &parsed_dst_name)) {
      s = errors::Internal("Bad recv_device attr '", dst_name, "' in node ",
                           n->name());
      return s;
    }
    if (!DeviceNameUtils::IsSameAddressSpace(parsed_dst_name, local_dev_name)) {
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
  } else if (dst->type_string() == "ToFloat") {
    for (auto e : dst->out_edges()) {
      s = InferAllocAttr(n, e->dst(), local_dev_name, attr);
      if (!s.ok()) return s;
    }
  }
  return s;
}

// The state associated with one invocation of HPXExecutorImpl::Run.
// HPXExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class HPXExecutorState {
 public:
  HPXExecutorState(const Executor::Args& args, HPXExecutorImpl* impl);
  ~HPXExecutorState();

  void RunAsync(Executor::DoneCallback done);

 private:
  // Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
  // TODO(yuanbyu): A better way to do "has_value"?
  struct Entry {
    Entry() {}
    Entry(const Entry& other)
        : ref(other.ref),
          ref_mu(other.ref_mu),
          has_value(other.has_value),
          val_field_is_set(other.val_field_is_set),
          alloc_attr(other.alloc_attr),
          device_context(other.device_context) {
      if (val_field_is_set) {
        val.Init(*other.val);
      }
    }
    ~Entry() {
      if (val_field_is_set) val.Destroy();
    }

    Entry& operator=(const Entry& other) {
      if (val_field_is_set) {
        val.Destroy();
      }
      ref = other.ref;
      ref_mu = other.ref_mu;
      has_value = other.has_value;
      val_field_is_set = other.val_field_is_set;
      alloc_attr = other.alloc_attr;
      device_context = other.device_context;
      if (val_field_is_set) {
        val.Init(*other.val);
      }
      return *this;
    }

    // Clears the <val> field.
    void ClearVal() {
      if (val_field_is_set) {
        val.Destroy();
        val_field_is_set = false;
      }
    }

    // A tensor value, if val_field_is_set.
    ManualConstructor<Tensor> val;

    Tensor* ref = nullptr;    // A tensor reference.
    mutex* ref_mu = nullptr;  // mutex for *ref if ref is not nullptr.

    // Whether the value exists, either in <val> or <ref>.
    bool has_value = false;

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
  
  struct promise_map
  {
    
    hpx::lcos::promise<Entry>& operator[](std::string key)
    {
      return map_[key];
    }
    
    std::unordered_map<std::string, hpx::lcos::promise<Entry> > map_;
    
    void clear()
    {
      map_.clear();
    }
  };
  
  std::unordered_map<std::string, hpx::lcos::promise<Entry> > sync_map_;

  typedef gtl::InlinedVector<Entry, 4> EntryVector;

  const bool vlog_;  // true if VLOG_IS_ON(1). Used to check vlog cheaply.

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

  // Owned.

  // A flag that is set on error after the frame state has been
  // dumped for diagnostic purposes.
  bool dumped_on_error_ = false;

  // Invoked when the execution finishes.
  Executor::DoneCallback done_cb_;

  std::atomic_int_fast32_t num_outstanding_ops_;

  mutex mu_;
  Status status_ GUARDED_BY(mu_);

  // Process a ready node in current thread.
  void Process(const Node* node, Entry* inputs, int64 scheduled_usec);

  // Before invoking item->kernel, fills in its "inputs".
  Status PrepareInputs(const NodeItem& item, Entry* first_input,
                       TensorValueVec* inputs,
                       DeviceContextVec* input_device_contexts,
                       AllocatorAttributeVec* input_alloc_attrs,
                       bool* is_input_dead);

  // After item->kernel computation is done, processes its outputs.
  Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                        EntryVector* outputs, NodeExecStats* stats);

  // After processing the outputs, propagates the outputs to their dsts.
  void PropagateOutputs(const NodeItem& node_item,
                        const EntryVector& outputs);



  // Schedule all the expensive nodes in 'ready', and put all the inexpensive
  // nodes in 'ready' into 'inline_ready'.
  hpx::future<void> Schedule(const Node* node);

  const Tensor* GetTensorValueForDump(const Entry& input);

  // Clean up when this executor is done.
  void Finish();
};

HPXExecutorState::HPXExecutorState(const Executor::Args& args, HPXExecutorImpl* impl)
    : vlog_(VLOG_IS_ON(1)),
      log_memory_(LogMemory::IsEnabled()),
      step_id_(args.step_id),
      rendezvous_(args.rendezvous),
      session_state_(args.session_state),
      tensor_store_(args.tensor_store),
      step_container_(args.step_container),
      stats_collector_(args.stats_collector),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      impl_(impl),
      cancellation_manager_(args.cancellation_manager),
      runner_(args.runner),
      sync_on_finish_(args.sync_on_finish),
      num_outstanding_ops_(0) {
        sync_map_.clear();
}

HPXExecutorState::~HPXExecutorState() {
  for (auto it : device_context_map_) {
    it->Unref();
  }
  delete slice_reader_cache_;
}

void HPXExecutorState::RunAsync(Executor::DoneCallback done) {
  const Graph* graph = impl_->graph_;

  // Ask the device to fill in the device context map.
  Device* device = impl_->params_.device;
  Status fill_status = device->FillContextMap(graph, &device_context_map_);
  if (!fill_status.ok()) {
    done(fill_status);
    return;

  }

  num_outstanding_ops_ = graph->num_node_ids();
  done_cb_ = done;
  // Schedule to run all the ready ops in thread pool.
  std::vector<hpx::future<void> > comp_futures;
  comp_futures.reserve(num_outstanding_ops_);
  
  for (const Node* n : graph->nodes())
      comp_futures.push_back(Schedule(n));

  hpx::when_all(comp_futures).then(
  [this](hpx::future<std::vector<hpx::future<void> > >)
  {
    Finish();
  });
}

void HPXExecutorState::Process(const Node* node, Entry* input_tensors, int64 scheduled_usec) {
  bool is_dead = false;
  
  const NodeItem* nodes = impl_->nodes_;

  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs ;
  DeviceContextVec input_device_contexts;
  AllocatorAttributeVec input_alloc_attrs;

  OpKernelContext::Params params;
  params.step_id = step_id_;
  Device* device = impl_->params_.device;
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
  bool completed = false;
  
  const int id = node->id();
  const NodeItem& item = nodes[id];

  // Set the device_context for this node id, if it exists.
  if (id < device_context_map_.size()) {
    params.op_device_context = device_context_map_[id];
  }

  params.track_allocations = false;
  stats = nullptr;
  if (stats_collector_) {
    // track allocations if and only if we are collecting statistics
    params.track_allocations = true;
    stats = new NodeExecStats;
    stats->set_node_name(node->name());
    nodestats::SetScheduled(stats, scheduled_usec);
    nodestats::SetAllStart(stats);
  }
                
  if (vlog_) {
    VLOG(1) << "Process node: " << id << " step " << params.step_id << " "
            << SummarizeNodeDef(node->def());
  }

  Entry* first_input = input_tensors;// + item.input_start;
  outputs.clear();

  TensorReferenceVector accessed_tensors;
  DeviceContext* device_context = nullptr;
  // Only execute this node if it is not dead or it is a send/recv
  // transfer node. For transfer nodes, we need to propagate the "dead"
  // bit even when the node is dead.
  bool launched_asynchronously = false;
  if (is_dead && !IsTransferNode(node)) {
    outputs.resize(item.num_outputs);
  } else {
    // Prepares inputs.
    bool is_input_dead = false;
    if (first_input)
    {
      s = PrepareInputs(item, first_input, &inputs, &input_device_contexts,
                        &input_alloc_attrs, &is_input_dead);
      if (!s.ok()) {
        // Clear inputs.
        int num_inputs = item.num_inputs;
        for (int i = 0; i < num_inputs; ++i) {
          (first_input + i)->ClearVal();
        }
       // MaybeMarkCompleted(input_frame, input_iter, id);
        // Continue to process the nodes in 'inline_ready'.
       // completed = NodeDone(s, item.node, ready, stats, &inline_ready);
      }
    }


    // Set up compute params.
    OpKernel* op_kernel = item.kernel;
    params.op_kernel = op_kernel;
    //params.frame_iter = FrameAndIter(input_frame->frame_id, input_iter);
    params.frame_iter = FrameAndIter(0, 0);
    
   // params.is_input_dead = is_input_dead;
    params.output_attr_array =
        gtl::vector_as_array(&impl_->output_attrs_) + item.output_attr_start;

    // Synchronous computes.
    OpKernelContext ctx(&params, item.num_outputs);
        
    Status s;
            
    if (stats) nodestats::SetOpStart(stats);
    device->Compute(CHECK_NOTNULL(op_kernel), &ctx);
    
    if (stats) nodestats::SetOpEnd(stats);
    s = ProcessOutputs(item, &ctx, &outputs, stats);
    if (s.ok() && impl_->device_record_tensor_accesses_) {
      // Get the list of all tensors accessed during the execution
      ctx.retrieve_accessed_tensors(&accessed_tensors);
      device_context = ctx.op_device_context();
    }
    if (stats) nodestats::SetMemory(stats, &ctx);
    const int num_inputs = item.num_inputs;
    for (int i = 0; i < num_inputs; ++i) {
      (first_input + i)->ClearVal();
    }
   // MaybeMarkCompleted(input_frame, input_iter, id);
    // Propagates outputs.
    if (s.ok()) {
      PropagateOutputs(item, outputs);
    }
    outputs.clear();
    if (!accessed_tensors.empty()) {
      if (stats) nodestats::SetReferencedTensors(stats, accessed_tensors);
      // device_context is set above in synchronous computes
      device->ConsumeListOfAccessedTensors(device_context, accessed_tensors);
    }
    if (stats) {
      //scheduled_usec = nodestats::NowInUsec();
    }
  }
  
  num_outstanding_ops_--;
}

Status HPXExecutorState::PrepareInputs(const NodeItem& item, Entry* first_input,
                                    TensorValueVec* inputs,
                                    DeviceContextVec* input_device_contexts,
                                    AllocatorAttributeVec* input_alloc_attrs,
                                    bool* is_input_dead) {
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
        return AttachDef(
            errors::FailedPrecondition("Attempting to use uninitialized value ",
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

Status HPXExecutorState::ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                                     EntryVector* outputs,
                                     NodeExecStats* stats) {
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
    impl_->params_.node_outputs_cb(item.node->name(), -1, nullptr, false, ctx);
  }

  for (int i = 0; i < item.num_outputs; ++i) {
    TensorValue val = ctx->release_output(i);
    if (*ctx->is_output_dead() || val.tensor == nullptr) {
      // Unless it's a Switch or a Recv, the node must produce a
      // tensor value at i-th output.
      if (!IsSwitch(node) && !IsRecv(node)) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
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
      if (val.is_ref()) dtype = MakeRefType(dtype);
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
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, to_log);
          }

          // Experimental: debugger (tfdb) access to intermediate node outputs.
          if (impl_->params_.node_outputs_cb != nullptr) {
            impl_->params_.node_outputs_cb(item.node->name(), i, out->ref, true,
                                           ctx);
          }
        } else {
          // NOTE that std::move is used here, so val.tensor goes to
          // uninitialized state (val.tensor->IsInitialized return false).
          DCHECK(!out->val_field_is_set);
          out->has_value = true;
          out->val_field_is_set = true;
          out->val.Init(std::move(*val.tensor));
          if (log_memory_) {
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, *out->val);
          }

          // Experimental: debugger access to intermediate node outputs.
          if (impl_->params_.node_outputs_cb != nullptr) {
            impl_->params_.node_outputs_cb(item.node->name(), i, out->val.get(),
                                           false, ctx);
          }
        }
      } else {
        s.Update(errors::Internal("Output ", i, " of type ",
                                  DataTypeString(dtype),
                                  " does not match declared output type ",
                                  DataTypeString(item.output_type(i)),
                                  " for node ", SummarizeNodeDef(node->def())));
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

void HPXExecutorState::PropagateOutputs(const NodeItem& node_item,
                                     const EntryVector& outputs) {
  const Node* node = node_item.node;

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.


  if (IsEnter(node)) {
    /*bool is_constant;
    Status s = GetNodeAttr(node->def(), "is_constant", &is_constant);
    DCHECK(s.ok()) << s;
    FindOrCreateChildFrame(input_frame, input_iter, node, &output_frame);
    output_iter = 0;
    {
      mutex_lock l(output_frame->mu);
      if (is_constant) {
        // Propagate to all active iterations if this is a loop invariant.
        output_frame->AddLoopInv(node, outputs[0], ready);
      } else {
        output_frame->ActivateNodes(node, is_dead, output_iter, outputs, ready);
      }
      output_frame->num_pending_inputs--;
    }
    is_frame_done = input_frame->DecrementOutstandingOps(input_iter, ready);*/
  } else if (IsExit(node)) {
    /*if (is_dead) {
      mutex_lock l(input_frame->mu);
      // Stop and remember this node if it is a dead exit.
      if (input_iter == input_frame->iteration_count) {
        input_frame->dead_exits.push_back(node);
      }
      input_frame->GetIteration(input_iter)->outstanding_ops--;
      is_frame_done = input_frame->CleanupIterations(input_iter, ready);
    } else {
      output_frame = input_frame->parent_frame;
      output_iter = input_frame->parent_iter;
      {
        mutex_lock l(output_frame->mu);
        output_frame->ActivateNodes(node, is_dead, output_iter, outputs, ready);
      }
      is_frame_done = input_frame->DecrementOutstandingOps(input_iter, ready);
    }*/
  } else {
    //mutex_lock l(input_frame->mu);
    /*if (IsNextIteration(node)) {
      if (is_dead) {
        // Stop the deadness propagation.
        output_frame = nullptr;
      } else {
        if (input_iter == input_frame->iteration_count &&
            input_frame->num_outstanding_iterations ==
                input_frame->max_parallel_iterations) {
          // Reached the maximum for parallel iterations.
          input_frame->next_iter_roots.push_back({node, outputs[0]});
          output_frame = nullptr;
        } else {
          // If this is a new iteration, start it.
          if (input_iter == input_frame->iteration_count) {
            input_frame->IncrementIteration(ready);
          }
          output_iter = input_iter + 1;
        }
      }
    }*/
  
    // This is the case when node is not Enter, Exit, or NextIteration.
    unsigned src_id = node->id();
    for (const Edge* e : node->out_edges()) {
      unsigned dst_id = e->dst()->id();

      std::string key = std::to_string(step_id_) + ";"
        + std::to_string(src_id) + ";" + std::to_string(dst_id);
          
      if (sync_map_.count(key) == 0)
      {
        if (e->IsControlEdge())
        {
          sync_map_[key].set_value(Entry());
        }
        else
        {
          unsigned src_slot = e->src_output();
          unsigned dst_slot = e->dst_input();
          
          key += ";" + std::to_string(src_slot) + ";" + std::to_string(dst_slot);
          
          sync_map_[key].set_value(outputs[src_slot]);      
        }
      } 
    }
  }
}


hpx::future<void> HPXExecutorState::Schedule(const Node* node) {            
  int64 scheduled_usec = 0;
  if (stats_collector_) {
    scheduled_usec = nodestats::NowInUsec();
  }
  std::vector<hpx::lcos::future<Entry> > input_futures(node->num_inputs());
  
  unsigned dst_id = node->id();
  unsigned last_used = node->num_inputs();
  for (const Edge* e : node->in_edges())
  {        
    unsigned src_id = e->src()->id();
    std::string key = std::to_string(step_id_) + ";" +
      std::to_string(src_id) + ";" + std::to_string(dst_id);
      
    if (e->IsControlEdge())
    {
    //  input_futures[--last_used] = sync_map_[key].get_future(); 
    }
    else
    {
      unsigned src_slot = e->src_output();
      unsigned dst_slot = e->dst_input();
                
      key += ";" + std::to_string(src_slot) + ";" + std::to_string(dst_slot);
      input_futures[dst_slot] = sync_map_[key].get_future();      
    }
    
  }

  return hpx::when_all(input_futures).then(
  hpx::util::unwrapped2(
    [this, node, scheduled_usec](std::vector<Entry> inputs)
    {
      Process(node, inputs.data(), scheduled_usec);
    }
  ));
}

const Tensor* HPXExecutorState::GetTensorValueForDump(const Entry& input) {
  if (!input.has_value) {
    return kEmptyTensor;
  } else if (input.ref == nullptr) {
    return input.val.get();
  } else {
    return input.ref;
  }
}

void HPXExecutorState::Finish() {
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

void HPXExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  auto state = new HPXExecutorState(args, this);
  
  auto f = std::bind(&HPXExecutorState::RunAsync, state, done);
    
  hpx::threads::run_as_hpx_thread(f);        
}
}  // end namespace

Status NewLocalHPXExecutor(const LocalExecutorParams& params, const Graph* graph,
                        Executor** executor) {
  HPXExecutorImpl* impl = new HPXExecutorImpl(params, graph);
  Status s = impl->Initialize();
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}

}  // end namespace tensorflow
