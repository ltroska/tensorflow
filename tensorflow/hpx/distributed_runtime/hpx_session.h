#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_SESSION_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_SESSION_H_

#include "tensorflow/hpx/hpx_global_runtime.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow
{

class MasterInterface;

// A Session instance lets the caller drive a TensorFlow graph
// computation on potentially remote sets of devices. This is a thin
// wrapper around tensorflow::grpc::MasterService.
//
// Multiple threads must synchronize their accesses to a single
// session.
class HPXSession : public Session
{
  protected:
  explicit HPXSession(const SessionOptions& options);

  public:
  static Status Create(const SessionOptions& options,
                       std::unique_ptr<HPXSession>* out_session);
  // Resets the resource containers.
  static Status Reset(const SessionOptions& options,
                      const std::vector<string>& containers);

  ~HPXSession() override;

  global_runtime* GetRuntime()
  {
    return &init_;
  }

  // Creates a session with the "target". The session carries out
  // the graph computation defined by "graph", and will have version
  // number "initial_version".
  Status Create(const GraphDef& graph) override;
  Status Create(const RunOptions& run_options, const GraphDef& graph) override;

  // Runs with and without RunOptions.
  Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override;
  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs,
             RunMetadata* run_metadata) override;

  Status Extend(const GraphDef& graph) override;
  Status Extend(const RunOptions& run_options, const GraphDef& graph) override;

  Status Close() override;

  // NOTE: This API is still experimental and may change.
  ::tensorflow::Status PRunSetup(const std::vector<string>& input_names,
                                 const std::vector<string>& output_names,
                                 const std::vector<string>& target_nodes,
                                 string* handle) override;

  // NOTE: This API is still experimental and may change.
  ::tensorflow::Status
  PRun(const string& handle,
       const std::vector<std::pair<string, Tensor> >& inputs,
       const std::vector<string>& output_names,
       std::vector<Tensor>* outputs) override;

  std::vector<DeviceAttributes> ListDevices();

  protected:
  // Takes ownership of `*master`.
  void SetRemoteMaster(std::unique_ptr<MasterInterface> master);

  private:
  global_runtime init_;

  std::string target_;

  SessionOptions options_;
  std::unique_ptr<MasterInterface> master_;
  mutex mu_;

  // handle_ returned by the master to identify this session.
  string handle_ GUARDED_BY(mu_);

  // The current version of the graph.
  int64 current_graph_version_ GUARDED_BY(mu_);

  Status RunHelper(const RunOptions& run_options,
                   const std::vector<std::pair<string, Tensor> >& inputs,
                   const std::vector<string>& output_tensor_names,
                   const std::vector<string>& target_node_names,
                   std::vector<Tensor>* outputs,
                   RunMetadata* run_metadata,
                   const string& prun_handle);

  Status RunProto(CallOptions* call_options,
                  MutableRunStepRequestWrapper* req,
                  MutableRunStepResponseWrapper* resp);

  // Implementations for all the public interfaces.
  Status CreateImpl(CallOptions* call_options, const GraphDef& graph);
  Status ExtendImpl(CallOptions* call_options, const GraphDef& graph);

  TF_DISALLOW_COPY_AND_ASSIGN(HPXSession);
};

} // namespace tensorflow

#endif // TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_SESSION_H_
