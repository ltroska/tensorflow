#include "tensorflow/hpx/distributed_runtime/hpx_session.h"
#include "tensorflow/hpx/distributed_runtime/hpx_master.h"

#include <unordered_map>

#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow
{

namespace
{
  const char* kSchemePrefix = "hpx://";
  const size_t kSchemePrefixLength = strlen(kSchemePrefix);
} // namespace

HPXSession::HPXSession(const SessionOptions& options)
    : options_(options)
    , current_graph_version_(-1)
{
  auto hostname_and_port = options.target.substr(kSchemePrefixLength);
  const std::vector<string> hostname_port =
      str_util::Split(hostname_and_port, ':');

  init_.start("localhost", "7100", hostname_port[0], hostname_port[1]);
}

HPXSession::~HPXSession()
{
  master_.reset();
  init_.stop_and_shutdown();
}

/* static */
Status HPXSession::Create(const SessionOptions& options,
                          std::unique_ptr<HPXSession>* out_session)
{

  std::unique_ptr<HPXSession> ret(new HPXSession(options));
  std::unique_ptr<MasterInterface> master;
  // For testing, we enable the client to disable the use of the local
  // master registry, so that the RPC stack is exercised.
  if (!options.config.rpc_options().use_rpc_for_inprocess_master()) {
    master = LocalMaster::Lookup(options.target);
  }

  if (!master) {
    master.reset(NewHPXMaster(ret->GetRuntime(),
                              options.target.substr(kSchemePrefixLength)));
  }
  ret->SetRemoteMaster(std::move(master));
  *out_session = std::move(ret);
  return Status::OK();
}

namespace
{
  // Re-encodes constant represented in tensor proto into
  // tensor_content, which is slightly better (less copies and lower peak
  // memory usage) when used with rpc subsystems.
  void ReEncodeConsts(GraphDef* gdef)
  {
    for (NodeDef& ndef : *(gdef->mutable_node())) {
      if (ndef.op() == "Const") {
        TensorProto* proto = nullptr;
        for (auto& attr : *ndef.mutable_attr()) {
          if (attr.first == "value") {
            proto = attr.second.mutable_tensor();
          }
        }
        if (proto != nullptr && proto->tensor_content().empty() &&
            proto->ByteSize() > 64) {
          // If the constant is encoded with repeated proto fields and
          // it is moderate large, we re-encode it in tensor_content as
          // a Cord. This is mildly helpful for reducing the peak memory
          // usage on the server side where GraphDef/NodeDef are copied
          // quite often.
          Tensor parsed(proto->dtype());
          if (parsed.FromProto(*proto)) {
            parsed.AsProtoTensorContent(proto);
          }
        }
      }
    }
  }
} // namespace

Status HPXSession::CreateImpl(CallOptions* call_options, const GraphDef& graph)
{
  {
    mutex_lock l(mu_);
    if (!handle_.empty()) {
      return errors::InvalidArgument("A session is alive.");
    }
  }
  CreateSessionRequest req;
  *req.mutable_config() = options_.config;
  *req.mutable_graph_def() = graph;
  ReEncodeConsts(req.mutable_graph_def());
  CreateSessionResponse resp;
  Status s = master_->CreateSession(call_options, &req, &resp);
  if (s.ok()) {
    mutex_lock l(mu_);
    swap(handle_, *(resp.mutable_session_handle()));
    current_graph_version_ = resp.graph_version();
  }
  return s;
}

Status HPXSession::Create(const GraphDef& graph)
{
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return CreateImpl(&call_options, graph);
}

Status HPXSession::Create(const RunOptions& run_options, const GraphDef& graph)
{
  CallOptions call_options;
  call_options.SetTimeout(run_options.timeout_in_ms());
  return CreateImpl(&call_options, graph);
}

Status HPXSession::ExtendImpl(CallOptions* call_options, const GraphDef& graph)
{
  bool handle_is_empty;
  {
    mutex_lock l(mu_);
    handle_is_empty = handle_.empty();
  }
  if (handle_is_empty) {
    // Session was unitialized, so simply initialize the session with 'graph'.
    return Create(graph);
  }
  mutex_lock l(mu_);
  ExtendSessionRequest req;
  req.set_session_handle(handle_);
  *req.mutable_graph_def() = graph;
  req.set_current_graph_version(current_graph_version_);
  ExtendSessionResponse resp;
  Status s = master_->ExtendSession(call_options, &req, &resp);
  if (s.ok()) {
    current_graph_version_ = resp.new_graph_version();
  }
  return s;
}

Status HPXSession::Extend(const GraphDef& graph)
{
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  return ExtendImpl(&call_options, graph);
}

Status HPXSession::Extend(const RunOptions& run_options, const GraphDef& graph)
{
  CallOptions call_options;
  call_options.SetTimeout(run_options.timeout_in_ms());
  return ExtendImpl(&call_options, graph);
}

Status
HPXSession::RunHelper(const RunOptions& run_options,
                      const std::vector<std::pair<string, Tensor> >& inputs,
                      const std::vector<string>& output_tensor_names,
                      const std::vector<string>& target_node_names,
                      std::vector<Tensor>* outputs,
                      RunMetadata* run_metadata,
                      const string& prun_handle)
{

  // Convert to proto
  std::unique_ptr<MutableRunStepRequestWrapper> req(
      master_->CreateRunStepRequest());
  std::unique_ptr<MutableRunStepResponseWrapper> resp(
      master_->CreateRunStepResponse());

  *req->mutable_options() = run_options;

  if (!prun_handle.empty()) {
    req->set_partial_run_handle(prun_handle);
  }

  for (const auto& it : inputs) {
    req->add_feed(it.first, it.second);
  }

  // Build an index from fetch tensor name to offset.
  std::unordered_map<string, int> output_name_to_offset;
  for (const string& output_name : output_tensor_names) {
    req->add_fetch(output_name);
    output_name_to_offset.insert(
        std::make_pair(output_name, output_name_to_offset.size()));
  }
  for (const string& target : target_node_names) {
    req->add_target(target);
  }

  CallOptions call_options;
  call_options.SetTimeout(run_options.timeout_in_ms());
  TF_RETURN_IF_ERROR(RunProto(&call_options, req.get(), resp.get()));

  if (!output_tensor_names.empty()) {
    outputs->resize(output_tensor_names.size());
  }

  // Convert response back to Tensors in the correct order.
  for (size_t i = 0; i < resp->num_tensors(); ++i) {
    auto fetch_it = output_name_to_offset.find(resp->tensor_name(i));
    if (fetch_it == output_name_to_offset.end()) {
      return errors::Internal("Received response for unrequested fetch: ",
                              resp->tensor_name(i));
    }

    Tensor output;
    TF_RETURN_IF_ERROR(resp->TensorValue(i, &output));
    (*outputs)[fetch_it->second] = output;
  }

  if (run_metadata) {
    run_metadata->Swap(resp->mutable_metadata());
  }

  return Status::OK();
}

Status HPXSession::Run(const RunOptions& run_options,
                       const std::vector<std::pair<string, Tensor> >& inputs,
                       const std::vector<string>& output_tensor_names,
                       const std::vector<string>& target_node_names,
                       std::vector<Tensor>* outputs,
                       RunMetadata* run_metadata)
{
  return RunHelper(run_options,
                   inputs,
                   output_tensor_names,
                   target_node_names,
                   outputs,
                   run_metadata,
                   /* prun_handle */ "");
}

Status HPXSession::Run(const std::vector<std::pair<string, Tensor> >& inputs,
                       const std::vector<string>& output_tensor_names,
                       const std::vector<string>& target_node_names,
                       std::vector<Tensor>* outputs)
{
  RunOptions run_options;
  run_options.set_timeout_in_ms(options_.config.operation_timeout_in_ms());
  return Run(run_options,
             inputs,
             output_tensor_names,
             target_node_names,
             outputs,
             nullptr);
}

Status HPXSession::RunProto(CallOptions* call_options,
                            MutableRunStepRequestWrapper* req,
                            MutableRunStepResponseWrapper* resp)
{
  {
    mutex_lock l(mu_);
    if (handle_.empty()) {
      return errors::InvalidArgument("A session is not created yet....");
    }

    req->set_session_handle(handle_);
  }
  return master_->RunStep(call_options, req, resp);
}

Status HPXSession::PRunSetup(const std::vector<string>& input_names,
                             const std::vector<string>& output_names,
                             const std::vector<string>& target_nodes,
                             string* handle)
{
  // Convert to proto
  PartialRunSetupRequest req;
  PartialRunSetupResponse resp;
  CallOptions call_options;
  {
    mutex_lock l(mu_);
    if (handle_.empty()) {
      return errors::InvalidArgument("A session is not created yet....");
    }

    req.set_session_handle(handle_);
  }
  for (const string& feed : input_names) {
    req.add_feed(feed);
  }
  for (const string& fetch : output_names) {
    req.add_fetch(fetch);
  }
  for (const string& target : target_nodes) {
    req.add_target(target);
  }
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  TF_RETURN_IF_ERROR(master_->PartialRunSetup(&call_options, &req, &resp));
  *handle = resp.partial_run_handle();
  return Status::OK();
}

Status HPXSession::PRun(const string& handle,
                        const std::vector<std::pair<string, Tensor> >& inputs,
                        const std::vector<string>& output_names,
                        std::vector<Tensor>* outputs)
{
  RunOptions run_options;
  run_options.set_timeout_in_ms(options_.config.operation_timeout_in_ms());
  return RunHelper(run_options,
                   inputs,
                   output_names,
                   /* targets */ {},
                   outputs,
                   /* run_metadata */ nullptr,
                   handle);
}

Status HPXSession::Close()
{
  CloseSessionRequest req;
  {
    mutex_lock l(mu_);
    if (handle_.empty()) {
      return errors::InvalidArgument("A session is not created yet....");
    }
    req.set_session_handle(handle_);
    handle_.clear();
  }
  CloseSessionResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  Status s = master_->CloseSession(&call_options, &req, &resp);
  return s;
}

std::vector<DeviceAttributes> HPXSession::ListDevices()
{
  std::vector<DeviceAttributes> devices;

  ListDevicesRequest req;
  ListDevicesResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options_.config.operation_timeout_in_ms());
  Status s = master_->ListDevices(&call_options, &req, &resp);
  if (!s.ok()) {
    LOG(ERROR) << "Could not list devices: " << s;
    return devices;
  }

  for (const auto& device_attr : resp.local_device()) {
    devices.push_back(device_attr);
  }
  for (const auto& device_attr : resp.remote_device()) {
    devices.push_back(device_attr);
  }

  return devices;
}

void HPXSession::SetRemoteMaster(std::unique_ptr<MasterInterface> master)
{
  master_ = std::move(master);
}

// Static method.
Status HPXSession::Reset(const SessionOptions& options,
                         const std::vector<string>& containers)
{
  /*auto master = NewGrpcMaster(master_channel);
  ResetRequest req;
  for (const auto& c : containers) req.add_container(c);
  ResetResponse resp;
  CallOptions call_options;
  call_options.SetTimeout(options.config.operation_timeout_in_ms());
  Status ret = master->Reset(&call_options, &req, &resp);
  delete master;
  return ret;*/
  return Status::OK();
}

class HPXSessionFactory : public SessionFactory
{
  public:
  bool AcceptsOptions(const SessionOptions& options) override
  {
    return StringPiece(options.target).starts_with(kSchemePrefix);
  }

  Session* NewSession(const SessionOptions& options) override
  {
    std::unique_ptr<HPXSession> ret;
    Status s = HPXSession::Create(options, &ret);
    if (s.ok()) {
      return ret.release();
    } else {
      LOG(ERROR) << "Error during session construction: " << s.ToString();
      return nullptr;
    }
  }

  // Invokes the session specific static method to reset containers.
  Status Reset(const SessionOptions& options,
               const std::vector<string>& containers) override
  {
    return HPXSession::Reset(options, containers);
  }
};

class HPXSessionRegistrar
{
  public:
  HPXSessionRegistrar()
  {
    SessionFactory::Register("HPX_SESSION", new HPXSessionFactory());
  }
};
static HPXSessionRegistrar registrar;

} // namespace tensorflow
