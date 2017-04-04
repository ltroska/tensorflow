#include "tensorflow/hpx/distributed_runtime/server/hpx_worker_server.h"

#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"

namespace tensorflow
{
namespace server
{

  HPXWorkerImpl::HPXWorkerImpl(WorkerEnv* env) : Worker(env)
  {
  }

  void HPXWorkerImpl::RecvTensorAsync(CallOptions* opts,
                                      const RecvTensorRequest* request,
                                      std::vector< ::grpc::Slice>* response,
                                      StatusCallback done)
  {
    const int64 step_id = request->step_id();
    WorkerSession* session = env_->session_mgr->WorkerSessionForStepId(step_id);
    const string& key = request->rendezvous_key();
    // TRACEPRINTF("RecvTensor: %lld %s", step_id, key.c_str());
    Rendezvous::ParsedKey parsed;
    Status s = Rendezvous::ParseKey(key, &parsed);

    Device* src_dev = nullptr;
    if (s.ok()) {
      s = PrepareRecvTensor(parsed, &src_dev);
    }
    if (!s.ok()) {
      done(s);
      return;
    }

    // Request the tensor associated with the rendezvous key. Any time
    // while waiting for the tensor to be produced, up until the start
    // of execution of the callback lambda body below, an RPC
    // cancellation should abort the rendezvous.
    opts->SetCancelCallback([this, step_id]() { AbortStep(step_id); });
    session->rendezvous_mgr->RecvLocalAsync(
        step_id,
        parsed,
        [opts, response, done, src_dev](const Status& status,
                                        const Rendezvous::Args& send_args,
                                        const Rendezvous::Args& recv_args,
                                        const Tensor& val,
                                        const bool is_dead) {
          opts->ClearCancelCallback();
          if (status.ok()) {
            // DMA can only be used for Tensors that do not fall into
            // the following three odd edge cases: 1) a zero-size
            // buffer, 2) a dead tensor which has an uninit value, and
            // 3) the tensor has the on_host allocation attribute,
            // i.e. it's in CPU RAM *independent of its assigned
            // device type*.
            const bool on_host = send_args.alloc_attrs.on_host();
            {
              // Non-DMA cases.
              if (src_dev->tensorflow_gpu_device_info() && (!on_host)) {
                done(errors::Internal("No GPU device in process"));
              } else {
                grpc::EncodeTensorToVector(is_dead, val, response);

                done(Status::OK());
              }
            }
          } else {
            //  !s.ok()
            done(status);
          }
        });
  }

  HPXWorkerServer::HPXWorkerServer(WorkerEnv* worker_env)
  {
    SetWorkerEnv(worker_env);
  }

  std::string HPXWorkerServer::GetWorkerName() const
  {
    return name_;
  }  
  
  void HPXWorkerServer::SetWorkerName(std::string const& name)
  {
    name_ = name;
  }

  std::pair<Status, GetStatusResponse>
  HPXWorkerServer::GetStatus(GetStatusRequest const& request)
  {
    GetStatusResponse response;

    Status s = worker_->GetStatus(&request, &response);

    return std::make_pair(std::move(s), std::move(response));
  }
  
  std::pair<Status, CreateWorkerSessionResponse>
  HPXWorkerServer::CreateWorkerSession(CreateWorkerSessionRequest const& request)
  {
    CreateWorkerSessionResponse response;
    
    hpx::lcos::local::promise<Status> p;
    auto f = p.get_future();
    auto done = [&p](Status const& s) {p.set_value(s);}; 
    
    worker_->CreateWorkerSessionAsync(&request, &response, std::move(done));
    
    return std::make_pair(std::move(f.get()), std::move(response));
  }
  
  std::pair<Status, RegisterGraphResponse>
  HPXWorkerServer::RegisterGraph(RegisterGraphRequest const& request)
  {
    RegisterGraphResponse response;

    Status s = worker_->RegisterGraph(&request, &response);

    return std::make_pair(std::move(s), std::move(response));
  }

  std::pair<Status, DeregisterGraphResponse>
  HPXWorkerServer::DeregisterGraph(DeregisterGraphRequest const& request)
  {
    DeregisterGraphResponse response;

    Status s = worker_->DeregisterGraph(&request, &response);

    return std::make_pair(std::move(s), std::move(response));
  }

  std::pair<Status, RunGraphResponse>
  HPXWorkerServer::RunGraph(RunGraphRequest const& request)
  {
    CallOptions opts;

    RunGraphResponse response;

    RunGraphRequestWrapper* wrapped_request =
        new ProtoRunGraphRequest(&request);

    MutableRunGraphResponseWrapper* wrapped_response =
        new NonOwnedProtoRunGraphResponse(&response);

    hpx::lcos::local::promise<Status> p;
    auto f = p.get_future();

    auto done = [&p, wrapped_request, wrapped_response](Status s) {
      p.set_value(s);
      delete wrapped_request;
      delete wrapped_response;
    };

    worker_->RunGraphAsync(
        &opts, wrapped_request, wrapped_response, std::move(done));

    return std::make_pair(f.get(), std::move(response));
  }

  std::pair<Status, CleanupGraphResponse>
  HPXWorkerServer::CleanupGraph(CleanupGraphRequest const& request)
  {
    CleanupGraphResponse response;

    Status s = worker_->CleanupGraph(&request, &response);

    return std::make_pair(std::move(s), std::move(response));
  }

  std::pair<Status, CleanupAllResponse>
  HPXWorkerServer::CleanupAll(CleanupAllRequest const& request)
  {
    CleanupAllResponse response;

    Status s = worker_->CleanupAll(&request, &response);

    return std::make_pair(std::move(s), std::move(response));
  }

  std::pair<Status, std::vector< ::grpc::Slice> >
  HPXWorkerServer::RecvTensor(RecvTensorRequest const& request)
  {
    std::vector< ::grpc::Slice> response;
    CallOptions opts;

    hpx::lcos::local::promise<Status> p;
    auto f = p.get_future();

    auto done = [&p](Status s) { p.set_value(s); };

    worker_->RecvTensorAsync(&opts, &request, &response, std::move(done));

    auto s = f.get();

    return std::make_pair(std::move(s), std::move(response));
  }

  std::pair<Status, LoggingResponse>
  HPXWorkerServer::Logging(LoggingRequest const& request)
  {
    LoggingResponse response;

    Status s = worker_->Logging(&request, &response);

    return std::make_pair(std::move(s), std::move(response));
  }

  std::pair<Status, TracingResponse>
  HPXWorkerServer::Tracing(TracingRequest const& request)
  {
    TracingResponse response;

    Status s = worker_->Tracing(&request, &response);

    return std::make_pair(std::move(s), std::move(response));
  }

  void HPXWorkerServer::SetWorkerEnv(WorkerEnv* worker_env)
  {
    worker_env_ = worker_env;
    worker_.reset(new HPXWorkerImpl(worker_env));
  }
}
}

using hpx_worker_server_component = tensorflow::server::HPXWorkerServer;
using hpx_worker_server_type =
    hpx::components::component<hpx_worker_server_component>;

HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_COMPONENT(hpx_worker_server_type, hpx_worker_server_component);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::GetWorkerNameAction,
                    HPXWorkerServerGetWorkerNameAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::SetWorkerNameAction,
                    HPXWorkerServerSetWorkerNameAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::GetStatusAction,
                    HPXWorkerServerGetStatusAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::CreateWorkerSessionAction,
                    HPXWorkerServerCreateWorkerSessionAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::RegisterGraphAction,
                    HPXWorkerServerRegisterGraphAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::DeregisterGraphAction,
                    HPXWorkerServerDeregisterGraphAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::RunGraphAction,
                    HPXWorkerServerRunGraphAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::CleanupGraphAction,
                    HPXWorkerServerCleanupGraphAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::CleanupAllAction,
                    HPXWorkerServerCleanupAllAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::RecvTensorAction,
                    HPXWorkerServerRecvTensorAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::LoggingAction,
                    HPXWorkerServerLoggingAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::TracingAction,
                    HPXWorkerServerTracingAction);