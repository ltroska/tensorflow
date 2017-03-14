#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_SERVER_HPX_WORKER_SERVER_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_SERVER_HPX_WORKER_SERVER_H_

#include <hpx/include/components.hpp>
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/hpx/core/distributed_runtime/hpx_tensorflow_serialization.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"

namespace tensorflow
{
namespace server
{

  struct HPXWorkerImpl : Worker
  {
    HPXWorkerImpl(WorkerEnv* env) : Worker(env)
    {
    }

    void RecvTensorAsync(CallOptions* opts,
                         const RecvTensorRequest* request,
                         RecvTensorResponse* response,
                         StatusCallback done)
    {
      const int64 step_id = request->step_id();
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
      env_->rendezvous_mgr->RecvLocalAsync(
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
                  response->set_is_dead(is_dead);
                  auto proto = response->mutable_tensor();
                  proto->set_dtype(val.dtype());

                  val.shape().AsProto(proto->mutable_tensor_shape());
                  val.AsProtoTensorContent(proto);

                  done(Status::OK());
                }
              }
            } else {
              //  !s.ok()
              done(status);
            }
          });
    }
  };

  struct HPXWorkerServer
      : hpx::components::simple_component_base<HPXWorkerServer>
  {
    HPXWorkerServer()
    {
    }
    ~HPXWorkerServer()
    {
    }

    HPXWorkerServer(WorkerEnv* worker_env)
    {
      SetWorkerEnv(worker_env);
    }

    std::string GetWorkerName() const
    {
      return worker_env_->worker_name;
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer,
                                GetWorkerName,
                                GetWorkerNameAction);

    std::pair<Status, GetStatusResponse>
    GetStatus(GetStatusRequest const& request)
    {
      GetStatusResponse response;

      Status s = worker_->GetStatus(&request, &response);

      return std::make_pair(std::move(s), std::move(response));
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, GetStatus, GetStatusAction);

    std::pair<Status, RegisterGraphResponse>
    RegisterGraph(RegisterGraphRequest const& request)
    {
      RegisterGraphResponse response;

      Status s = worker_->RegisterGraph(&request, &response);

      return std::make_pair(std::move(s), std::move(response));
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer,
                                RegisterGraph,
                                RegisterGraphAction);

    std::pair<Status, DeregisterGraphResponse>
    DeregisterGraph(DeregisterGraphRequest const& request)
    {
      DeregisterGraphResponse response;

      Status s = worker_->DeregisterGraph(&request, &response);

      return std::make_pair(std::move(s), std::move(response));
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer,
                                DeregisterGraph,
                                DeregisterGraphAction);

    std::pair<Status, RunGraphResponse> RunGraph(RunGraphRequest const& request)
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

    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, RunGraph, RunGraphAction);

    std::pair<Status, CleanupGraphResponse>
    CleanupGraph(CleanupGraphRequest const& request)
    {
      CleanupGraphResponse response;

      Status s = worker_->CleanupGraph(&request, &response);

      return std::make_pair(std::move(s), std::move(response));
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer,
                                CleanupGraph,
                                CleanupGraphAction);

    std::pair<Status, CleanupAllResponse>
    CleanupAll(CleanupAllRequest const& request)
    {
      CleanupAllResponse response;

      Status s = worker_->CleanupAll(&request, &response);

      return std::make_pair(std::move(s), std::move(response));
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, CleanupAll, CleanupAllAction);

    std::pair<Status, RecvTensorResponse>
    RecvTensor(RecvTensorRequest const& request)
    {
      RecvTensorResponse response;
      CallOptions opts;

      hpx::lcos::local::promise<Status> p;
      auto f = p.get_future();

      auto done = [&p](Status s) { p.set_value(s); };

      worker_->RecvTensorAsync(&opts, &request, &response, std::move(done));

      return std::make_pair(std::move(f.get()), std::move(response));
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, RecvTensor, RecvTensorAction);

    std::pair<Status, LoggingResponse> Logging(LoggingRequest const& request)
    {
      LoggingResponse response;

      Status s = worker_->Logging(&request, &response);

      return std::make_pair(std::move(s), std::move(response));
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, Logging, LoggingAction);

    std::pair<Status, TracingResponse> Tracing(TracingRequest const& request)
    {
      TracingResponse response;

      Status s = worker_->Tracing(&request, &response);

      return std::make_pair(std::move(s), std::move(response));
    }
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, Tracing, TracingAction);

    void SetWorkerEnv(WorkerEnv* worker_env)
    {
      worker_env_ = worker_env;
      worker_.reset(new HPXWorkerImpl(worker_env));
    }

private:
    std::unique_ptr<HPXWorkerImpl> worker_;
    WorkerEnv* worker_env_;
  };
}
}

HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::GetWorkerNameAction,
    HPXWorkerServerGetWorkerNameAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::GetStatusAction,
    HPXWorkerServerGetStatusAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::RegisterGraphAction,
    HPXWorkerServerRegisterGraphAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::DeregisterGraphAction,
    HPXWorkerServerDeregisterGraphAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::RunGraphAction,
    HPXWorkerServerRunGraphAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::CleanupGraphAction,
    HPXWorkerServerCleanupGraphAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::CleanupAllAction,
    HPXWorkerServerCleanupAllAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::RecvTensorAction,
    HPXWorkerServerRecvTensorAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::LoggingAction,
    HPXWorkerServerLoggingAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::TracingAction,
    HPXWorkerServerTracingAction);

#endif
