#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_SERVER_HPX_WORKER_SERVER_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_SERVER_HPX_WORKER_SERVER_H_

#include <hpx/include/components.hpp>
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/hpx/distributed_runtime/hpx_tensorflow_serialization.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"

namespace tensorflow
{
namespace server
{

  struct HPXWorkerImpl : Worker
  {
    HPXWorkerImpl(WorkerEnv* env);

    void RecvTensorAsync(CallOptions* opts,
                         const RecvTensorRequest* request,
                         std::vector< ::grpc::Slice>* response,
                         StatusCallback done);
  };

  struct HPXWorkerServer
      : hpx::components::simple_component_base<HPXWorkerServer>
  {
    HPXWorkerServer() = default;

    ~HPXWorkerServer() = default;

    HPXWorkerServer(WorkerEnv* worker_env);

    std::string GetWorkerName() const;
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer,
                                GetWorkerName,
                                GetWorkerNameAction);    
                                
    void SetWorkerName(std::string const& name);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer,
                                SetWorkerName,
                                SetWorkerNameAction);

    std::pair<Status, GetStatusResponse>
    GetStatus(GetStatusRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, GetStatus, GetStatusAction);

    std::pair<Status, CreateWorkerSessionResponse>
    CreateWorkerSession(CreateWorkerSessionRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer,
                                CreateWorkerSession,
                                CreateWorkerSessionAction);

    std::pair<Status, RegisterGraphResponse>
    RegisterGraph(RegisterGraphRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer,
                                RegisterGraph,
                                RegisterGraphAction);

    std::pair<Status, DeregisterGraphResponse>
    DeregisterGraph(DeregisterGraphRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer,
                                DeregisterGraph,
                                DeregisterGraphAction);

    std::pair<Status, RunGraphResponse>
    RunGraph(RunGraphRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, RunGraph, RunGraphAction);

    std::pair<Status, CleanupGraphResponse>
    CleanupGraph(CleanupGraphRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer,
                                CleanupGraph,
                                CleanupGraphAction);

    std::pair<Status, CleanupAllResponse>
    CleanupAll(CleanupAllRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, CleanupAll, CleanupAllAction);

    std::pair<Status, std::vector< ::grpc::Slice> >
    RecvTensor(RecvTensorRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, RecvTensor, RecvTensorAction);

    std::pair<Status, LoggingResponse> Logging(LoggingRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, Logging, LoggingAction);

    std::pair<Status, TracingResponse> Tracing(TracingRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXWorkerServer, Tracing, TracingAction);

    void SetWorkerEnv(WorkerEnv* worker_env);

private:
    std::unique_ptr<HPXWorkerImpl> worker_;
    WorkerEnv* worker_env_;
    std::string name_;
  };
}
}

HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::GetWorkerNameAction,
    HPXWorkerServerGetWorkerNameAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::SetWorkerNameAction,
    HPXWorkerServerSetWorkerNameAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::GetStatusAction,
    HPXWorkerServerGetStatusAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXWorkerServer::CreateWorkerSessionAction,
    HPXWorkerServerCreateWorkerSessionAction);
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
