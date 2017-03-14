#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_CLIENT_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_CLIENT_H_

#include "tensorflow/hpx/core/distributed_runtime/server/hpx_worker_server.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"

namespace tensorflow
{

struct HPXWorkerClient
    : hpx::components::client_base<HPXWorkerClient, server::HPXWorkerServer>
{
  using base_type =
      hpx::components::client_base<HPXWorkerClient, server::HPXWorkerServer>;

  HPXWorkerClient()
  {
  }

  HPXWorkerClient(hpx::future<hpx::id_type>&& id) : base_type(std::move(id))
  {
  }

  HPXWorkerClient(hpx::id_type&& id) : base_type(std::move(id))
  {
  }

  std::string GetWorkerName() const
  {
    server::HPXWorkerServer::GetWorkerNameAction act;
    return act(get_id());
  }

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done)
  {
    server::HPXWorkerServer::GetStatusAction act;
    AsyncCallAndAttachCallback(
        std::move(act), request, response, std::move(done));
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done)
  {
    server::HPXWorkerServer::RegisterGraphAction act;
    AsyncCallAndAttachCallback(
        std::move(act), request, response, std::move(done));
  };

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done)
  {
    server::HPXWorkerServer::DeregisterGraphAction act;
    AsyncCallAndAttachCallback(
        std::move(act), request, response, std::move(done));
  };

  void RunGraphAsync(CallOptions* opts,
                     const RunGraphRequest* request,
                     RunGraphResponse* response,
                     StatusCallback done)
  {
    server::HPXWorkerServer::RunGraphAction act;
    AsyncCallAndAttachCallback(
        std::move(act), request, response, std::move(done));
  };

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done)
  {
    server::HPXWorkerServer::CleanupGraphAction act;
    AsyncCallAndAttachCallback(
        std::move(act), request, response, std::move(done));
  };

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done)
  {
    server::HPXWorkerServer::CleanupAllAction act;
    AsyncCallAndAttachCallback(
        std::move(act), request, response, std::move(done));
  };

  void RecvTensorAsync(CallOptions* opts,
                       const RecvTensorRequest* request,
                       RecvTensorResponse* response,
                       StatusCallback done)
  {
    server::HPXWorkerServer::RecvTensorAction act;
    AsyncCallAndAttachCallback(
        std::move(act), request, response, std::move(done));
  };

  void LoggingAsync(const LoggingRequest* request,
                    LoggingResponse* response,
                    StatusCallback done)
  {
    server::HPXWorkerServer::LoggingAction act;
    AsyncCallAndAttachCallback(
        std::move(act), request, response, std::move(done));
  };

  void TracingAsync(const TracingRequest* request,
                    TracingResponse* response,
                    StatusCallback done)
  {
    server::HPXWorkerServer::TracingAction act;
    AsyncCallAndAttachCallback(
        std::move(act), request, response, std::move(done));
  };

  protected:
  template <typename Action, typename Req, typename Resp>
  void AsyncCallAndAttachCallback(Action&& act,
                                  Req* request,
                                  Resp* response,
                                  StatusCallback&& done)
  {
    hpx::async(act, get_id(), *request).then(hpx::util::unwrapped(
        [ done = std::move(done), response ](std::pair<Status, Resp> p) mutable {
                                              *response = std::move(p.second);

                                              done(p.first);
                                            }));
  }
};
}

#endif