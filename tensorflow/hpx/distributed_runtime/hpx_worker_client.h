#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_CLIENT_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_CLIENT_H_

#include "tensorflow/hpx/distributed_runtime/server/hpx_worker_server.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"

namespace tensorflow
{

struct HPXWorkerClient
    : hpx::components::client_base<HPXWorkerClient, server::HPXWorkerServer>
{
  using base_type =
      hpx::components::client_base<HPXWorkerClient, server::HPXWorkerServer>;

  HPXWorkerClient() = default;

  HPXWorkerClient(hpx::future<hpx::id_type>&& id);

  HPXWorkerClient(hpx::id_type&& id);

  std::string GetWorkerName() const;  
  
  void SetWorkerName(std::string const& name);

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done);

  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done);

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done);

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done);

  void RunGraphAsync(CallOptions* opts,
                     const RunGraphRequest* request,
                     RunGraphResponse* response,
                     StatusCallback done);

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done);

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done);

  void RecvTensorAsync(CallOptions* opts,
                       const RecvTensorRequest* request,
                       TensorResponse* response,
                       StatusCallback done);

  void LoggingAsync(const LoggingRequest* request,
                    LoggingResponse* response,
                    StatusCallback done);

  void TracingAsync(const TracingRequest* request,
                    TracingResponse* response,
                    StatusCallback done);

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