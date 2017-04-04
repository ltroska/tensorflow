#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_H_

#include "tensorflow/hpx/distributed_runtime/hpx_worker_client.h"
#include "tensorflow/hpx/hpx_global_runtime.h"
#include <hpx/include/run_as.hpp>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"

namespace tensorflow
{

struct HPXWorker : public WorkerInterface
{
  HPXWorker() = default;

  HPXWorker(global_runtime* rt,
            std::string const& basename,
            std::size_t const id,
            WorkerEnv* worker_env);

  HPXWorker(global_runtime* rt, std::string const& basename);

  HPXWorker(HPXWorker const& other);
  HPXWorker(HPXWorker&& other);

  void operator=(HPXWorker const& other);
  void operator=(HPXWorker&& other);

  static void ListWorkers(std::vector<std::string>* workers,
                          std::size_t const num_workers,
                          global_runtime* init);

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done);
                      
  void CreateWorkerSessionAsync(
      const CreateWorkerSessionRequest* request,
      CreateWorkerSessionResponse* response, StatusCallback done);
      
  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done);

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done);

  void RunGraphAsync(CallOptions* opts,
                     RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
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
  template <typename F, typename... Args>
  void
  CallAction(F&& f, StatusCallback&& done, char const* reg_string, Args... args)
  {
    MaybeRunAsHPXThread([
                          this,
                          f,
                          args...,
                          done = std::move(done)
                        ]() { (client_.*f)(args..., done); },
                        reg_string);
  }

  template <typename F>
  inline void MaybeRunAsHPXThread(F&& f, char const* reg_string)
  {
    MaybeRunAsHPXThreadGlobal(std::move(f), reg_string, init_);
  }

  private:
  HPXWorkerClient client_;
  global_runtime* init_;
  std::string basename_;
  std::size_t id_;
};
}

#endif