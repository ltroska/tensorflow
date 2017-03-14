#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_H_

#include "tensorflow/hpx/core/distributed_runtime/hpx_worker_client.h"
#include "tensorflow/hpx/core/hpx_global_runtime.h"
#include "tensorflow/hpx/core/hpx_thread_registration.h"
#include <hpx/include/run_as.hpp>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"

namespace tensorflow
{

struct HPXWorker : public WorkerInterface
{
  HPXWorker() {};

  HPXWorker(global_runtime* rt,
            std::string const& basename,
            WorkerEnv* worker_env)
      : init_(rt)
      , basename_("worker:" + basename)
  {
    MaybeRunAsHPXThread(
        [this, worker_env]() {
          client_ = hpx::id_type(
              hpx::components::server::construct<
                  hpx::components::component<server::HPXWorkerServer> >(
                  worker_env),
              hpx::id_type::managed);
          hpx::register_with_basename(basename_, client_.get_id(), 0);
          hpx::register_with_basename("/all_workers/", client_.get_id());
        },
        "HPXWorker::HPXWorker(3)");
  }

  HPXWorker(global_runtime* rt, std::string const& basename)
      : init_(rt)
      , basename_("worker:" + basename)
  {
    MaybeRunAsHPXThread([this]() {
                          client_ = hpx::find_from_basename(basename_, 0);
                        },
                        "HPXWorker::HPXWorker(2)");
  }

  HPXWorker(HPXWorker const& other)
      : client_(other.client_)
      , init_(other.init_)
      , basename_(other.basename_)
  {
  }
  HPXWorker(HPXWorker&& other)
      : client_(std::move(other.client_))
      , init_(other.init_)
      , basename_(std::move(other.basename_))
  {
  }

  void operator=(HPXWorker const& other)
  {
    client_ = other.client_;
    init_ = other.init_;
    basename_ = other.basename_;
  }

  void operator=(HPXWorker&& other)
  {
    client_ = std::move(other.client_);
    init_ = other.init_;
    basename_ = std::move(other.basename_);
  }

  static void ListWorkers(std::vector<std::string>* workers,
                          global_runtime* init)
  {
    auto f = [workers]() {
      std::size_t const num_localities =
          hpx::get_num_localities(hpx::launch::sync);

      auto futures =
          hpx::find_all_from_basename("/all_workers/", num_localities - 1);

      workers->reserve(num_localities);

      for (auto&& id : futures)
        workers->push_back(HPXWorkerClient(std::move(id)).GetWorkerName());
    };

    MaybeRunAsHPXThreadGlobal(std::move(f), "ListWorkers", init);
  }

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done)
  {
    CallAction(&HPXWorkerClient::GetStatusAsync,
               std::move(done),
               "GetStatusAsync",
               request,
               response);
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done)
  {
    CallAction(&HPXWorkerClient::RegisterGraphAsync,
               std::move(done),
               "RegisterGraphAsync",
               request,
               response);
  };

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done)
  {
    CallAction(&HPXWorkerClient::DeregisterGraphAsync,
               std::move(done),
               "DeregisterGraphAsync",
               request,
               response);
  };

  void RunGraphAsync(CallOptions* opts,
                     RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done)
  {
    CallAction(&HPXWorkerClient::RunGraphAsync,
               std::move(done),
               "RunGraphAsync",
               opts,
               &request->ToProto(),
               get_proto_from_wrapper(response));
  };

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done)
  {
    CallAction(&HPXWorkerClient::CleanupGraphAsync,
               std::move(done),
               "CleanupGraphAsync",
               request,
               response);
  };

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done)
  {
    CallAction(&HPXWorkerClient::CleanupAllAsync,
               std::move(done),
               "CleanupAllAsync",
               request,
               response);
  };

  void RecvTensorAsync(CallOptions* opts,
                       const RecvTensorRequest* request,
                       TensorResponse* response,
                       StatusCallback done)
  {
    auto resp = new RecvTensorResponse();

    auto wrap = [ resp, request, response, done = std::move(done) ](Status s)
    {
      response->InitFrom(resp);
      delete resp;
      done(s);
    };

    CallAction(&HPXWorkerClient::RecvTensorAsync,
               std::move(wrap),
               "RecvTensorAsync",
               opts,
               request,
               resp);
  };

  void LoggingAsync(const LoggingRequest* request,
                    LoggingResponse* response,
                    StatusCallback done)
  {
    CallAction(&HPXWorkerClient::LoggingAsync,
               std::move(done),
               "LoggingAsync",
               request,
               response);
  };

  void TracingAsync(const TracingRequest* request,
                    TracingResponse* response,
                    StatusCallback done)
  {
    CallAction(&HPXWorkerClient::TracingAsync,
               std::move(done),
               "TracingAsync",
               request,
               response);
  };

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
};
}

#endif