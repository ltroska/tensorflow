#include "tensorflow/hpx/distributed_runtime/hpx_worker.h"

#include "tensorflow/hpx/hpx_thread_registration.h"

namespace tensorflow
{

HPXWorker::HPXWorker(global_runtime* rt,
                     std::string const& basename,
                     std::size_t const id,
                     WorkerEnv* worker_env)
    : init_(rt)
    , basename_("worker:" + basename)
    , id_(id)
{
  MaybeRunAsHPXThread(
      [this, worker_env, basename]() {
        client_ = hpx::id_type(
            hpx::components::server::construct<
                hpx::components::component<server::HPXWorkerServer> >(
                worker_env),
            hpx::id_type::managed);
        hpx::register_with_basename(basename_, client_.get_id(), 0);
        hpx::register_with_basename("/all/", client_.get_id(), id_);
        
        client_.SetWorkerName(basename);
      },
      "HPXWorker::HPXWorker(3)");
}

HPXWorker::HPXWorker::HPXWorker(global_runtime* rt, std::string const& basename)
    : init_(rt)
    , basename_("worker:" + basename)
{
  MaybeRunAsHPXThread([this]() {
                        client_ = hpx::find_from_basename(basename_, 0);
                      },
                      "HPXWorker::HPXWorker(2)");
}

HPXWorker::HPXWorker(HPXWorker const& other)
    : client_(other.client_)
    , init_(other.init_)
    , basename_(other.basename_)
{
}

HPXWorker::HPXWorker(HPXWorker&& other)
    : client_(std::move(other.client_))
    , init_(other.init_)
    , basename_(std::move(other.basename_))
{
}

void HPXWorker::operator=(HPXWorker const& other)
{
  client_ = other.client_;
  init_ = other.init_;
  basename_ = other.basename_;
}

void HPXWorker::operator=(HPXWorker&& other)
{
  client_ = std::move(other.client_);
  init_ = other.init_;
  basename_ = std::move(other.basename_);
}

void HPXWorker::ListWorkers(std::vector<std::string>* workers,
                            std::size_t const num_workers,
                            global_runtime* init)
{
  auto f = [workers, num_workers]() {
    auto futures = hpx::find_all_from_basename("/all/", num_workers);

    workers->reserve(num_workers);

    for (auto&& id : futures)
      workers->push_back(HPXWorkerClient(std::move(id)).GetWorkerName());
  };

  MaybeRunAsHPXThreadGlobal(std::move(f), "ListWorkers", init);
}

void HPXWorker::GetStatusAsync(const GetStatusRequest* request,
                               GetStatusResponse* response,
                               StatusCallback done)
{
  CallAction(&HPXWorkerClient::GetStatusAsync,
             std::move(done),
             "GetStatusAsync",
             request,
             response);
}

void HPXWorker::CreateWorkerSessionAsync(
      const CreateWorkerSessionRequest* request,
      CreateWorkerSessionResponse* response, StatusCallback done)
{
  CallAction(&HPXWorkerClient::CreateWorkerSessionAsync,
           std::move(done),
           "CreateWorkerSessionAsync",
           request,
           response);
}

void HPXWorker::RegisterGraphAsync(const RegisterGraphRequest* request,
                                   RegisterGraphResponse* response,
                                   StatusCallback done)
{
  CallAction(&HPXWorkerClient::RegisterGraphAsync,
             std::move(done),
             "RegisterGraphAsync",
             request,
             response);
}

void HPXWorker::DeregisterGraphAsync(const DeregisterGraphRequest* request,
                                     DeregisterGraphResponse* response,
                                     StatusCallback done)
{
  CallAction(&HPXWorkerClient::DeregisterGraphAsync,
             std::move(done),
             "DeregisterGraphAsync",
             request,
             response);
}

void HPXWorker::RunGraphAsync(CallOptions* opts,
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
}

void HPXWorker::CleanupGraphAsync(const CleanupGraphRequest* request,
                                  CleanupGraphResponse* response,
                                  StatusCallback done)
{
  CallAction(&HPXWorkerClient::CleanupGraphAsync,
             std::move(done),
             "CleanupGraphAsync",
             request,
             response);
}

void HPXWorker::CleanupAllAsync(const CleanupAllRequest* request,
                                CleanupAllResponse* response,
                                StatusCallback done)
{
  CallAction(&HPXWorkerClient::CleanupAllAsync,
             std::move(done),
             "CleanupAllAsync",
             request,
             response);
}

void HPXWorker::RecvTensorAsync(CallOptions* opts,
                                const RecvTensorRequest* request,
                                TensorResponse* response,
                                StatusCallback done)
{
  CallAction(&HPXWorkerClient::RecvTensorAsync,
             std::move(done),
             "RecvTensorAsync",
             opts,
             request,
             response);
}

void HPXWorker::LoggingAsync(const LoggingRequest* request,
                             LoggingResponse* response,
                             StatusCallback done)
{
  CallAction(&HPXWorkerClient::LoggingAsync,
             std::move(done),
             "LoggingAsync",
             request,
             response);
}

void HPXWorker::TracingAsync(const TracingRequest* request,
                             TracingResponse* response,
                             StatusCallback done)
{
  CallAction(&HPXWorkerClient::TracingAsync,
             std::move(done),
             "TracingAsync",
             request,
             response);
}
}