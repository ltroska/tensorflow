#include "tensorflow/hpx/distributed_runtime/hpx_worker_client.h"

namespace tensorflow
{

HPXWorkerClient::HPXWorkerClient(hpx::future<hpx::id_type>&& id)
    : base_type(std::move(id))
{
}

HPXWorkerClient::HPXWorkerClient(hpx::id_type&& id) : base_type(std::move(id))
{
}

std::string HPXWorkerClient::GetWorkerName() const
{
  server::HPXWorkerServer::GetWorkerNameAction act;
  return act(get_id());
}

void HPXWorkerClient::SetWorkerName(std::string const& name)
{
  server::HPXWorkerServer::SetWorkerNameAction act;
  act(get_id(), name);
}

void HPXWorkerClient::GetStatusAsync(const GetStatusRequest* request,
                                     GetStatusResponse* response,
                                     StatusCallback done)
{
  server::HPXWorkerServer::GetStatusAction act;
  AsyncCallAndAttachCallback(
      std::move(act), request, response, std::move(done));
}

void HPXWorkerClient::CreateWorkerSessionAsync(
      const CreateWorkerSessionRequest* request,
      CreateWorkerSessionResponse* response, StatusCallback done)
{
  server::HPXWorkerServer::CreateWorkerSessionAction act;
  AsyncCallAndAttachCallback(
      std::move(act), request, response, std::move(done));
}

void HPXWorkerClient::RegisterGraphAsync(const RegisterGraphRequest* request,
                                         RegisterGraphResponse* response,
                                         StatusCallback done)
{
  server::HPXWorkerServer::RegisterGraphAction act;
  AsyncCallAndAttachCallback(
      std::move(act), request, response, std::move(done));
}

void
HPXWorkerClient::DeregisterGraphAsync(const DeregisterGraphRequest* request,
                                      DeregisterGraphResponse* response,
                                      StatusCallback done)
{
  server::HPXWorkerServer::DeregisterGraphAction act;
  AsyncCallAndAttachCallback(
      std::move(act), request, response, std::move(done));
}

void HPXWorkerClient::RunGraphAsync(CallOptions* opts,
                                    const RunGraphRequest* request,
                                    RunGraphResponse* response,
                                    StatusCallback done)
{
  server::HPXWorkerServer::RunGraphAction act;
  AsyncCallAndAttachCallback(
      std::move(act), request, response, std::move(done));
}

void HPXWorkerClient::CleanupGraphAsync(const CleanupGraphRequest* request,
                                        CleanupGraphResponse* response,
                                        StatusCallback done)
{
  server::HPXWorkerServer::CleanupGraphAction act;
  AsyncCallAndAttachCallback(
      std::move(act), request, response, std::move(done));
}

void HPXWorkerClient::CleanupAllAsync(const CleanupAllRequest* request,
                                      CleanupAllResponse* response,
                                      StatusCallback done)
{
  server::HPXWorkerServer::CleanupAllAction act;
  AsyncCallAndAttachCallback(
      std::move(act), request, response, std::move(done));
}

void HPXWorkerClient::RecvTensorAsync(CallOptions* opts,
                                      const RecvTensorRequest* request,
                                      TensorResponse* response,
                                      StatusCallback done)
{
  server::HPXWorkerServer::RecvTensorAction act;
  hpx::async(act, get_id(), *request).then(hpx::util::unwrapped(
      [ this,
        done = std::move(done),
        response ](std::pair<Status, std::vector< ::grpc::Slice> > p) {
                    auto data = std::move(p.second);

                    auto tmp = ::grpc::ByteBuffer(data.data(), data.size());
                    grpc_byte_buffer* buf = new grpc_byte_buffer();
                    bool own;
                    ::grpc::SerializationTraits< ::grpc::ByteBuffer>::Serialize(
                        tmp, &buf, &own);

                    ::grpc::SerializationTraits<
                        tensorflow::TensorResponse>::Deserialize(buf,
                                                                 response,
                                                                 own);

                    done(std::move(p.first));
                  }));
}

void HPXWorkerClient::LoggingAsync(const LoggingRequest* request,
                                   LoggingResponse* response,
                                   StatusCallback done)
{
  server::HPXWorkerServer::LoggingAction act;
  AsyncCallAndAttachCallback(
      std::move(act), request, response, std::move(done));
}

void HPXWorkerClient::TracingAsync(const TracingRequest* request,
                                   TracingResponse* response,
                                   StatusCallback done)
{
  server::HPXWorkerServer::TracingAction act;
  AsyncCallAndAttachCallback(
      std::move(act), request, response, std::move(done));
}
}