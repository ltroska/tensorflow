#include "tensorflow/hpx/core/distributed_runtime/server/hpx_worker_server.h"

using hpx_worker_server_component = tensorflow::server::HPXWorkerServer;
using hpx_worker_server_type =
    hpx::components::component<hpx_worker_server_component>;

// Please note that the second argument to this macro must be a
// (system-wide) unique C++-style identifier (without any namespaces)
//
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_COMPONENT(hpx_worker_server_type, hpx_worker_server_component);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::GetWorkerNameAction,
                    HPXWorkerServerGetWorkerNameAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXWorkerServer::GetStatusAction,
                    HPXWorkerServerGetStatusAction);
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