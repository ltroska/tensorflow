#include "tensorflow/hpx/core/distributed_runtime/server/hpx_master_server.h"

using hpx_master_server_component = tensorflow::server::HPXMasterServer;
using hpx_master_server_type = hpx::components::component<hpx_master_server_component>;

// Please note that the second argument to this macro must be a
// (system-wide) unique C++-style identifier (without any namespaces)
//
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_COMPONENT(hpx_master_server_type, hpx_master_server_component);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::CreateSessionAction, HPXMasterServerCreateSessionAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::ExtendSessionAction, HPXMasterServerExtendSessionAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::PartialRunSetupAction, HPXMasterServerPartialRunSetupAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::RunStepAction, HPXMasterServerRunStepAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::CloseSessionAction, HPXMasterServerCloseSessionAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::ListDevicesAction, HPXMasterServerListDevicesAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::ResetAction, HPXMasterServerResetAction);