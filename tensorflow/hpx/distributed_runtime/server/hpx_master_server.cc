#include "tensorflow/hpx/distributed_runtime/server/hpx_master_server.h"

namespace tensorflow
{
namespace server
{

  HPXMasterServer::HPXMasterServer(Master* master) : master_(master)
  {
  }

  std::pair<Status, CreateSessionResponse>
  HPXMasterServer::CreateSession(CreateSessionRequest const& request)
  {
    CreateSessionResponse response;
    return std::make_pair(
        CallMasterSync(&Master::CreateSession, &request, &response),
        std::move(response));
  }

  std::pair<Status, ExtendSessionResponse>
  HPXMasterServer::ExtendSession(ExtendSessionRequest const& request)
  {
    ExtendSessionResponse response;
    return std::make_pair(
        CallMasterSync(&Master::ExtendSession, &request, &response),
        std::move(response));
  }

  std::pair<Status, PartialRunSetupResponse>
  HPXMasterServer::PartialRunSetup(PartialRunSetupRequest const& request)
  {
    PartialRunSetupResponse response;
    return std::make_pair(
        CallMasterSync(&Master::PartialRunSetup, &request, &response),
        std::move(response));
  }

  std::pair<Status, RunStepResponse>
  HPXMasterServer::RunStep(RunStepRequest const& request)
  {
    CallOptions opt;
    RunStepResponse response;

    RunStepRequestWrapper* wrapped_request = new ProtoRunStepRequest(&request);
    MutableRunStepResponseWrapper* wrapped_response =
        new NonOwnedProtoRunStepResponse(&response);

    hpx::lcos::local::promise<Status> p;
    auto fut = p.get_future();

    auto done = [&p, wrapped_request, wrapped_response](Status const& s) {
      p.set_value(s);
      delete wrapped_request;
      delete wrapped_response;
    };

    (master_->RunStep)(
        &opt, wrapped_request, wrapped_response, std::move(done));
    return std::make_pair(std::move(fut.get()), std::move(response));
  }

  std::pair<Status, CloseSessionResponse>
  HPXMasterServer::CloseSession(CloseSessionRequest const& request)
  {
    CloseSessionResponse response;

    return std::make_pair(
        CallMasterSync(&Master::CloseSession, &request, &response),
        std::move(response));
  }

  std::pair<Status, ListDevicesResponse>
  HPXMasterServer::ListDevices(ListDevicesRequest const& request)
  {

    ListDevicesResponse response;
    return std::make_pair(
        CallMasterSync(&Master::ListDevices, &request, &response),
        std::move(response));
  }

  std::pair<Status, ResetResponse>
  HPXMasterServer::Reset(ResetRequest const& request)
  {

    ResetResponse response;
    return std::make_pair(CallMasterSync(&Master::Reset, &request, &response),
                          std::move(response));
  }

} // namespace server
} // namespace tensorflow

using hpx_master_server_component = tensorflow::server::HPXMasterServer;
using hpx_master_server_type =
    hpx::components::component<hpx_master_server_component>;

HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_COMPONENT(hpx_master_server_type, hpx_master_server_component);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::CreateSessionAction,
                    HPXMasterServerCreateSessionAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::ExtendSessionAction,
                    HPXMasterServerExtendSessionAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::PartialRunSetupAction,
                    HPXMasterServerPartialRunSetupAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::RunStepAction,
                    HPXMasterServerRunStepAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::CloseSessionAction,
                    HPXMasterServerCloseSessionAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::ListDevicesAction,
                    HPXMasterServerListDevicesAction);
HPX_REGISTER_ACTION(tensorflow::server::HPXMasterServer::ResetAction,
                    HPXMasterServerResetAction);