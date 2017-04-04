#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_SERVER_HPX_MASTER_SERVER_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_SERVER_HPX_MASTER_SERVER_H_

#include <hpx/include/components.hpp>
#include "tensorflow/hpx/distributed_runtime/hpx_tensorflow_serialization.h"

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"

namespace tensorflow
{
namespace server
{

  struct HPXMasterServer
      : public hpx::components::simple_component_base<HPXMasterServer>
  {
    HPXMasterServer() = default;

    ~HPXMasterServer() = default;

    HPXMasterServer(Master* master);

    std::pair<Status, CreateSessionResponse>
    CreateSession(CreateSessionRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXMasterServer,
                                CreateSession,
                                CreateSessionAction);

    std::pair<Status, ExtendSessionResponse>
    ExtendSession(ExtendSessionRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXMasterServer,
                                ExtendSession,
                                ExtendSessionAction);

    std::pair<Status, PartialRunSetupResponse>
    PartialRunSetup(PartialRunSetupRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXMasterServer,
                                PartialRunSetup,
                                PartialRunSetupAction);

    std::pair<Status, RunStepResponse> RunStep(RunStepRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXMasterServer, RunStep, RunStepAction);

    std::pair<Status, CloseSessionResponse>
    CloseSession(CloseSessionRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXMasterServer,
                                CloseSession,
                                CloseSessionAction);

    std::pair<Status, ListDevicesResponse>
    ListDevices(ListDevicesRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXMasterServer,
                                ListDevices,
                                ListDevicesAction);

    std::pair<Status, ResetResponse> Reset(ResetRequest const& request);
    HPX_DEFINE_COMPONENT_ACTION(HPXMasterServer, Reset, ResetAction);

protected:
    template <typename F, typename Req, typename Resp, typename... Args>
    inline Status
    CallMasterSync(F&& func, Req* request, Resp* response, Args... args)
    {
      hpx::lcos::local::promise<Status> p;
      auto fut = p.get_future();

      (master_->*func)(args..., request, response, [&p](Status const& s) {
        p.set_value(s);
      });

      return std::move(fut.get());
    }

private:
    Master* master_;
  };

} // namespace server
} // namespace tensorflow

HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXMasterServer::CreateSessionAction,
    HPXMasterServerCreateSessionAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXMasterServer::ExtendSessionAction,
    HPXMasterServerExtendSessionAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXMasterServer::PartialRunSetupAction,
    HPXMasterServerPartialRunSetupAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXMasterServer::RunStepAction,
    HPXMasterServerRunStepAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXMasterServer::CloseSessionAction,
    HPXMasterServerCloseSessionAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXMasterServer::ListDevicesAction,
    HPXMasterServerListDevicesAction);
HPX_REGISTER_ACTION_DECLARATION(
    tensorflow::server::HPXMasterServer::ResetAction,
    HPXMasterServerResetAction);

#endif