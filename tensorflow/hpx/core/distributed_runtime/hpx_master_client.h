#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_MASTER_CLIENT_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_MASTER_CLIENT_H_

#include "tensorflow/hpx/core/distributed_runtime/server/hpx_master_server.h"
#include "tensorflow/core/distributed_runtime/master_env.h"

namespace tensorflow
{

struct HPXMasterClient
    : hpx::components::client_base<HPXMasterClient, server::HPXMasterServer>
{
  using base_type =
      hpx::components::client_base<HPXMasterClient, server::HPXMasterServer>;

  using s_type = server::HPXMasterServer;

  HPXMasterClient()
  {
  }

  HPXMasterClient(hpx::future<hpx::id_type>&& id) : base_type(std::move(id))
  {
  }

  HPXMasterClient(hpx::id_type&& id) : base_type(std::move(id))
  {
  }

  Status CreateSession(CallOptions* call_options,
                       const CreateSessionRequest* request,
                       CreateSessionResponse* response)
  {
    return CallActionSync<s_type::CreateSessionAction>(request, response);
  }

  Status ExtendSession(CallOptions* call_options,
                       const ExtendSessionRequest* request,
                       ExtendSessionResponse* response)
  {
    return CallActionSync<s_type::ExtendSessionAction>(request, response);
  }

  Status PartialRunSetup(CallOptions* call_options,
                         const PartialRunSetupRequest* request,
                         PartialRunSetupResponse* response)
  {
    return CallActionSync<s_type::PartialRunSetupAction>(request, response);
  }

  Status RunStep(CallOptions* call_options,
                 const RunStepRequest* request,
                 RunStepResponse* response)
  {
    return CallActionSync<s_type::RunStepAction>(request, response);
  }

  Status CloseSession(CallOptions* call_options,
                      const CloseSessionRequest* request,
                      CloseSessionResponse* response)
  {
    return CallActionSync<s_type::CloseSessionAction>(request, response);
  }

  Status ListDevices(CallOptions* call_options,
                     const ListDevicesRequest* request,
                     ListDevicesResponse* response)
  {
    return CallActionSync<s_type::ListDevicesAction>(request, response);
  }

  Status Reset(CallOptions* call_options,
               const ResetRequest* request,
               ResetResponse* response)
  {
    return CallActionSync<s_type::ResetAction>(request, response);
  }

  protected:
  template <typename Action, typename Request, typename Response>
  inline Status CallActionSync(Request* req, Response* resp)
  {
    std::pair<Status, Response> result = Action()(get_id(), *req);

    *resp = std::move(result.second);

    return std::move(result.first);
  }
};
}

#endif