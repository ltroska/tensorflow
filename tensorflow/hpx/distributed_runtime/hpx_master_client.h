#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_MASTER_CLIENT_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_MASTER_CLIENT_H_

#include "tensorflow/hpx/distributed_runtime/server/hpx_master_server.h"
#include "tensorflow/core/distributed_runtime/master_env.h"

namespace tensorflow
{

struct HPXMasterClient
    : hpx::components::client_base<HPXMasterClient, server::HPXMasterServer>
{
  using base_type =
      hpx::components::client_base<HPXMasterClient, server::HPXMasterServer>;

  using s_type = server::HPXMasterServer;

  HPXMasterClient() = default;

  HPXMasterClient(hpx::future<hpx::id_type>&& id);
  HPXMasterClient(hpx::id_type&& id);
  
  Status CreateSession(CallOptions* call_options,
                       const CreateSessionRequest* request,
                       CreateSessionResponse* response);

  Status ExtendSession(CallOptions* call_options,
                       const ExtendSessionRequest* request,
                       ExtendSessionResponse* response);

  Status PartialRunSetup(CallOptions* call_options,
                         const PartialRunSetupRequest* request,
                         PartialRunSetupResponse* response);
                         
  Status RunStep(CallOptions* call_options,
                 const RunStepRequest* request,
                 RunStepResponse* response);

  Status CloseSession(CallOptions* call_options,
                      const CloseSessionRequest* request,
                      CloseSessionResponse* response);

  Status ListDevices(CallOptions* call_options,
                     const ListDevicesRequest* request,
                     ListDevicesResponse* response);

  Status Reset(CallOptions* call_options,
               const ResetRequest* request,
               ResetResponse* response);

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