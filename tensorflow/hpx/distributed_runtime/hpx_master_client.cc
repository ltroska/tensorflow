#include "tensorflow/hpx/distributed_runtime/hpx_master_client.h"

namespace tensorflow
{

HPXMasterClient::HPXMasterClient(hpx::future<hpx::id_type>&& id) : base_type(std::move(id))
{
}

HPXMasterClient::HPXMasterClient(hpx::id_type&& id) : base_type(std::move(id))
{
}

Status HPXMasterClient::CreateSession(CallOptions* call_options,
                     const CreateSessionRequest* request,
                     CreateSessionResponse* response)
{
  return CallActionSync<s_type::CreateSessionAction>(request, response);
}

Status HPXMasterClient::ExtendSession(CallOptions* call_options,
                     const ExtendSessionRequest* request,
                     ExtendSessionResponse* response)
{
  return CallActionSync<s_type::ExtendSessionAction>(request, response);
}

Status HPXMasterClient::PartialRunSetup(CallOptions* call_options,
                       const PartialRunSetupRequest* request,
                       PartialRunSetupResponse* response)
{
  return CallActionSync<s_type::PartialRunSetupAction>(request, response);
}

Status HPXMasterClient::RunStep(CallOptions* call_options,
               const RunStepRequest* request,
               RunStepResponse* response)
{
  return CallActionSync<s_type::RunStepAction>(request, response);
}

Status HPXMasterClient::CloseSession(CallOptions* call_options,
                    const CloseSessionRequest* request,
                    CloseSessionResponse* response)
{
  return CallActionSync<s_type::CloseSessionAction>(request, response);
}

Status HPXMasterClient::ListDevices(CallOptions* call_options,
                   const ListDevicesRequest* request,
                   ListDevicesResponse* response)
{
  return CallActionSync<s_type::ListDevicesAction>(request, response);
}

Status HPXMasterClient::Reset(CallOptions* call_options,
             const ResetRequest* request,
             ResetResponse* response)
{
  return CallActionSync<s_type::ResetAction>(request, response);
}

}