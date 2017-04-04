#include "tensorflow/hpx/distributed_runtime/hpx_master.h"

namespace tensorflow
{

HPXMaster::HPXMaster(global_runtime* rt,
                     std::string const& basename,
                     Master* master)
    : init_(rt)
    , basename_("master:" + basename)
{
  MaybeRunAsHPXThread(
      [this, master]() {
        client_ = hpx::id_type(
            hpx::components::server::construct<
                hpx::components::component<server::HPXMasterServer> >(master),
            hpx::id_type::managed);
        hpx::register_with_basename(basename_, client_.get_id(), 0);
      },
      "HPXMaster::HPXMaster(3)");
}

HPXMaster::HPXMaster(global_runtime* rt, std::string const& basename)
    : init_(rt)
    , basename_("master:" + basename)
{
  MaybeRunAsHPXThread([this]() {
                        client_ = hpx::find_from_basename(basename_, 0);
                      },
                      "HPXMaster::HPXMaster(2)");
}

HPXMaster::HPXMaster(HPXMaster const& other)
    : client_(other.client_)
    , init_(other.init_)
    , basename_(other.basename_)
{
}

HPXMaster::HPXMaster(HPXMaster&& other)
    : client_(std::move(other.client_))
    , init_(other.init_)
    , basename_(std::move(other.basename_))
{
}

void HPXMaster::operator=(HPXMaster const& other)
{
  client_ = other.client_;
  init_ = other.init_;
  basename_ = other.basename_;
}

void HPXMaster::operator=(HPXMaster&& other)
{
  client_ = std::move(other.client_);
  init_ = other.init_;
  basename_ = std::move(other.basename_);
}

Status HPXMaster::CreateSession(CallOptions* call_options,
                                const CreateSessionRequest* request,
                                CreateSessionResponse* response)
{
  return CallAction(&HPXMasterClient::CreateSession,
                    "CreateSession",
                    call_options,
                    request,
                    response);
}

Status HPXMaster::ExtendSession(CallOptions* call_options,
                                const ExtendSessionRequest* request,
                                ExtendSessionResponse* response)
{
  return CallAction(&HPXMasterClient::ExtendSession,
                    "ExtendSession",
                    call_options,
                    request,
                    response);
}

Status HPXMaster::PartialRunSetup(CallOptions* call_options,
                                  const PartialRunSetupRequest* request,
                                  PartialRunSetupResponse* response)
{
  return CallAction(&HPXMasterClient::PartialRunSetup,
                    "PartialRunSetup",
                    call_options,
                    request,
                    response);
}

Status HPXMaster::RunStep(CallOptions* call_options,
                          RunStepRequestWrapper* request,
                          MutableRunStepResponseWrapper* response)
{
  return CallAction(&HPXMasterClient::RunStep,
                    "RunStep",
                    call_options,
                    &request->ToProto(),
                    get_proto_from_wrapper(response));
}

Status HPXMaster::CloseSession(CallOptions* call_options,
                               const CloseSessionRequest* request,
                               CloseSessionResponse* response)
{
  return CallAction(&HPXMasterClient::CloseSession,
                    "CloseSession",
                    call_options,
                    request,
                    response);
}

Status HPXMaster::ListDevices(CallOptions* call_options,
                              const ListDevicesRequest* request,
                              ListDevicesResponse* response)
{
  return CallAction(&HPXMasterClient::ListDevices,
                    "ListDevices",
                    call_options,
                    request,
                    response);
}

Status HPXMaster::Reset(CallOptions* call_options,
                        const ResetRequest* request,
                        ResetResponse* response)
{
  return CallAction(
      &HPXMasterClient::Reset, "Reset", call_options, request, response);
}

HPXMaster* NewHPXMaster(global_runtime* init, std::string const& basename)
{
  return new HPXMaster(init, basename);
}
}