#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_MASTER_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_MASTER_H_

#include "tensorflow/hpx/core/distributed_runtime/hpx_master_client.h"
#include "tensorflow/hpx/core/hpx_global_runtime.h"
#include "tensorflow/hpx/core/hpx_thread_registration.h"
#include <hpx/include/run_as.hpp>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/distributed_runtime/master_interface.h"

namespace tensorflow
{

struct HPXMaster : public MasterInterface
{
  HPXMaster() {};

  HPXMaster(global_runtime* rt, std::string const& basename, Master* master)
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

  HPXMaster(global_runtime* rt, std::string const& basename)
      : init_(rt)
      , basename_("master:" + basename)
  {
    MaybeRunAsHPXThread([this]() {
                          client_ = hpx::find_from_basename(basename_, 0);
                        },
                        "HPXMaster::HPXMaster(2)");
  }

  HPXMaster(HPXMaster const& other)
      : client_(other.client_)
      , init_(other.init_)
      , basename_(other.basename_)
  {
  }
  HPXMaster(HPXMaster&& other)
      : client_(std::move(other.client_))
      , init_(other.init_)
      , basename_(std::move(other.basename_))
  {
  }

  void operator=(HPXMaster const& other)
  {
    client_ = other.client_;
    init_ = other.init_;
    basename_ = other.basename_;
  }

  void operator=(HPXMaster&& other)
  {
    client_ = std::move(other.client_);
    init_ = other.init_;
    basename_ = std::move(other.basename_);
  }

  Status CreateSession(CallOptions* call_options,
                       const CreateSessionRequest* request,
                       CreateSessionResponse* response) override
  {
    return CallAction(&HPXMasterClient::CreateSession,
                      "CreateSession",
                      call_options,
                      request,
                      response);
  }

  Status ExtendSession(CallOptions* call_options,
                       const ExtendSessionRequest* request,
                       ExtendSessionResponse* response) override
  {
    return CallAction(&HPXMasterClient::ExtendSession,
                      "ExtendSession",
                      call_options,
                      request,
                      response);
  }

  Status PartialRunSetup(CallOptions* call_options,
                         const PartialRunSetupRequest* request,
                         PartialRunSetupResponse* response) override
  {
    return CallAction(&HPXMasterClient::PartialRunSetup,
                      "PartialRunSetup",
                      call_options,
                      request,
                      response);
  }

  Status RunStep(CallOptions* call_options,
                 RunStepRequestWrapper* request,
                 MutableRunStepResponseWrapper* response) override
  {
    return CallAction(&HPXMasterClient::RunStep,
                      "RunStep",
                      call_options,
                      &request->ToProto(),
                      get_proto_from_wrapper(response));
  }

  Status CloseSession(CallOptions* call_options,
                      const CloseSessionRequest* request,
                      CloseSessionResponse* response) override
  {
    return CallAction(&HPXMasterClient::CloseSession,
                      "CloseSession",
                      call_options,
                      request,
                      response);
  }

  Status ListDevices(CallOptions* call_options,
                     const ListDevicesRequest* request,
                     ListDevicesResponse* response) override
  {
    return CallAction(&HPXMasterClient::ListDevices,
                      "ListDevices",
                      call_options,
                      request,
                      response);
  }

  Status Reset(CallOptions* call_options,
               const ResetRequest* request,
               ResetResponse* response) override
  {
    return CallAction(
        &HPXMasterClient::Reset, "Reset", call_options, request, response);
  }

  protected:
  template <typename F, typename... Args>
  inline Status CallAction(F&& f, char const* reg_string, Args... args)
  {
    return MaybeRunAsHPXThread([ this, f, args... ]()->Status {
                                 return (client_.*f)(args...);
                               },
                               reg_string);
  }

  template <typename F>
  inline typename std::result_of<F()>::type
  MaybeRunAsHPXThread(F&& f, char const* reg_string)
  {
    return MaybeRunAsHPXThreadGlobal(std::move(f), reg_string, init_);
  }

  private:
  HPXMasterClient client_;
  global_runtime* init_;
  std::string basename_;
};

inline HPXMaster* NewHPXMaster(global_runtime* init,
                               std::string const& basename)
{
  return new HPXMaster(init, basename);
}
}

#endif