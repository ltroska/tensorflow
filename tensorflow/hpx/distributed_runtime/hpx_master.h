#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_MASTER_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_MASTER_H_

#include "tensorflow/hpx/distributed_runtime/hpx_master_client.h"
#include "tensorflow/hpx/hpx_global_runtime.h"
#include "tensorflow/hpx/hpx_thread_registration.h"
#include <hpx/include/run_as.hpp>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/distributed_runtime/master_interface.h"

namespace tensorflow
{

struct HPXMaster : public MasterInterface
{
  HPXMaster() = default;

  HPXMaster(global_runtime* rt, std::string const& basename, Master* master);

  HPXMaster(global_runtime* rt, std::string const& basename);

  HPXMaster(HPXMaster const& other);
  HPXMaster(HPXMaster&& other);

  void operator=(HPXMaster const& other);

  void operator=(HPXMaster&& other);

  Status CreateSession(CallOptions* call_options,
                       const CreateSessionRequest* request,
                       CreateSessionResponse* response) override;

  Status ExtendSession(CallOptions* call_options,
                       const ExtendSessionRequest* request,
                       ExtendSessionResponse* response) override;

  Status PartialRunSetup(CallOptions* call_options,
                         const PartialRunSetupRequest* request,
                         PartialRunSetupResponse* response) override;

  Status RunStep(CallOptions* call_options,
                 RunStepRequestWrapper* request,
                 MutableRunStepResponseWrapper* response);

  Status CloseSession(CallOptions* call_options,
                      const CloseSessionRequest* request,
                      CloseSessionResponse* response) override;

  Status ListDevices(CallOptions* call_options,
                     const ListDevicesRequest* request,
                     ListDevicesResponse* response) override;
  Status Reset(CallOptions* call_options,
               const ResetRequest* request,
               ResetResponse* response) override;

  protected:
  template <typename F, typename... Args>
  Status CallAction(F&& f, char const* reg_string, Args... args)
  {
    return MaybeRunAsHPXThread([ this, reg_string, f, args... ]()->Status {
                                 return (client_.*f)(args...);
                               },
                               reg_string);
  }

  template <typename F>
  typename std::result_of<F()>::type
  MaybeRunAsHPXThread(F&& f, char const* reg_string)
  {
    return MaybeRunAsHPXThreadGlobal(std::move(f), reg_string, init_);
  }

  private:
  HPXMasterClient client_;
  global_runtime* init_;
  std::string basename_;
};


HPXMaster* NewHPXMaster(global_runtime* init, std::string const& basename);
}

#endif