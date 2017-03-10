/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_SERVER_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_SERVER_H_

#include "tensorflow/hpx/core/distributed_runtime/hpx_worker.h"
#include "tensorflow/hpx/core/distributed_runtime/hpx_master.h"


#include <memory>

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class GrpcWorker;
class Master;

class HPXServer : public ServerInterface {
 protected:
  HPXServer(const ServerDef& server_def, Env* env);

 public:
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);

  // Destruction is only supported in the factory method.
  virtual ~HPXServer();

  // Implementations of ServerInterface methods.
  Status Start() override;
  Status Stop() override;
  Status Join() override;
  const string target() const override;

 protected:
  Status Init();
  
  std::unique_ptr<Master> CreateMaster(MasterEnv* master_env);

private: 
  // The overall server configuration.
  const ServerDef server_def_;
  Env* env_;

  // The port requested for this server.
  std::string hostname_;
  std::string port_;

  // Guards state transitions.
  mutex mu_;

  // Represents the current state of the server, which changes as follows:
  //
  //                 Join()            Join()
  //                  ___               ___
  //      Start()     \ /    Stop()     \ /
  // NEW ---------> STARTED --------> STOPPED
  //   \                          /
  //    \________________________/
  //            Stop(), Join()
  enum State { NEW, STARTED, STOPPED };
  State state_ GUARDED_BY(mu_);

  // Implementation of a TensorFlow master
  MasterEnv master_env_;
  HPXMaster hpx_master_;
  std::unique_ptr<Master> master_impl_;

  // Implementation of a TensorFlow worker
  WorkerEnv worker_env_;
  HPXWorker hpx_worker_;

  global_runtime init_;

};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SERVER_LIB_H_
