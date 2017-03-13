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

#include "tensorflow/hpx/core/distributed_runtime/hpx_server.h"
#include "tensorflow/hpx/core/hpx_global_runtime.h"
#include "tensorflow/hpx/core/distributed_runtime/hpx_worker_cache.h"
#include <hpx/include/run_as.hpp>

#include <limits>
#include <memory>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/local_master.h"
#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/master_session.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

HPXServer::HPXServer(const ServerDef& server_def, Env* env)
    : server_def_(server_def), env_(env), state_(NEW) {}

HPXServer::~HPXServer() {
  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());

  // TODO(mrry): Refactor the *Env classes so that it is less fiddly
  // to destroy them.
  delete master_env_.worker_cache;  // Shared with worker_env.worker_cache.

  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  delete worker_env_.graph_mgr;
  delete worker_env_.device_mgr;

  delete worker_env_.rendezvous_mgr;

  // Do not delete (as these are not owned by the server):
  // - master_env_.env
  // - worker_env_.env
  // - worker_env_.compute_pool
}

Status HPXServer::Init() {
  mutex_lock l(mu_);
  CHECK_EQ(state_, NEW);
  master_env_.env = env_;
  worker_env_.env = env_;

  SessionOptions sess_opts;
  sess_opts.config = server_def_.default_session_config();

  // Configure shared devices between master and worker.
  string name_prefix =
      strings::StrCat("/job:", server_def_.job_name(), "/replica:0", "/task:",
                      server_def_.task_index());
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(sess_opts, name_prefix,
                                               &master_env_.local_devices));
  worker_env_.device_mgr = new DeviceMgr(master_env_.local_devices);
  string unused;
  if (!DeviceNameUtils::SplitDeviceName(master_env_.local_devices[0]->name(),
                                        &worker_env_.worker_name, &unused)) {
    return errors::Internal("Could not parse worker name.");
  }

  // Look up the port that has been requested for this task in `server_def_`.
  for (const auto& job : server_def_.cluster().job()) {
    if (job.name() == server_def_.job_name()) {
      auto iter = job.tasks().find(server_def_.task_index());
      if (iter == job.tasks().end()) {
        return errors::InvalidArgument("Task ", server_def_.task_index(),
                                       " was not defined in job \"",
                                       server_def_.job_name(), "\"");
      }
            
      const std::vector<string> hostname_port =
          str_util::Split(iter->second, ':');
      if (hostname_port.size() != 2) {
        return errors::InvalidArgument(
            "Could not parse port for local server from \"", iter->second,
            "\"");
      } else {
        hostname_ = hostname_port[0];        
        port_ = hostname_port[1];  
        break;
      }
    }
  }
  if (port_.empty()) {
    return errors::Internal("Job \"", server_def_.job_name(),
                            "\" was not defined in cluster");
  }
  
  std::string root_hostname;
  std::string root_port;
  for (const auto& job : server_def_.cluster().job()) {
    if (job.name() == "hpx_root")
    {
      auto iter = job.tasks().find(0);
      if (iter == job.tasks().end()) {
        return errors::InvalidArgument("'hpx_root' job contains no server");
      }
            
      const std::vector<string> hostname_port =
          str_util::Split(iter->second, ':');
      if (hostname_port.size() != 2) {
        return errors::InvalidArgument(
            "Could not parse port for local server from \"", iter->second,
            "\"");
      } else {
        root_hostname = hostname_port[0];        
        root_port = hostname_port[1];  
        break;
      }
    }
  
  }
  
  if (root_hostname.empty() || root_port.empty())
    return errors::Internal("No hpx_root server found");
  
  bool is_root = (hostname_ == root_hostname && port_ == root_port);
  init_.start(hostname_, port_, root_hostname, root_port, is_root);

  hpx_worker_ = HPXWorker(&init_, name_prefix, &worker_env_);
  worker_env_.worker_cache = NewHPXWorkerCacheWithLocalWorker(
      &hpx_worker_, name_prefix, &init_);;

  // Finish setting up master environment.
  master_impl_ = CreateMaster(&master_env_);      
  hpx_master_ = HPXMaster(&init_, hostname_ + ":" + port_, master_impl_.get());
  master_env_.ops = OpRegistry::Global();
  master_env_.worker_cache = worker_env_.worker_cache;
  master_env_.master_session_factory = [](const SessionOptions& options,
                                          const MasterEnv* env,
                                          std::vector<Device*>* remote_devs) {
    return new MasterSession(options, env, remote_devs,
                             CreateNoOpStatsPublisher);
  };

  // Finish setting up worker environment.
  worker_env_.graph_mgr = new GraphMgr(&worker_env_, true);
  worker_env_.compute_pool = ComputePool(sess_opts);
  worker_env_.rendezvous_mgr = new RpcRendezvousMgr(&worker_env_);

  // Provide direct access to the master from in-process clients.
  LocalMaster::Register(target(), master_impl_.get());

  return Status::OK();
}

Status HPXServer::Start() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW: {
      state_ = STARTED;
      LOG(INFO) << "Started server with target: " << target();
      
      MaybeRunAsHPXThreadGlobal([this](){init_.spin_until_stopped();}, "MainLoop", &init_);
            
      return Status::OK();
    }
    case STARTED:
      LOG(INFO) << "Server already started (target: " << target() << ")";
      return Status::OK();
    case STOPPED:
      return errors::FailedPrecondition("Server has stopped.");
    default:
      CHECK(false);
  }
}

Status HPXServer::Stop() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
      MaybeRunAsHPXThreadGlobal([this](){init_.stop();}, "Stop", &init_);
      return Status::OK();
    case STOPPED:
      LOG(INFO) << "Server already stopped (target: " << target() << ")";
      return Status::OK();
    default:
      CHECK(false);
  }
}

Status HPXServer::Join() {
  mutex_lock l(mu_);
  switch (state_) {
    case NEW:
      // Prevent the server from being started subsequently.
      state_ = STOPPED;
      return Status::OK();
    case STARTED:
    case STOPPED:
      return Status::OK();
    default:
      CHECK(false);
  }
}

const string HPXServer::target() const {
  return strings::StrCat("hpx://localhost:", port_);
}

std::unique_ptr<Master> HPXServer::CreateMaster(MasterEnv* master_env) {
  return std::unique_ptr<Master>(new Master(master_env, 0.0));
}

/* static */
Status HPXServer::Create(const ServerDef& server_def, Env* env,
                          std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<HPXServer> ret(new HPXServer(server_def, Env::Default()));
  TF_RETURN_IF_ERROR(ret->Init());
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class HPXServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "hpx";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return HPXServer::Create(server_def, Env::Default(), out_server);
  }
};

// Registers a `ServerFactory` for `HPXServer` instances.
class HPXServerRegistrar {
 public:
  HPXServerRegistrar() {
    /*//gpr_allocation_functions alloc_fns;
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);*/
    ServerFactory::Register("HPX_SERVER", new HPXServerFactory());
  }
};
static HPXServerRegistrar registrar;

}  // namespace
}  // namespace tensorflow
