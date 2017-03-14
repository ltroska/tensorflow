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

#include "tensorflow/hpx/core/distributed_runtime/hpx_worker_cache.h"
#include <hpx/include/run_as.hpp>

#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/hpx/core/distributed_runtime/hpx_worker.h"

namespace tensorflow
{

namespace
{

  class HPXWorkerCache : public WorkerCachePartial
  {
public:
    explicit HPXWorkerCache(WorkerInterface* local_worker,
                            const string& local_target,
                            global_runtime* init)
        : local_target_(local_target)
        , local_worker_(local_worker)
        , init_(init)
    {
    }

    void ListWorkers(std::vector<string>* workers) override
    {
      HPXWorker::ListWorkers(workers, init_);
    }

    WorkerInterface* CreateWorker(const string& target) override
    {
      if (target == local_target_) {
        return local_worker_;
      } else {
        return new HPXWorker(init_, target);
      }
    }

    void ReleaseWorker(const string& target, WorkerInterface* worker) override
    {
      if (target == local_target_) {
        CHECK_EQ(worker, local_worker_)
            << "Releasing a worker that was not returned by this WorkerCache";
      } else {
        WorkerCacheInterface::ReleaseWorker(target, worker);
      }
    }

    void SetLogging(bool v) override
    {
      logger_.SetLogging(v);
    }

    void ClearLogs() override
    {
      logger_.ClearLogs();
    }

    bool RetrieveLogs(int64 step_id, StepStats* ss) override
    {
      return logger_.RetrieveLogs(step_id, ss);
    }

private:
    const string local_target_;
    WorkerInterface* const local_worker_; // Not owned.
    WorkerCacheLogger logger_;
    global_runtime* init_;
  };

} // namespace

WorkerCacheInterface* NewHPXWorkerCache()
{
  return new HPXWorkerCache(nullptr, "", nullptr);
}

WorkerCacheInterface*
NewHPXWorkerCacheWithLocalWorker(WorkerInterface* local_worker,
                                 const string& local_target,
                                 global_runtime* init)
{
  return new HPXWorkerCache(local_worker, local_target, init);
}

} // namespace tensorflow
