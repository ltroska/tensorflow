#include "tensorflow/hpx/distributed_runtime/hpx_worker_cache.h"
#include <hpx/include/run_as.hpp>

#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/hpx/distributed_runtime/hpx_worker.h"

namespace tensorflow
{

namespace
{

  class HPXWorkerCache : public WorkerCachePartial
  {
public:
    explicit HPXWorkerCache(WorkerInterface* local_worker,
                            const string& local_target,
                            const std::size_t num_workers,
                            global_runtime* init)
        : local_target_(local_target)
        , num_workers_(num_workers)
        , local_worker_(local_worker)
        , init_(init)
    {
    }

    void ListWorkers(std::vector<string>* workers) override
    {
      HPXWorker::ListWorkers(workers, num_workers_, init_);
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
    const std::size_t num_workers_;
    WorkerInterface* const local_worker_; // Not owned.
    WorkerCacheLogger logger_;
    global_runtime* init_;
  };

} // namespace

WorkerCacheInterface* NewHPXWorkerCache()
{
  return new HPXWorkerCache(nullptr, "", 1, nullptr);
}

WorkerCacheInterface*
NewHPXWorkerCacheWithLocalWorker(WorkerInterface* local_worker,
                                 const string& local_target,
                                 const std::size_t num_workers,
                                 global_runtime* init)
{
  return new HPXWorkerCache(local_worker, local_target, num_workers, init);
}

} // namespace tensorflow
