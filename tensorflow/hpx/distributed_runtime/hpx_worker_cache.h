#ifndef THIRD_PARTY_TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_CACHE_H_
#define THIRD_PARTY_TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_CACHE_H_

#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/hpx/hpx_global_runtime.h"

namespace tensorflow
{

// The returned WorkerCacheInterface object takes the ownership of "cc".
WorkerCacheInterface* NewHPXWorkerCache();

WorkerCacheInterface*
NewHPXWorkerCacheWithLocalWorker(WorkerInterface* local_worker,
                                 const string& local_target,
                                 global_runtime* init);

} // namespace tensorflow
#endif // THIRD_PARTY_TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_CACHE_H_
