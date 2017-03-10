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

#ifndef THIRD_PARTY_TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_CACHE_H_
#define THIRD_PARTY_TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_CACHE_H_

#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/hpx/core/hpx_global_runtime.h"

namespace tensorflow {

// The returned WorkerCacheInterface object takes the ownership of "cc".
WorkerCacheInterface* NewHPXWorkerCache();

WorkerCacheInterface* NewHPXWorkerCacheWithLocalWorker(
    WorkerInterface* local_worker, const string& local_target,
    global_runtime* init);

}  // namespace tensorflow
#endif  // THIRD_PARTY_TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_CACHE_H_
