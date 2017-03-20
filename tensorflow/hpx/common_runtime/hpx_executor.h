#ifndef TENSORFLOW_HPX_CORE_COMMON_RUNTIME_EXECUTOR_H_
#define TENSORFLOW_HPX_CORE_COMMON_RUNTIME_EXECUTOR_H_

#include "tensorflow/core/common_runtime/executor.h"

namespace tensorflow
{

::tensorflow::Status NewLocalHPXExecutor(const LocalExecutorParams& params,
                                         const Graph* graph,
                                         Executor** executor);

} // end namespace tensorflow

#endif // TENSORFLOW_HPX_CORE_COMMON_RUNTIME_EXECUTOR_H_
