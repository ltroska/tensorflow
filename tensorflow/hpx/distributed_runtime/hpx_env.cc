#include "tensorflow/hpx/distributed_runtime/hpx_env.h"
#include <hpx/include/run_as.hpp>
#include "tensorflow/hpx/hpx_thread_registration.h"
#include <iostream>

namespace tensorflow
{

void HPXEnv::SchedClosure(std::function<void()> closure)
{
  hpx::threads::run_as_os_thread(closure);
}
}