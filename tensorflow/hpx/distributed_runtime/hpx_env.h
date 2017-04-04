#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_ENV_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_ENV_H_

#include "tensorflow/core/platform/env.h"

#include "tensorflow/hpx/hpx_global_runtime.h"

namespace tensorflow
{

struct HPXEnv : public EnvWrapper
{
  HPXEnv(Env* env, global_runtime* rt)
      : EnvWrapper(env)
      , rt_(rt)
  {
  }

  ~HPXEnv() = default;

  void SchedClosure(std::function<void()> closure) override;

  private:
  global_runtime* rt_;
};
}

#endif