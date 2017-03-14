#ifndef HPX_CORE_HPX_THREAD_REGISTRATION_H_
#define HPX_CORE_HPX_THREAD_REGISTRATION_H_

#include "tensorflow/hpx/core/hpx_global_runtime.h"

struct thread_registration_wrapper
{
  thread_registration_wrapper(global_runtime* init, char const* name)
      : init_(init)
      , name_(name)
  {
    // Register this thread with HPX, this should be done once for
    // each external OS-thread intended to invoke HPX functionality.
    // Calling this function more than once will silently fail (will
    // return false).
    // global_runtime::get_instance().register_thread(name);
    init_->register_thread(name_);
  }
  ~thread_registration_wrapper()
  {
    // Unregister the thread from HPX, this should be done once in the
    // end before the external thread exists.
    // global_runtime::get_instance().unregister_thread();
    init_->unregister_thread();
  }

  global_runtime* init_;
  char const* name_;
};

template <typename F>
typename std::result_of<F()>::type
MaybeRunAsHPXThreadGlobal(F&& f, char const* reg_string, global_runtime* init)
{
  if (hpx::threads::get_self_ptr() == nullptr) {
    thread_registration_wrapper reg(init, reg_string);

    return hpx::threads::run_as_hpx_thread(f);
  } else {
    return f();
  }
}

#endif