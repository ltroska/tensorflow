#ifndef TENSORFLOW_HPX_CORE_HPX_GLOBAL_RUNTIME_H_
#define TENSORFLOW_HPX_CORE_HPX_GLOBAL_RUNTIME_H_

#include <hpx/hpx.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_init.hpp>

#include <mutex>
#include <string>
#include <vector>

struct global_runtime
{
  // registration of external (to HPX) threads
  void register_thread(char const* name);
  void unregister_thread();

  void start(std::string const& hostname,
             std::string const& port,
             std::string const& root_hostname,
             std::string const& root_port,
             bool is_root = false);

  global_runtime();

  ~global_runtime() = default;

  void stop_and_shutdown();

  void wait_for_stop_then_shutdown();

  bool is_running();

  void stop();

  private:
  // Main HPX thread, does nothing but wait for the application to exit
  int hpx_main(int argc, char* argv[]);

  hpx::lcos::local::spinlock mtx_;
  hpx::lcos::local::condition_variable_any cond_;

  std::mutex startup_mtx_;
  std::condition_variable startup_cond_;
  bool running_;
  bool is_root_;
  std::string port_;
  std::string hostname_;
  hpx::runtime* rts_;

  bool owning_;

  std::atomic_uint_fast64_t thread_count_;
};
#endif // TENSORFLOW_HPX_CORE_HPX_GLOBAL_RUNTIME_H_