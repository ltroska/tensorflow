#include "tensorflow/hpx/hpx_global_runtime.h"

// registration of external (to HPX) threads
void global_runtime::register_thread(char const* name)
{
  hpx::register_thread(
      rts_, (std::string(name) + std::to_string(thread_count_++)).c_str());
}
void global_runtime::unregister_thread()
{
  hpx::unregister_thread(rts_);
}

void global_runtime::start(std::string const& hostname,
                           std::string const& port,
                           std::string const& root_hostname,
                           std::string const& root_port,
                           bool is_root)
{
  if (hpx::get_runtime_ptr() == nullptr) {
    owning_ = true;
    is_root_ = is_root;

    port_ = port;
    hostname_ = hostname;

#if defined(HPX_WINDOWS)
    hpx::detail::init_winsocket();
#endif

    std::vector<std::string> const cfg = {
      // make sure hpx_main is always executed
      "hpx.run_hpx_main!=1",
      // allow for unknown command line options
      "hpx.commandline.allow_unknown!=1",
      // ignore batch environment (in particular hpx.localities)
      "hpx.ignore_batch_env!=1",
      // disable HPX' short options
      "hpx.commandline.aliasing!=0",       "hpx.os_threads!=1",
      "hpx.agas.address=" + root_hostname, "hpx.agas.port=" + root_port,
      "hpx.parcel.address=" + hostname_,   "hpx.parcel.port=" + port_
    };

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    hpx::util::function_nonser<int(int, char**)> start_function =
        hpx::util::bind(&global_runtime::hpx_main, this, _1, _2);

    char* dummy_argv[] = { const_cast<char*>(HPX_APPLICATION_STRING), nullptr };

    bool started;
    if (is_root)
      started = hpx::start(
          start_function, 1, dummy_argv, cfg, hpx::runtime_mode_console);
    else
      started = hpx::start(
          start_function, 1, dummy_argv, cfg, hpx::runtime_mode_connect);

    if (!started) {
      // Something went wrong while initializing the runtime.
      // This early we can't generate any output, just bail out.
      std::abort();
    }

    // Wait for the main HPX thread (hpx_main below) to have started running
    std::unique_lock<std::mutex> lk(startup_mtx_);
    while (!running_)
      startup_cond_.wait(lk);
  } else {
    owning_ = false;
  }

  running_ = true;
  rts_ = hpx::get_runtime_ptr();
}

global_runtime::global_runtime()
    : running_(false)
    , rts_(nullptr)
    , thread_count_(0)
{
}

void global_runtime::stop_and_shutdown()
{
  if (owning_) {
    register_thread("stop_and_shutdown");
    stop();
    hpx::stop();
  }
}

void global_runtime::wait_for_stop_then_shutdown()
{
  if (owning_) {
    register_thread("shutdown");
    hpx::stop();
    rts_ = nullptr;
  }
}

bool global_runtime::is_running()
{
  return running_;
}

void global_runtime::stop()
{
  if (owning_) {
    register_thread("stop");
    if (running_) {
      // notify hpx_main above to tear down the runtime
      {
        std::lock_guard<hpx::lcos::local::spinlock> lk(mtx_);
        running_ = false; // reset pointer
      }

      cond_.notify_one();
    }
  }
}

int global_runtime::hpx_main(int argc, char* argv[])
{
  // Signal to constructor that thread has started running.
  {
    std::lock_guard<std::mutex> lk(startup_mtx_);
    running_ = true;
  }

  startup_cond_.notify_one();

  {
    std::unique_lock<hpx::lcos::local::spinlock> lk(mtx_);
    if (running_)
      cond_.wait(lk);
  }

  // tell the runtime it's ok to exit
  if (is_root_)
    return hpx::finalize();

  return hpx::disconnect();
}