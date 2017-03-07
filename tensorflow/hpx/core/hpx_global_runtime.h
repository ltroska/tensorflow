#ifndef TENSORFLOW_HPX_CORE_HPX_GLOBAL_RUNTIME_H_
#define TENSORFLOW_HPX_CORE_HPX_GLOBAL_RUNTIME_H_

#include <hpx/hpx.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_init.hpp>

#include <mutex>
#include <string>
#include <vector>

/*///////////////////////////////////////////////////////////////////////////////
// Store the command line arguments in global variables to make them available
// to the startup code.

#if defined(linux) || defined(__linux) || defined(__linux__)

int __argc = 0;
char** __argv = nullptr;

void set_argv_argv(int argc, char* argv[], char* env[])
{
    __argc = argc;
    __argv = argv;
}

__attribute__((section(".init_array")))
    void (*set_global_argc_argv)(int, char*[], char*[]) = &set_argv_argv;

#elif defined(__APPLE__)

#include <crt_externs.h>

inline int get_arraylen(char** argv)
{
    int count = 0;
    if (nullptr != argv)
    {
        while(nullptr != argv[count])
            ++count;
    }
    return count;
}

int __argc = get_arraylen(*_NSGetArgv());
char** __argv = *_NSGetArgv();

#endif*/

struct global_runtime
{    
 
   // global_runtime(global_runtime const&) = delete;
    //void operator=(global_runtime const&) = delete;
  
    // registration of external (to HPX) threads
    void register_thread(char const* name)
    {
        hpx::register_thread(rts_, (std::string(name) + std::to_string(thread_count_++)).c_str());
    }
    void unregister_thread()
    {
        hpx::unregister_thread(rts_);
    }
        
    void start(unsigned port = 7100, bool is_root = false, std::string const& where = "unkown")
    {
      std::cout << "starting HPX runtime" << (is_root ? " (root)" : "") <<" on port " << port << " from " << where << std::endl;
      
      is_root_ = is_root;
      port_ = port;
      
      #if defined(HPX_WINDOWS)
        hpx::detail::init_winsocket();
      #endif

      std::vector<std::string> const cfg = {
          // make sure hpx_main is always executed
          "hpx.run_hpx_main!=1",
          // allow for unknown command line options
          "hpx.commandline.allow_unknown!=1",
          // disable HPX' short options
          "hpx.commandline.aliasing!=0",
          "hpx.threads!=4",
          "hpx.localities=3",
          "hpx.agas.address=127.0.0.1",
          "hpx.agas.port=2223",
          "hpx.parcel.address=127.0.0.1",
          "hpx.parcel.port=" + std::to_string(port)
      };

      using hpx::util::placeholders::_1;
      using hpx::util::placeholders::_2;
      hpx::util::function_nonser<int(int, char**)> start_function =
          hpx::util::bind(&global_runtime::hpx_main, this, _1, _2);

      char *dummy_argv[2] = { const_cast<char*>(HPX_APPLICATION_STRING), nullptr };
      
      bool started;
      if (is_root)      
        started = hpx::start(start_function, 1, dummy_argv, cfg, hpx::runtime_mode_console);
      else
        started = hpx::start(start_function, 1, dummy_argv, cfg, hpx::runtime_mode_connect);
        
      if (!started)
      {
          // Something went wrong while initializing the runtime.
          // This early we can't generate any output, just bail out.
          std::abort();
      }

      // Wait for the main HPX thread (hpx_main below) to have started running
      std::unique_lock<std::mutex> lk(startup_mtx_);
      while (!running_)
          startup_cond_.wait(lk);
    }
    
    global_runtime() : thread_count_(0), running_(false), rts_(nullptr) {};

    ~global_runtime()
    {
      if (running_)
      {
        // notify hpx_main above to tear down the runtime
        {
            std::lock_guard<hpx::lcos::local::spinlock> lk(mtx_);
            rts_ = nullptr;               // reset pointer
        }

        cond_.notify_one();

        hpx::stop();
        
      }
    }
private:
    std::atomic_int thread_count_;
          
    // Main HPX thread, does nothing but wait for the application to exit
    int hpx_main(int argc, char* argv[])
    {
        rts_ = hpx::get_runtime_ptr();

        // Signal to constructor that thread has started running.
        {
            std::lock_guard<std::mutex> lk(startup_mtx_);
            running_ = true;
        }

        startup_cond_.notify_one();

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(mtx_);
            if (rts_ != nullptr)
                cond_.wait(lk);
        }
        
        std::cout << "shutting down runtime" << std::endl;
        // tell the runtime it's ok to exit
        if (is_root_)
          return hpx::finalize();
        return hpx::disconnect();
    }
    
    hpx::lcos::local::spinlock mtx_;
    hpx::lcos::local::condition_variable_any cond_;

    std::mutex startup_mtx_;
    std::condition_variable startup_cond_;
    bool running_;
    bool is_root_;
    unsigned port_;
    hpx::runtime* rts_;
};
#endif // TENSORFLOW_HPX_CORE_HPX_GLOBAL_RUNTIME_H_