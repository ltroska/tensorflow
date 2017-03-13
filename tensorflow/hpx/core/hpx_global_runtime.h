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
    void register_thread(char const* name)
    {
        hpx::register_thread(rts_, (std::string(name) + std::to_string(thread_count_++)).c_str());
    }
    void unregister_thread()
    {
        hpx::unregister_thread(rts_);
    }
         
    void start(std::string const& hostname, std::string const& port,
                std::string const& root_hostname, std::string const& root_port,
                bool is_root = false)
    {
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
          // disable HPX' short options
          "hpx.commandline.aliasing!=0",
          "hpx.os_threads!=all",
          "hpx.agas.address=" + root_hostname,
          "hpx.agas.port=" + root_port,
          "hpx.parcel.address=" + hostname_,
          "hpx.parcel.port=" + port_
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
    
    global_runtime() : running_(false), rts_(nullptr), thread_count_(0) {}
    ~global_runtime() {stop();}

    void spin_until_stopped()
    {        
      std::unique_lock<hpx::lcos::local::spinlock> lk(mtx_);
      if (rts_ != nullptr && running_)
          cond_.wait(lk);        
    }

    inline bool is_running()
    {
      return running_;
    }

    void stop()
    {
      if (running_)
      {
        // notify hpx_main above to tear down the runtime
        {
            std::lock_guard<hpx::lcos::local::spinlock> lk(mtx_);
            rts_ = nullptr;               // reset pointer
        }

        cond_.notify_one();
      }
      
      spin_until_stopped();
      
      if (is_root_)
        hpx::stop();
    }
private:          
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
        
        hpx::util::function_nonser<void()> shutdown_func =
          [this]()
          {
            {
                std::lock_guard<std::mutex> lk(startup_mtx_);
                running_ = false;
            }

            startup_cond_.notify_one();
          };
          
        hpx::register_shutdown_function(std::move(shutdown_func));

        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(mtx_);
            if (rts_ != nullptr)
                cond_.wait(lk);
        }
        
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
    std::string port_;
    std::string hostname_;
    hpx::runtime* rts_;
    
    std::atomic_uint_fast64_t thread_count_;
};
#endif // TENSORFLOW_HPX_CORE_HPX_GLOBAL_RUNTIME_H_