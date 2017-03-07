#ifndef TENSORFLOW_COMMON_RUNTIME_HPX_GLOBAL_RUNTIME_H_
#define TENSORFLOW_COMMON_RUNTIME_HPX_GLOBAL_RUNTIME_H_

#include <hpx/hpx.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/hpx_start.hpp>

#include <mutex>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Store the command line arguments in global variables to make them available
// to the startup code.
/*
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

///////////////////////////////////////////////////////////////////////////////
// This class demonstrates how to initialize a console instance of HPX
// (locality 0). In order to create an HPX instance which connects to a running
// HPX application two changes have to be made:
//
//  - replace hpx::runtime_mode_console with hpx::runtime_mode_connect
//  - replace hpx::finalize() with hpx::disconnect()
//
struct manage_global_runtime
{
    manage_global_runtime()
      : running_(false), rts_(nullptr)
    {
#if defined(HPX_WINDOWS)
        hpx::detail::init_winsocket();
#endif

        std::vector<std::string> const cfg = {
            // make sure hpx_main is always executed
            "hpx.run_hpx_main!=1",
            // allow for unknown command line options
            "hpx.commandline.allow_unknown!=1",
            // disable HPX' short options
            "hpx.commandline.aliasing!=0"
            
        };

        using hpx::util::placeholders::_1;
        using hpx::util::placeholders::_2;
        hpx::util::function_nonser<int(int, char**)> start_function =
            hpx::util::bind(&manage_global_runtime::hpx_main, this, _1, _2);

        char *dummy_argv[2] = { const_cast<char*>(HPX_APPLICATION_STRING), nullptr };

        if (!hpx::start(start_function, 1, dummy_argv, cfg, hpx::runtime_mode_console))
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

    ~manage_global_runtime()
    {
        // notify hpx_main above to tear down the runtime
        {
            std::lock_guard<hpx::lcos::local::spinlock> lk(mtx_);
            rts_ = nullptr;               // reset pointer
        }

        cond_.notify_one();     // signal exit

        // wait for the runtime to exit
        hpx::stop();
    }

    // registration of external (to HPX) threads
    void register_thread(char const* name)
    {
        hpx::register_thread(rts_, name);
    }
    void unregister_thread()
    {
        hpx::unregister_thread(rts_);
    }

protected:
    // Main HPX thread, does nothing but wait for the application to exit
    int hpx_main(int argc, char* argv[])
    {
        // Store a pointer to the runtime here.
        rts_ = hpx::get_runtime_ptr();

        // Signal to constructor that thread has started running.
        {
            std::lock_guard<std::mutex> lk(startup_mtx_);
            running_ = true;
        }

        startup_cond_.notify_one();

        // Here other HPX specific functionality could be invoked...

        // Now, wait for destructor to be called.
        {
            std::unique_lock<hpx::lcos::local::spinlock> lk(mtx_);
            if (rts_ != nullptr)
                cond_.wait(lk);
        }

        // tell the runtime it's ok to exit
        return hpx::finalize();
    }

private:
    hpx::lcos::local::spinlock mtx_;
    hpx::lcos::local::condition_variable_any cond_;

    std::mutex startup_mtx_;
    std::condition_variable startup_cond_;
    bool running_;

    hpx::runtime* rts_;
};

#endif