#ifndef TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_H_
#define TENSORFLOW_HPX_CORE_DISTRIBUTED_RUNTIME_HPX_WORKER_H_

#include "tensorflow/hpx/core/distributed_runtime/hpx_worker_client.h"
#include "tensorflow/hpx/core/hpx_global_runtime.h"
#include "tensorflow/hpx/core/hpx_thread_registration.h"
#include <hpx/include/run_as.hpp>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"

extern global_runtime init;

namespace tensorflow {

  
struct HPXWorker
  : public WorkerInterface
{
    HPXWorker() {};    
    
    HPXWorker(global_runtime* rt, std::string const& basename, WorkerEnv* worker_env)
    : init_(rt), basename_(basename)
    {
      std::cout << "HPXWorker::HPXWorker(3) register basename:" << basename_ << std::endl;

      MaybeRunAsHPXThread(
        [this, worker_env]() {
          client_ = 
            hpx::id_type(hpx::components::server::construct
                            <hpx::components::component<server::HPXWorkerServer> >
                            (worker_env)
                          ,hpx::id_type::managed);
          hpx::register_with_basename(basename_, client_.get_id(), 0);
        },
        "HPXWorker::HPXWorker(3)"
      );
    }     
    
    HPXWorker(global_runtime* rt, std::string const& basename)
    : init_(rt), basename_(basename)
    {      
      std::cout << "HPXWorker::HPXWorker(2) basename>" << basename_ << std::endl;
      
      MaybeRunAsHPXThread(
        [this]()
        {
          client_ = hpx::find_from_basename(basename_, 0);
        },
        "HPXWorker::HPXWorker(2)"
      );
    }    
        
    HPXWorker(HPXWorker const& other) : client_(other.client_), init_(other.init_), basename_(other.basename_) {}
    HPXWorker(HPXWorker&& other) : client_(std::move(other.client_)), init_(other.init_), basename_(std::move(other.basename_)) {}
    
    void operator=(HPXWorker const& other)
    {
      client_ = other.client_;
      init_ = other.init_;
      basename_ = other.basename_;
    }    
    
    void operator=(HPXWorker&& other)
    {
      client_ = std::move(other.client_);
      init_ = other.init_;
      basename_ = std::move(other.basename_);

    }

    void test()
    {
      thread_registration_wrapper reg(init_, "test");

      hpx::threads::run_as_hpx_thread([this](){
        return client_.test().get();
      });
      
    }
        
    void GetStatusAsync(const GetStatusRequest* request,
                              GetStatusResponse* response,
                              StatusCallback done) {
      std::cout << "HPXWorker::GetStatusAsync() basename>" << basename_ << std::endl;
      CallAction(&HPXWorkerClient::GetStatusAsync, std::move(done), "GetStatusAsync", request, response);
    }

    void RegisterGraphAsync(const RegisterGraphRequest* request,
                                    RegisterGraphResponse* response,
                                    StatusCallback done){
      std::cout << "HPXWorker::RegisterGraphAsync() basename>" << basename_ << std::endl;
      CallAction(&HPXWorkerClient::RegisterGraphAsync, std::move(done), "RegisterGraphAsync", request, response);
    };

    void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                                      DeregisterGraphResponse* response,
                                      StatusCallback done){
    std::cout << "HPXWorker::DeregisterGraphAsync() basename>" << basename_ << std::endl;
    CallAction(&HPXWorkerClient::DeregisterGraphAsync, std::move(done), "DeregisterGraphAsync", request, response);
    };

    void RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                               MutableRunGraphResponseWrapper* response,
                               StatusCallback done) {
      std::cout << "HPXWorker::RunGraphAsync() basename>" << basename_ << std::endl;
      CallAction(&HPXWorkerClient::RunGraphAsync, std::move(done), "RunGraphAsync", opts, &request->ToProto(), get_proto_from_wrapper(response));
    };
                               
    void CleanupGraphAsync(const CleanupGraphRequest* request,
                                 CleanupGraphResponse* response,
                                 StatusCallback done) {
    std::cout << "HPXWorker::CleanupGraphAsync() basename>" << basename_ << std::endl;
    CallAction(&HPXWorkerClient::CleanupGraphAsync, std::move(done), "CleanupGraphAsync", request, response);
    };

    void CleanupAllAsync(const CleanupAllRequest* request,
                                 CleanupAllResponse* response,
                                 StatusCallback done) {
    std::cout << "HPXWorker::CleanupAllAsync() basename>" << basename_ << std::endl;
    CallAction(&HPXWorkerClient::CleanupAllAsync, std::move(done), "CleanupAllAsync", request, response);
    };

    void RecvTensorAsync(CallOptions* opts,
                                 const RecvTensorRequest* request,
                                 TensorResponse* response,
                                 StatusCallback done) {
    std::cout << "HPXWorker::RecvTensorAsync() basename>" << basename_ << std::endl;

    auto resp = new RecvTensorResponse();
        
    auto wrap = [resp, request, response, done = std::move(done)](Status s)
    {
      response->InitFrom(resp);
      delete resp;
      done(s);
    };

    CallAction(&HPXWorkerClient::RecvTensorAsync, std::move(wrap), "RecvTensorAsync", opts, request, resp);
    };

    void LoggingAsync(const LoggingRequest* request,
                              LoggingResponse* response, StatusCallback done) {
    std::cout << "HPXWorker::LoggingAsync() basename>" << basename_ << std::endl;
    CallAction(&HPXWorkerClient::LoggingAsync, std::move(done), "LoggingAsync", request, response);
    };

    void TracingAsync(const TracingRequest* request,
                              TracingResponse* response, StatusCallback done) {
    std::cout << "HPXWorker::TracingAsync() basename>" << basename_ << std::endl;
    CallAction(&HPXWorkerClient::TracingAsync, std::move(done), "TracingAsync", request, response);
    };
                                
    template<typename F, typename ...Args>
    void CallAction(F&& f, StatusCallback&& done, char const* reg_string, Args... args)
    {
      MaybeRunAsHPXThread(
        [this, f, args..., done = std::move(done)] ()
        {
          (client_.*f)(args..., done);
        },
        reg_string
      );
    }    
    
    template<typename F>
    void MaybeRunAsHPXThread(F&& f, char const* reg_string)
    {
      if (hpx::this_thread::get_id() == hpx::thread::id(hpx::threads::invalid_thread_id))
      {
        thread_registration_wrapper reg(init_, reg_string);

        hpx::threads::run_as_hpx_thread(f);      
      }
      else
      {
        f();
      }
    }

  private:
    HPXWorkerClient client_;
    global_runtime* init_;
    std::string basename_;
    
};
}

#endif