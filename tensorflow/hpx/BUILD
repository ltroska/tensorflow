load(
    "@local_config_hpx//hpx:build_defs.bzl",
    "tf_hpx_library",
)

load(
    "@local_config_hpx//hpx:hpx_bazel_defs.bzl",
    "hpx_component_copts",
)

tf_hpx_library(
    name = "hpx_runtime",
    srcs = [],
    hdrs = ["hpx_global_runtime.h", "hpx_thread_registration.h"],
    alwayslink = 1,
    visibility = ["//visibility:public"]
    
)

tf_hpx_library(
    name = "hpx_executor",
    srcs = ["common_runtime/hpx_executor.cc"],
    hdrs = ["common_runtime/hpx_executor.h"],
    deps = [
	    "//tensorflow/core:core_cpu",
        ],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)

tf_hpx_library(
    name = "hpx_graph_runner",
    srcs = ["common_runtime/hpx_graph_runner.cc"],
    hdrs = ["common_runtime/hpx_graph_runner.h"],
    hpx_deps = [
        ":hpx_executor",        
        ":hpx_runtime",
        ],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)

tf_hpx_library(
    name = "hpx_worker_cache",
    srcs = ["distributed_runtime/hpx_worker_cache.cc"],
    hdrs = ["distributed_runtime/hpx_worker_cache.h"],
    hpx_deps = [
        ":hpx_worker",
        ":hpx_runtime",
        ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:worker_cache",
        "//tensorflow/core/distributed_runtime:worker_cache_logger",
        "//tensorflow/core/distributed_runtime:worker_cache_partial",
        "//tensorflow/core/distributed_runtime:worker_interface", 
      ],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)

tf_hpx_library(
    name = "hpx_server",
    srcs = ["distributed_runtime/hpx_server.cc"],
    hdrs = ["distributed_runtime/hpx_server.h"],
    hpx_deps = [
        ":hpx_worker",
        ":hpx_master",
        ":hpx_worker_cache",
        ":hpx_runtime",
        ],
    deps = [
        "//tensorflow/core/distributed_runtime:master_interface",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime:local_master",
        "//tensorflow/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:worker_proto_cc",            
      ],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)

tf_hpx_library(
    name = "hpx_session",
    srcs = ["distributed_runtime/hpx_session.cc"],
    hdrs = ["distributed_runtime/hpx_session.h"],
    hpx_deps = [
        ":hpx_master",
        ":hpx_runtime",
    ],
    deps = [
        "//tensorflow/core/distributed_runtime:master_interface",
        "//tensorflow/core/distributed_runtime:local_master",
        "//tensorflow/core:master_proto_cc",
      ],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)

tf_hpx_library(
    name = "core_hpx",
    hpx_deps = [
        ":hpx_server",
        ":hpx_graph_runner",
        ":hpx_session",
        ],
    alwayslink = 1,
    visibility = ["//visibility:public"]
)

tf_hpx_library(
    name = "hpx_serialization",
    hdrs = ["distributed_runtime/hpx_tensorflow_serialization.h"],
)



tf_hpx_library(
    name = "hpx_worker",
    srcs = ["distributed_runtime/server/hpx_worker_server.cc"],
    hdrs = ["distributed_runtime/server/hpx_worker_server.h", "distributed_runtime/hpx_worker_client.h", "distributed_runtime/hpx_worker.h"],
    copts = hpx_component_copts("hpx_worker_server"),
    hpx_deps = [
          ":hpx_serialization",
        ],
    deps = [
          "//tensorflow/core/distributed_runtime:worker",
          "//tensorflow/core/distributed_runtime/rpc:grpc_tensor_coding",
        ],
    #linkshared = 1,
    visibility = ["//visibility:public"]    
)

tf_hpx_library(
    name = "hpx_master",
    srcs = ["distributed_runtime/server/hpx_master_server.cc"],
    hdrs = ["distributed_runtime/server/hpx_master_server.h", "distributed_runtime/hpx_master_client.h", "distributed_runtime/hpx_master.h"],
    copts = hpx_component_copts("hpx_master_server"),
    hpx_deps = [
        ":hpx_serialization",
        ],
    deps = [
           "//tensorflow/core/distributed_runtime:master",
        ],
    #linkshared = 1,
    visibility = ["//visibility:public"]    
)
