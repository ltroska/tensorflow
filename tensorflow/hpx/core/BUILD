load(
    "@local_config_hpx//hpx:build_defs.bzl",
    "if_hpx",
)

if_hpx(
    cc_library(
        name = "hpx_runtime",
        srcs = [],
        hdrs = ["hpx_global_runtime.h", "hpx_thread_registration.h"],
        copts = ["-fexceptions"],
        deps = ["@local_config_hpx//hpx:hpx"],
        alwayslink = 1,
        visibility = ["//visibility:public"]
    )
)

if_hpx(
    cc_library(
	    name = "hpx_executor",
	    srcs = ["common_runtime/hpx_executor.cc"],
	    hdrs = ["common_runtime/hpx_executor.h"],
        copts = ["-fexceptions"],
	    deps = [
		    "//tensorflow/core:core_cpu",
		    "@local_config_hpx//hpx:hpx",
            ],
        alwayslink = 1,
        visibility = ["//visibility:public"]
    )
)

if_hpx(
    cc_library(
	    name = "hpx_graph_runner",
	    srcs = ["common_runtime/hpx_graph_runner.cc"],
	    hdrs = ["common_runtime/hpx_graph_runner.h"],
        copts = ["-fexceptions"],
	    deps = [
		    "@local_config_hpx//hpx:hpx",
            ":hpx_executor",                
            ],
        alwayslink = 1,
        visibility = ["//visibility:public"]
    )
)

if_hpx(
    cc_library(
        name = "hpx_rendezvous_mgr",
        srcs = ["distributed_runtime/hpx_rendezvous_mgr.cc"],
        hdrs = ["distributed_runtime/hpx_rendezvous_mgr.h"],
        copts = ["-fexceptions"],
        deps = [
              "@local_config_hpx//hpx:hpx",
              "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface"
        ],
        alwayslink = 1,
        visibility = ["//visibility:public"]
    )
)

if_hpx(
    cc_library(
	    name = "hpx_server",
	    srcs = ["distributed_runtime/hpx_server.cc"],
	    hdrs = ["distributed_runtime/hpx_server.h"],
        copts = ["-fexceptions"],
	    deps = [
		    "@local_config_hpx//hpx:hpx",
            ":hpx_rendezvous_mgr",
        "//tensorflow/core:master_proto_cc",
        "//tensorflow/core:worker_proto_cc",              
            ],
        alwayslink = 1,
        visibility = ["//visibility:public"]
    )
)

if_hpx(
    cc_library(
	    name = "core_hpx",
	    srcs = glob(["**/*.cc"], exclude = ["common_runtime/hpx_executor.cc", "distributed_runtime/server/hpx_worker_server.cc", "distributed_runtime/hpx_master.cc", "hpx_runtime.cc", "distributed_runtime/server/*"]),
	    hdrs = glob(["**/*.h"], exclude = ["common_runtime/hpx_executor.h", "distributed_runtime/server/hpx_worker_server.h", "distributed_runtime/hpx_master.h", "hpx_runtime.h", "distributed_runtime/server/*", "distributed_runtime/hpx_tensorflow_serialization.h"]),
        copts = ["-fexceptions"],
	    deps = [
            ":hpx_worker",
            #":hpx_master",
            ":hpx_executor",
            #":hpx_runtime",
            "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
            "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
            "//tensorflow/core/distributed_runtime/rpc:grpc_remote_master",
            "//tensorflow/core/distributed_runtime/rpc:grpc_worker_service",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:master_proto_cc",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core/distributed_runtime:call_options",
            "//tensorflow/core/distributed_runtime:local_master",
            "//tensorflow/core/distributed_runtime:master_interface",
            "//tensorflow/core:worker_proto_cc",
],
        alwayslink = 1,
        visibility = ["//visibility:public"]
    )
)

if_hpx(
    cc_library(
        name = "hpx_serialization",
        hdrs = ["distributed_runtime/hpx_tensorflow_serialization.h"],
    )
)


if_hpx(
    cc_library(
        name = "hpx_worker",
        srcs = ["distributed_runtime/server/hpx_worker_server.cc"],
        hdrs = ["distributed_runtime/server/hpx_worker_server.h", "distributed_runtime/hpx_worker_client.h", "distributed_runtime/hpx_worker.h"],
        copts = ["-fexceptions", '-DHPX_COMPONENT_NAME=hpx_worker_server', '-DHPX_COMPONENT_STRING=\\"hpx_worker_server\\"', '-DHPX_COMPONENT_EXPORTS','-ftemplate-backtrace-limit=0'],
        deps = [
		    "@local_config_hpx//hpx:hpx",
            "//tensorflow/core/distributed_runtime:worker",
            ":hpx_serialization",
            "//tensorflow/core/distributed_runtime/rpc:grpc_worker_service",
            ],
        #linkshared = 1,
        visibility = ["//visibility:public"]    
    )
)

if_hpx(
    cc_library(
        name = "hpx_master",
        srcs = ["distributed_runtime/server/hpx_master_server.cc"],
        hdrs = ["distributed_runtime/server/hpx_master_server.h", "distributed_runtime/hpx_master.h"],
        copts = ["-fexceptions", '-DHPX_COMPONENT_NAME=hpx_master_server', '-DHPX_COMPONENT_STRING=\\"hpx_master_server\\"', '-DHPX_COMPONENT_EXPORTS'],
        deps = [
		    "@local_config_hpx//hpx:hpx",

        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:graph_mgr",
        "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
        "//tensorflow/core/distributed_runtime:worker_interface",
        "//tensorflow/core/distributed_runtime:worker_env",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:core_cpu_internal",
            "//tensorflow/core/distributed_runtime:worker",
            ],
        #linkshared = 1,

        visibility = ["//visibility:public"]    
    )
)
