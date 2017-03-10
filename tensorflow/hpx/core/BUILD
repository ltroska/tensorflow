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
            ":hpx_runtime",
            ],
        alwayslink = 1,
        visibility = ["//visibility:public"]
    )
)

if_hpx(
    cc_library(
	    name = "hpx_worker_cache",
	    srcs = ["distributed_runtime/hpx_worker_cache.cc"],
	    hdrs = ["distributed_runtime/hpx_worker_cache.h"],
        copts = ["-fexceptions"],
	    deps = [
		    "@local_config_hpx//hpx:hpx",
            ":hpx_worker",
            ":hpx_runtime",
            "//tensorflow/core:lib",
            "//tensorflow/core/distributed_runtime:worker_cache",
            "//tensorflow/core/distributed_runtime:worker_cache_logger",
            "//tensorflow/core/distributed_runtime:worker_cache_partial",
            "//tensorflow/core/distributed_runtime:worker_interface", 
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
            ":hpx_worker",
            ":hpx_master",
            ":hpx_worker_cache",
            ":hpx_runtime",
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
)

if_hpx(
    cc_library(
	    name = "hpx_session",
	    srcs = ["distributed_runtime/hpx_session.cc"],
	    hdrs = ["distributed_runtime/hpx_session.h"],
        copts = ["-fexceptions"],
	    deps = [
		    "@local_config_hpx//hpx:hpx",
            ":hpx_master",
            ":hpx_runtime",
            "//tensorflow/core/distributed_runtime:master_interface",
            "//tensorflow/core/distributed_runtime:local_master",
            "//tensorflow/core:master_proto_cc",
          ],
        alwayslink = 1,
        visibility = ["//visibility:public"]
    )
)

if_hpx(
    cc_library(
	    name = "core_hpx",
	    srcs = [],
	    hdrs = [],
        copts = ["-fexceptions"],
	    deps = [
            ":hpx_server",
            ":hpx_graph_runner",
            ":hpx_session",
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
              ":hpx_serialization",
              "//tensorflow/core/distributed_runtime:worker",
              "//tensorflow/core/distributed_runtime/rpc:grpc_tensor_coding",
            ],
        #linkshared = 1,
        visibility = ["//visibility:public"]    
    )
)

if_hpx(
    cc_library(
        name = "hpx_master",
        srcs = ["distributed_runtime/server/hpx_master_server.cc"],
        hdrs = ["distributed_runtime/server/hpx_master_server.h", "distributed_runtime/hpx_master_client.h", "distributed_runtime/hpx_master.h"],
        copts = ["-fexceptions", '-DHPX_COMPONENT_NAME=hpx_master_server', '-DHPX_COMPONENT_STRING=\\"hpx_master_server\\"', '-DHPX_COMPONENT_EXPORTS','-ftemplate-backtrace-limit=0'],
        deps = [
		    "@local_config_hpx//hpx:hpx",
            "//tensorflow/core/distributed_runtime:master",
            ":hpx_serialization",
            ],
        #linkshared = 1,
        visibility = ["//visibility:public"]    
    )
)
