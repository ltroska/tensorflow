load(
    "@local_config_hpx//hpx:build_defs.bzl",
    "tf_hpx_library",
)

tf_hpx_library(
        name = "eigenvector",
        srcs = ["eigenvector.cc"],
        deps = [
            "//tensorflow/cc:cc_ops",
            "//tensorflow/core:all_kernels",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:tensorflow",
            "//tensorflow/cc:scope",
            ],
        hpx_deps = [
	        "//tensorflow/hpx:hpx_graph_runner",
        ],
)

tf_hpx_library(
        name = "load_graph",
        srcs = ["load_graph.cc"],
        deps = [
            "//tensorflow/cc:cc_ops",
            "//tensorflow/core:all_kernels",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
	        "//tensorflow/core:tensorflow",
            "//tensorflow/cc:scope",
            ],
        hpx_deps = [
	        "//tensorflow/hpx:hpx_graph_runner",
        ],
)
