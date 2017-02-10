load(
    "@local_config_hpx//hpx:build_defs.bzl",
    "if_hpx",
)

if_hpx(
    cc_binary(
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
	        "//tensorflow/hpx/core:core_hpx",
        ],
    )
)

if_hpx(
    cc_binary(
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
	        "//tensorflow/hpx/core:core_hpx",
        ],
    )
)
