load(
    "@local_config_hpx//hpx:build_defs.bzl",
    "if_hpx",
)

if_hpx(
    cc_library(
	    name = "core_hpx",
	    srcs = glob(["hpx*"]),
	    hdrs = glob(["hpx*"]),
        copts = ["-fexceptions"],
	    deps = [
		    "//tensorflow/core:core_cpu",
		    "@local_config_hpx//hpx:hpx",
		    "@local_config_hpx//hpx:boost",
		    ],
        alwayslink = 1,
        visibility = ["//visibility:public"]
    )
)
