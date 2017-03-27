licenses(["notice"])  # Boost license

package(default_visibility = ["//visibility:public"])

load(
    "@local_config_hpx//hpx:build_defs.bzl",
    "if_hpx",
)

load(
    "@local_config_hpx//hpx:hpx_bazel_defs.bzl",
    "hpx_copts",
    "hpx_link_opts",
)

if_hpx(
    cc_library(
	    name = "boost",
	    srcs = glob(["boost/lib/**/*.so*"], exclude = ["boost/lib/**/libboost_python*"]),
	    data = glob(["boost/lib/**/*.so*"], exclude = ["boost/lib/**/libboost_python*"]),
	    hdrs = glob(["boost/include/**/*"]),
        includes = ["boost/" , "boost/include/"],
        copts = ["-fexceptions"],
        linkstatic = 1,
	    visibility = ["//visibility:public"],
    )
)

if_hpx(
    cc_library(
	    name = "hpx",
	    srcs = glob(["hpx/lib/**/*.so*"]),
        data = glob(["hpx/lib/**/*.so*"]),
	    hdrs = glob(["hpx/include/**/*"]),
        includes = ["hpx/" , "hpx/include/"],
	    deps = [":boost"],
        copts = ["-fexceptions"] + hpx_copts(),
        linkopts = [":boost"],
        linkstatic = 1,
	    visibility = ["//visibility:public"],
    )
)

