licenses(["notice"])  # Boost

package(default_visibility = ["//visibility:public"])

load(
    "@local_config_hpx//hpx:build_defs.bzl",
    "if_hpx_is_configured",
)

config_setting(
    name = "using_hpx",
    values = {
        "define": "using_hpx=true",
    },
)

if_hpx_is_configured(
    cc_library(
	    name = "boost",
	    srcs = glob(["boost/lib/**/*.so*"], exclude = ["boost/lib/**/libboost_python*"]),
	    data = glob(["boost/lib/**/*.so*"], exclude = ["boost/lib/**/libboost_python*"]),
	    hdrs = glob(["boost/include/**/*"]),
        includes = ["boost/" , "boost/include/"],
        copts = ["-fexceptions"],
        alwayslink = 1,
	    visibility = ["//visibility:public"],
    )
)

if_hpx_is_configured(
    cc_library(
	    name = "hpx",
	    srcs = glob(["hpx/lib/**/*.so*"]),
        data = glob(["hpx/lib/**/*.so*"]),
	    hdrs = glob(["hpx/include/**/*"]),
        includes = ["hpx/" , "hpx/include/"],
	    deps = [":boost"],
        copts = ["-fexceptions"],
        alwayslink = 1,
	    visibility = ["//visibility:public"],
    )
)

