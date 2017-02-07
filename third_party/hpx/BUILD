# Description
#	HPX is a C++ Standards Library for Concurrency and Parallelism.

licenses(["notice"]) # Boost License

exports_files(["LICENSE"])

cc_library(
    	name = "hpx",
	srcs = glob(["hpx/lib/**/*.so*"]),
    	hdrs = glob(["hpx/include/**/*"]),
	copts = ["-Ithird_party/hpx/hpx/include", "-Ithird_party/boost/boost/include"],
	deps = ["//third_party/boost"],
	defines = ["HPX_NO_VERSION_CHECK"],
    	visibility = ["//visibility:public"],
)
