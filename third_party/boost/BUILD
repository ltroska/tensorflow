# Description:
#	The Boost library collection (http://www.boost.org)

licenses(["notice"]) # Boost License

exports_files(["LICENSE"])

cc_library(
	name = "boost",
	srcs = glob(["boost/lib/**/*.so*"], exclude = ["boost/lib/**/libboost_python*"]),
    	hdrs = glob(["boost/include/**/*"]),
	copts = ["-Ithird_party/boost/boost/include"],
    	visibility = ["//visibility:public"],
)
