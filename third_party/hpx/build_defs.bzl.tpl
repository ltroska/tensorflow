# Macros for building HPX code.
load("@local_config_hpx//hpx:hpx_bazel_defs.bzl", "hpx_copts")

def hpx_is_configured():
    return %{hpx_is_configured}

def if_hpx(x):
    if hpx_is_configured():
        return x
    return []

def tf_hpx_library(deps=None, hpx_deps=None, copts=None, **kwargs):
  """Generate a cc_library with a conditional set of HPX dependencies.

  When the library is built with --config=hpx:

  - both deps and hpx_deps are used as dependencies
  - the HPX runtime is added as a dependency (if necessary)
  - The library additionally passes -DHAVE_HPX=1 to the list of copts

  Args:
  - cuda_deps: BUILD dependencies which will be linked if and only if:
      '--config=cuda' is passed to the bazel command line.
  - deps: dependencies which will always be linked.
  - copts: copts always passed to the cc_library.
  - kwargs: Any other argument to cc_library.
  """
  if not deps:
    deps = []
  if not hpx_deps:
    hpx_deps = []
  if not copts:
    copts = []
  if "srcs" in kwargs:
    kwargs["srcs"] = if_hpx(kwargs["srcs"])
  if "hdrs" in kwargs:
    kwargs["hdrs"] = if_hpx(kwargs["hdrs"])

  native.cc_library(
      deps = deps + if_hpx(hpx_deps + [
          "@local_config_hpx//hpx:hpx"
      ]),
      copts = copts + if_hpx(["-DHAVE_HPX=1"] + hpx_copts()),
      **kwargs)
        

