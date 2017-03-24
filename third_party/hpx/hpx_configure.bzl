# -*- Python -*-
"""hpx autoconfiguration.
`hpx_configure` depends on the following environment variables:

  * HOST_CXX_COMPILER:  The host C++ compiler
  * HOST_C_COMPILER:    The host C compiler
  * HPX_PATH: The path to HPX.
  * BOOST_PATH: The path to Boost.
"""

_HOST_CXX_COMPILER = "HOST_CXX_COMPILER"
_HOST_C_COMPILER= "HOST_C_COMPILER"
_HPX_PATH = "HPX_PATH"

load(
    ":hpx_bazel_defs.bzl",
    "boost_path",
)

def _enable_hpx(repository_ctx):
  if "TF_NEED_HPX" in repository_ctx.os.environ:
    enable_hpx = repository_ctx.os.environ["TF_NEED_HPX"].strip()
    return enable_hpx == "1"
  return False
  
def auto_configure_fail(msg):
  """Output failure message when auto configuration fails."""
  red = "\033[0;31m"
  no_color = "\033[0m"
  fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))
# END cc_configure common functions (see TODO above).

def find_c(repository_ctx):
  """Find host C compiler."""
  c_name = "gcc"
  if _HOST_C_COMPILER in repository_ctx.os.environ:
    c_name = repository_ctx.os.environ[_HOST_C_COMPILER].strip()
  if c_name.startswith("/"):
    return c_name
  c = repository_ctx.which(c_name)
  if c == None:
    fail("Cannot find C compiler, please correct your path.")
  return c

def find_cc(repository_ctx):
  """Find host C++ compiler."""
  cc_name = "g++"
  if _HOST_CXX_COMPILER in repository_ctx.os.environ:
    cc_name = repository_ctx.os.environ[_HOST_CXX_COMPILER].strip()
  if cc_name.startswith("/"):
    return cc_name
  cc = repository_ctx.which(cc_name)
  if cc == None:
    fail("Cannot find C++ compiler, please correct your path.")
  return cc

def find_hpx_root(repository_ctx):
  """Find HPX root directory."""
  hpx_name = ""
  if _HPX_PATH in repository_ctx.os.environ:
    hpx_name = repository_ctx.os.environ[_HPX_PATH].strip()
  if hpx_name.startswith("/"):
    return hpx_name
  fail( "Cannot find HPX root directory, please correct your path")

def find_boost_root(repository_ctx):
  """Find Boost root directory."""
  return boost_path()

def _check_lib(repository_ctx, toolkit_path, lib):
  """Checks if lib exists under hpx_toolkit_path or fail if it doesn't.

  Args:
    repository_ctx: The repository context.
    toolkit_path: The toolkit directory containing the libraries.
    ib: The library to look for under toolkit_path.
  """
  lib_path = toolkit_path + "/" + lib
  if not repository_ctx.path(lib_path).exists:
    auto_configure_fail("Cannot find %s" % lib_path)

def _check_dir(repository_ctx, directory):
  """Checks whether the directory exists and fail if it does not.

  Args:
    repository_ctx: The repository context.
    directory: The directory to check the existence of.
  """
  if not repository_ctx.path(directory).exists:
    auto_configure_fail("Cannot find dir: %s" % directory)

def _symlink_dir(repository_ctx, src_dir, dest_dir):
  """Symlinks all the files in a directory.

  Args:
    repository_ctx: The repository context.
    src_dir: The source directory.
    dest_dir: The destination directory to create the symlinks in.
  """
  src_path = repository_ctx.path(src_dir)
  repository_ctx.symlink(src_path, dest_dir)

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl.replace(":", "/")
  repository_ctx.template(
      out,
      Label("//third_party/%s.tpl" % tpl),
      substitutions)

def _file(repository_ctx, label):
  repository_ctx.template(
      label.replace(":", "/"),
      Label("//third_party/%s" % label),
      {})

_DUMMY_HPX_BZL_FILE = """
def hpx_component_copts(component_name):
    return []

def hpx_application_copts():
    return []          

def hpx_copts():
    return []

def hpx_link_opts():
    return []

def hpx_path():
    return ""

def boost_path():
    return ""
"""

def _create_dummy_repository(repository_ctx):
  # Set up BUILD file for hpx
  _tpl(repository_ctx, "hpx:build_defs.bzl", {"%{hpx_is_configured}": "False"})
  _tpl(repository_ctx, "hpx:BUILD")
  repository_ctx.file("hpx/hpx_bazel_defs.bzl", _DUMMY_HPX_BZL_FILE)
  _file(repository_ctx, "hpx:LICENSE")
  _file(repository_ctx, "hpx:platform.bzl")


def _hpx_autoconf_imp(repository_ctx):
  """Implementation of the hpx_autoconf rule."""
  if not _enable_hpx(repository_ctx):
    _create_dummy_repository(repository_ctx)
  else:
    hpx_root = find_hpx_root(repository_ctx);
    _check_dir(repository_ctx, hpx_root)

    boost_root = find_boost_root(repository_ctx);
    _check_dir(repository_ctx, boost_root)

    # copy template files
    _tpl(repository_ctx, "hpx:build_defs.bzl", {"%{hpx_is_configured}": "True"})
    _tpl(repository_ctx, "hpx:BUILD")
    _file(repository_ctx, "hpx:platform.bzl")
    _file(repository_ctx, "hpx:LICENSE")
    _file(repository_ctx, "hpx:hpx_bazel_defs.bzl")

    # symlink libraries
    _symlink_dir(repository_ctx, hpx_root + "/lib", "hpx/hpx/lib")
    _symlink_dir(repository_ctx, hpx_root + "/include", "hpx/hpx/include")
    _symlink_dir(repository_ctx, boost_root + "/lib", "hpx/boost/lib")
    _symlink_dir(repository_ctx, boost_root + "/include", "hpx/boost/include")

hpx_configure = repository_rule(
  implementation = _hpx_autoconf_imp,
  local = True,
)
"""Detects and configures the hpx toolchain.

Add the following to your WORKSPACE FILE:

```python
hpx_configure(name = "local_config_hpx")
```

Args:
  name: A unique name for this workspace rule.
"""
