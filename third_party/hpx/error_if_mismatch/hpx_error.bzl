load(
    "@local_config_hpx//hpx:build_defs.bzl",
    "hpx_is_configured",
)

def fail_if_hpx_mismatch():
    if not hpx_is_configured():
        fail("ERROR: Building with --config=hpx but TensorFlow is not configured " +
        "to build with HPX support. Please re-run ./configure and enter 'Y' " +
        "at the prompt to build with HPX support.")
