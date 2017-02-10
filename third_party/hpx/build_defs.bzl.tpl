# Macros for building HPX code.
def if_hpx(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with HPX.

    Returns a select statement which evaluates to if_true if we're building
    with HPX enabled.  Otherwise, the select statement evaluates to if_false.

    """
    return select({
        "@local_config_hpx//hpx:using_hpx": if_true,
        "//conditions:default": if_false
    })

def hpx_is_configured():
    return %{hpx_is_configured}


def if_hpx_is_configured(x):
    if hpx_is_configured():
        return x
    return []

        

