import json
import os
import warnings
import tempfile
from contextlib import contextmanager

import numpy as np
import numpy.testing as npt
import pytest
import hypothesis

from scipy._lib._fpumode import get_fpu_mode
from scipy._lib._testutils import FPUModeChangeWarning
from scipy._lib._array_api import SCIPY_ARRAY_API, SCIPY_DEVICE
from scipy._lib import _pep440

try:
    from scipy_doctest.conftest import dt_config
    HAVE_SCPDT = True
except ModuleNotFoundError:
    HAVE_SCPDT = False


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Tests that are very slow.")
    config.addinivalue_line("markers", "xslow: mark test as extremely slow (not run unless explicitly requested)")
    config.addinivalue_line("markers", "xfail_on_32bit: mark test as failing on 32-bit platforms")
    try:
        import pytest_timeout
    except ImportError:
        config.addinivalue_line("markers", 'timeout: mark a test for a non-default timeout')
    try:
        from pytest_fail_slow import parse_duration
    except ImportError:
        config.addinivalue_line("markers", 'fail_slow: mark a test for a non-default timeout failure')
    config.addinivalue_line("markers", "skip_xp_backends(*backends, reasons=None, np_only=False, cpu_only=False, exceptions=None): mark the desired skip configuration for the `skip_xp_backends` fixture.")
    config.addinivalue_line("markers", "xfail_xp_backends(*backends, reasons=None, np_only=False, cpu_only=False, exceptions=None): mark the desired xfail configuration for the `xfail_xp_backends` fixture.")


def pytest_runtest_setup(item):
    mark = item.get_closest_marker("xslow")
    if mark is not None:
        try:
            v = int(os.environ.get('SCIPY_XSLOW', '0'))
        except ValueError:
            v = False
        if not v:
            pytest.skip("Very slow test; set environment variable SCIPY_XSLOW=1 to run it")
    
    mark = item.get_closest_marker("xfail_on_32bit")
    if mark is not None and np.intp(0).itemsize < 8:
        pytest.xfail(f'Fails on our 32-bit test platform(s): {mark.args[0]}')

    with npt.suppress_warnings() as sup:
        sup.filter(pytest.PytestUnraisableExceptionWarning)
        try:
            from threadpoolctl import threadpool_limits
            HAS_THREADPOOLCTL = True
        except ImportError:
            HAS_THREADPOOLCTL = False

        if HAS_THREADPOOLCTL:
            try:
                xdist_worker_count = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', 0))
                if not os.getenv('OMP_NUM_THREADS'):
                    max_openmp_threads = os.cpu_count() // 2
                    threads_per_worker = max(max_openmp_threads // xdist_worker_count, 1)
                    try:
                        threadpool_limits(threads_per_worker, user_api='blas')
                    except AttributeError:
                        pass
            except ValueError:
                pass


@pytest.fixture(scope="function", autouse=True)
def check_fpu_mode(request):
    """
    Check FPU mode was not changed during the test.
    """
    old_mode = get_fpu_mode()
    yield
    new_mode = get_fpu_mode()
    if old_mode != new_mode:
        warnings.warn(f"FPU mode changed from {old_mode:#x} to {new_mode:#x} during the test", category=FPUModeChangeWarning)


# Array API backend handling
xp_available_backends = {'numpy': np}

if SCIPY_ARRAY_API and isinstance(SCIPY_ARRAY_API, str):
    try:
        import array_api_strict
        xp_available_backends.update({'array_api_strict': array_api_strict})
        if _pep440.parse(array_api_strict.__version__) < _pep440.Version('2.0'):
            raise ImportError("array-api-strict must be >= version 2.0")
        array_api_strict.set_array_api_strict_flags(api_version='2023.12')
    except ImportError:
        pass

    try:
        import torch
        xp_available_backends.update({'pytorch': torch})
        torch.set_default_device(SCIPY_DEVICE)
    except ImportError:
        pass

    try:
        import cupy
        xp_available_backends.update({'cupy': cupy})
    except ImportError:
        pass

    try:
        import jax.numpy
        xp_available_backends.update({'jax.numpy': jax.numpy})
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_device", jax.devices(SCIPY_DEVICE)[0])
    except ImportError:
        pass

    if SCIPY_ARRAY_API.lower() not in ("1", "true"):
        SCIPY_ARRAY_API_ = json.loads(SCIPY_ARRAY_API)
        if 'all' in SCIPY_ARRAY_API_:
            pass
        else:
            try:
                xp_available_backends = {backend: xp_available_backends[backend] for backend in SCIPY_ARRAY_API_}
            except KeyError:
                raise ValueError(f"'--array-api-backend' must be in {xp_available_backends.keys()}")

if 'cupy' in xp_available_backends:
    SCIPY_DEVICE = 'cuda'

array_api_compatible = pytest.mark.parametrize("xp", xp_available_backends.values())

skip_xp_invalid_arg = pytest.mark.skipif(SCIPY_ARRAY_API, reason='Test involves masked arrays, object arrays, or other types that are not valid input when `SCIPY_ARRAY_API` is used.')


@pytest.fixture
def skip_xp_backends(xp, request):
    """See the `skip_or_xfail_xp_backends` docstring."""
    if "skip_xp_backends" not in request.keywords:
        return
    backends = request.keywords["skip_xp_backends"].args
    kwargs = request.keywords["skip_xp_backends"].kwargs
    skip_or_xfail_xp_backends(xp, backends, kwargs, skip_or_xfail='skip')


@pytest.fixture
def xfail_xp_backends(xp, request):
    """See the `skip_or_xfail_xp_backends` docstring."""
    if "xfail_xp_backends" not in request.keywords:
        return
    backends = request.keywords["xfail_xp_backends"].args
    kwargs = request.keywords["xfail_xp_backends"].kwargs
    skip_or_xfail_xp_backends(xp, backends, kwargs, skip_or_xfail='xfail')


def skip_or_xfail_xp_backends(xp, backends, kwargs, skip_or_xfail='skip'):
    """
    Skip based on the ``skip_xp_backends`` or ``xfail_xp_backends`` marker.
    
    See the "Support for the array API standard" docs page for usage examples.

    Parameters
    ----------
    *backends : tuple
        Backends to skip, e.g. ``("array_api_strict", "torch")``.
    reasons : list, optional
        A list of reasons for each skip. When ``np_only`` is ``True``, this should be a singleton list.
    np_only : bool, optional
        When ``True``, the test is skipped for all backends other than the default NumPy backend.
    cpu_only : bool, optional
        When ``True``, the test is skipped/x-failed on non-CPU devices.
    exceptions : list, optional
        A list of exceptions for use with `cpu_only` or `np_only`.
    """
    skip_or_xfail = getattr(pytest, skip_or_xfail)

    np_only = kwargs.get("np_only", False)
    cpu_only = kwargs.get("cpu_only", False)
    exceptions = kwargs.get("exceptions", [])

    if np_only and cpu_only:
        raise ValueError("At most one of `np_only` and `cpu_only` should be provided")
    if exceptions and not (cpu_only or np_only):
        raise ValueError("`exceptions` is only valid alongside `cpu_only` or `np_only`")

    if np_only:
        reasons = kwargs.get("reasons", ["do not run with non-NumPy backends."])
        if len(reasons) > 1:
            raise ValueError("Please provide a singleton list to `reasons` when using `np_only`")
        reason = reasons[0]
        if xp.__name__ != 'numpy' and xp.__name__ not in exceptions:
            skip_or_xfail(reason=reason)
        return

    if cpu_only:
        reason = "No array-agnostic implementation or delegation available for this backend and device"
        exceptions = [] if exceptions is None else exceptions
        if SCIPY_ARRAY_API and SCIPY_DEVICE != 'cpu':
            if xp.__name__ == 'cupy' and 'cupy' not in exceptions:
                skip_or_xfail(reason=reason)
            elif xp.__name__ == 'torch' and 'torch' not in exceptions:
                if 'cpu' not in xp.empty(0).device.type:
                    skip_or_xfail(reason=reason)
            elif xp.__name__ == 'jax.numpy' and 'jax.numpy' not in exceptions:
                if 'cpu' not in xp.device():
                    skip_or_xfail(reason=reason)
            elif xp.__name__ == 'array_api_strict' and 'array_api_strict' not in exceptions:
                skip_or_xfail(reason=reason)
        else:
            if xp.__name__ == 'cupy' and 'cupy' not in exceptions:
                skip_or_xfail(reason=reason)
            elif xp.__name__ == 'torch' and 'torch' not in exceptions:
                if 'cpu' not in xp.device():
                    skip_or_xfail(reason=reason)
            elif xp.__name__ == 'jax.numpy' and 'jax.numpy' not in exceptions:
                if 'cpu' not in xp.device():
                    skip_or_xfail(reason=reason)
            elif xp.__name__ == 'array_api_strict' and 'array_api_strict' not in exceptions:
                skip_or_xfail(reason=reason)
        return

    if xp.__name__ in backends:
        reason = kwargs.get("reasons", ["Not available for this backend."])[0]
        skip_or_xfail(reason=reason)
