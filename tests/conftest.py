"""Shared fixtures.

The one thing here is cache isolation, and it is a correctness fixture rather
than a tidiness one.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_m3d_cache(tmp_path, monkeypatch):
    """Point the near-field library cache at a throwaway directory.

    Two reasons, and the second is the one that matters:

    * A test run should not fill — or evict from — the user's real
      ``presets/m3d`` cache, which represents minutes of solving.
    * A stored library is keyed on the *configuration*, not on the solver that
      produced it. Editing :mod:`litho_sim.expose.m3d.yee` and re-running the
      suite would otherwise compare the new solver against fields cached by the
      old one, and the tests that exist to catch a broken solve would pass on
      last week's answer. ``LIBRARY_CACHE_VERSION`` covers deliberate format
      changes; this covers the ordinary edit-and-test loop.
    """
    from litho_sim.expose.m3d import provider

    monkeypatch.setattr(provider, "LIBRARY_CACHE_DIR", tmp_path / "m3d")
    provider.clear_library_cache()
    yield
    provider.clear_library_cache()


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every pyplot figure a test left open.

    The figure-returning functions go through pyplot so that a CLI run can
    still ``plt.show()`` them, which means each one stays registered until
    closed. Left alone, a plotting test file crosses matplotlib's 20-figure
    warning threshold and the suite ends with a memory warning.
    """
    yield
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a core dependency
        return
    plt.close("all")
