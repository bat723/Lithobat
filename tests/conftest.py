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
