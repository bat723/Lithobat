"""
The defect-study subpackage: kept importable and honest.

``litho_sim.ml`` is the original autoencoder study. Its defect generators
are plain numpy and are used by the dataset script; the models need torch,
so those tests skip cleanly where it is absent rather than failing.
"""

from __future__ import annotations

import numpy as np
import pytest

from litho_sim.ml.defects import add_bridge_defect, add_line_roughness, add_particle_defect


@pytest.fixture
def clean():
    x = np.zeros((64, 64))
    x[:, 20:44] = 1.0          # one bright line
    return x


def test_particle_defect_is_local_and_deterministic(clean):
    # Placed in the dark field: on the bright line a particle changes nothing,
    # which is correct and is why the position is pinned here.
    a = add_particle_defect(clean, particle_radius_px=4.0, position=(10, 8))
    b = add_particle_defect(clean, particle_radius_px=4.0, position=(10, 8))
    assert a.shape == clean.shape
    np.testing.assert_array_equal(a, b)
    changed = a != clean
    assert 0 < changed.sum() < 0.1 * clean.size, "a particle is a small blemish"
    assert changed[10, 8] and not changed[40, 50]


def test_line_roughness_perturbs_only_the_edges(clean):
    rough = add_line_roughness(clean, roughness_amplitude=0.2, seed=2)
    assert rough.shape == clean.shape
    assert not np.array_equal(rough, clean)
    assert rough.min() >= 0.0 and rough.max() <= 1.0 + 1e-12


def test_bridge_defect_crosses_the_space(clean):
    bridged = add_bridge_defect(clean, row=32, bridge_width_px=4, seed=3)
    assert bridged.shape == clean.shape
    assert bridged[32].sum() > clean[32].sum(), "the bridge adds material on its row"


@pytest.mark.parametrize("kind", ("conv", "simple"))
def test_autoencoders_reproduce_their_input_shape(kind):
    torch = pytest.importorskip("torch")
    from litho_sim.ml.models import build_model

    model = build_model(kind).eval()
    x = torch.rand(2, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == tuple(x.shape)


def test_unknown_model_type_is_refused():
    pytest.importorskip("torch")
    from litho_sim.ml.models import build_model

    with pytest.raises(ValueError):
        build_model("transformer")
