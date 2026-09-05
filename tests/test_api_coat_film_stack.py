"""
Interface tests for the ``litho_sim.coat`` film-stack API.

Where ``test_films.py`` validates the transfer-matrix *physics* against
closed-form answers, these tests exercise the public surface: what the
package exports, what ``film_stack_from_records`` accepts (material-name
records vs explicit n/k records, exactly as ``ResistConfig.film_stack``
stores them), the structure of the stack that comes back, and how bad
input fails.
"""

from __future__ import annotations

import pytest

from litho_sim.coat import (
    Film,
    FilmStack,
    film_stack_from_records,
    resist_index_from_dill,
)

LAM = 193e-9


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_public_names_are_exported():
    """The four names are importable from the package, not just the module."""
    import litho_sim.coat as coat

    assert set(coat.__all__) == {
        "Film", "FilmStack", "film_stack_from_records", "resist_index_from_dill"
    }
    for name in coat.__all__:
        assert getattr(coat, name) is not None


# ---------------------------------------------------------------------------
# Happy path: records in, stack out
# ---------------------------------------------------------------------------


def test_barc_stack_structure_from_mixed_records():
    """One material-name record plus one explicit n/k record, top-down.

    The stack must come back ordered ambient / resist / layers / substrate,
    with the resist as the first real layer and the record order preserved.
    """
    n_res = resist_index_from_dill(1.70, 0.8, 0.05, LAM)
    records = [
        {"material": "SiARC", "thickness": 30e-9},
        {"n": 1.5, "k": 0.3, "thickness": 60e-9, "name": "SOC-like"},
    ]
    stack = film_stack_from_records(records, 120e-9, n_res)

    assert isinstance(stack, FilmStack)
    assert len(stack.films) == 5  # ambient + resist + 2 records + substrate
    assert [f.name for f in stack.films] == [
        "ambient", "resist", "SiARC", "SOC-like", "Si"
    ]
    assert stack.resist_index == 1
    assert stack.resist is stack.films[1]
    assert stack.resist.thickness == pytest.approx(120e-9)
    assert stack.films[2].thickness == pytest.approx(30e-9)
    assert stack.films[3].n == pytest.approx(1.5 + 0.3j)
    # Semi-infinite boundaries carry zero thickness.
    assert stack.films[0].thickness == 0.0
    assert stack.films[-1].thickness == 0.0
    # And the assembled stack is usable: physical reflectance, not a dud.
    assert 0.0 <= stack.reflectance(LAM) <= 1.0


def test_resist_config_records_flow_straight_in():
    """A ``ResistConfig.film_stack`` list is accepted verbatim.

    This is the exact hand-off the ``optical_model="tmm"`` pipeline performs
    (develop/resist3d.py builds the stack from these records), so the config
    schema and the builder must agree on record shape.
    """
    from litho_sim.core.config import ResistConfig

    cfg = ResistConfig(
        thickness=120e-9,
        n_resist=1.70,
        optical_model="tmm",
        film_stack=[{"n": 1.80, "k": 0.40, "thickness": 38e-9}],
    )
    n_res = resist_index_from_dill(cfg.n_resist, cfg.dill_A, cfg.dill_B, LAM)
    stack = film_stack_from_records(cfg.film_stack, cfg.thickness, n_res)
    assert len(stack.films) == 4
    assert stack.films[2].thickness == pytest.approx(38e-9)
    assert 0.0 <= stack.reflectance(LAM) <= 1.0


def test_empty_records_mean_bare_substrate():
    """The documented default: ``film_stack=[]`` is resist on bare Si."""
    stack = film_stack_from_records([], 100e-9, 1.7 + 0.013j)
    assert [f.name for f in stack.films] == ["ambient", "resist", "Si"]
    assert stack.films[-1].n == pytest.approx(0.883 + 2.778j)


def test_ambient_and_substrate_are_configurable():
    stack = film_stack_from_records(
        [], 100e-9, 1.7 + 0j, ambient_n=1.44 + 0j, substrate="SiO2"
    )
    assert stack.films[0].n == pytest.approx(1.44 + 0j)
    assert stack.films[-1].name == "SiO2"


# ---------------------------------------------------------------------------
# resist_index_from_dill: plausible parameters in, sane index out
# ---------------------------------------------------------------------------


def test_dill_index_preserves_n_and_derives_positive_k():
    n = resist_index_from_dill(1.70, 0.8, 0.05, LAM)
    assert isinstance(n, complex)
    assert n.real == pytest.approx(1.70)
    assert n.imag > 0  # absorbing, per the n + ik storage convention
    # k = (A+B)[1/um] * 1e6 * lam / 4pi — small for a transparent DUV resist.
    assert n.imag < 0.1


def test_dill_index_zero_absorption_is_lossless():
    n = resist_index_from_dill(1.70, 0.0, 0.0, LAM)
    assert n == pytest.approx(1.70 + 0j)


def test_dill_index_k_grows_with_absorption():
    """More Dill absorption must mean more extinction, monotonically."""
    ks = [
        resist_index_from_dill(1.70, A, 0.05, LAM).imag
        for A in (0.0, 0.4, 0.8, 1.6)
    ]
    assert ks == sorted(ks)
    assert ks[0] < ks[-1]


def test_dill_index_scales_with_wavelength():
    """Same alpha over a longer wave means proportionally more k."""
    k193 = resist_index_from_dill(1.70, 0.8, 0.05, 193e-9).imag
    k365 = resist_index_from_dill(1.70, 0.8, 0.05, 365e-9).imag
    assert k365 / k193 == pytest.approx(365.0 / 193.0)


# ---------------------------------------------------------------------------
# Errors and edges
# ---------------------------------------------------------------------------


def test_unknown_material_name_raises_with_the_known_list():
    with pytest.raises(KeyError, match="Unknown material 'unobtainium'"):
        film_stack_from_records(
            [{"material": "unobtainium", "thickness": 10e-9}], 100e-9, 1.7 + 0j
        )


def test_unknown_substrate_raises():
    with pytest.raises(KeyError, match="Unknown material"):
        film_stack_from_records([], 100e-9, 1.7 + 0j, substrate="adamantium")


def test_negative_record_thickness_raises():
    """A negative layer would run the matrix backwards and amplify (R > 1)."""
    with pytest.raises(ValueError, match="negative thickness"):
        film_stack_from_records(
            [{"n": 1.8, "k": 0.4, "thickness": -38e-9}], 100e-9, 1.7 + 0j
        )


def test_negative_resist_thickness_raises():
    with pytest.raises(ValueError, match="resist thickness"):
        film_stack_from_records([], -100e-9, 1.7 + 0j)


def test_record_without_thickness_raises():
    with pytest.raises(KeyError):
        film_stack_from_records([{"material": "SiARC"}], 100e-9, 1.7 + 0j)


def test_zero_thickness_layer_is_allowed_and_inert():
    """thickness=0 is a no-op layer, not an error — the identity matrix."""
    bare = film_stack_from_records([], 100e-9, 1.7 + 0.013j)
    with_zero = film_stack_from_records(
        [{"n": 1.8, "k": 0.4, "thickness": 0.0}], 100e-9, 1.7 + 0.013j
    )
    assert with_zero.reflectance(LAM) == pytest.approx(bare.reflectance(LAM))


def test_film_is_frozen():
    f = Film("layer", 10e-9, 1.5 + 0j)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        f.thickness = 20e-9  # type: ignore[misc]
