"""
Wafer film stack — the 3-D material state that process steps mutate.

The stack is a ``(nz, ny, nx)`` array of ``uint8`` material IDs, with
``0`` reserved for vacuum.  ``iz = 0`` is the **bottom** (substrate side),
so ``stack.mat[iz]`` is a horizontal slice and the lateral axes match the
``(ny, nx)`` convention every mask and aerial image already uses.

Why voxels rather than a list of per-layer height fields
--------------------------------------------------------
Height fields are cheaper and render for free, but they cannot express the
one thing self-aligned patterning is built on.  After a conformal spacer is
deposited over a mandrel and the mandrel is pulled, a single ``(x, y)``
column reads::

    vacuum → spacer → vacuum → spacer

That is two disjoint z-intervals in one column — a freestanding wall.  A
height field stores one interval per column and simply cannot hold it.
Voxels can, and the memory cost is negligible: a 128×128×160 stack is
2.6 MB.

Height fields are not lost, they are *derived*: :meth:`Stack.height_fields`
extracts them on demand for the 3-D renderer, which wants exactly that
(see :mod:`litho_sim.viz.viz3d`).

Conventions
-----------
* Lateral pixel size comes from :class:`~litho_sim.core.config.GridConfig`.
* Vertical voxel height ``dz`` is independent and usually finer — it has to
  resolve the standing-wave period, λ/(2·n_resist) ≈ 57 nm at 193 nm.
* All physical quantities are in metres.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.ndimage import distance_transform_edt

from litho_sim.core.config import GridConfig
from litho_sim.wafer.etch_profile import EtchProfile
from litho_sim.wafer.materials import (
    _BY_ID,
    VACUUM,
    Material,
    get_material,
)

logger = logging.getLogger(__name__)

#: A material etching this slowly or slower counts as a stop layer. Consulted
#: only by the geometric profile path, which has no cost integral to arrest it.
#: Inclusive on purpose: the interface's own default stop rate is the value most
#: often compared against it, and a threshold a user cannot land on is a trap.
_STOP_RATE = 0.01


def _names_of(ids: Iterable[int]) -> str:
    """Material IDs as a stable, readable list, for history lines."""
    return ", ".join(sorted(get_material(int(i)).name for i in ids))


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------


@dataclass
class Stack:
    """A 3-D wafer film stack.

    Parameters
    ----------
    mat : NDArray[np.uint8]
        Material IDs, shape ``(nz, ny, nx)``.  ``iz=0`` is the bottom.
    grid : GridConfig
        Lateral grid (supplies ``pixel_size``).
    dz : float
        Vertical voxel height [m].
    materials : dict
        ``{id: Material}`` for every material present.  Defaults to the
        built-in library so saved stacks render without extra setup.
    history : list of str
        Human-readable log of the steps applied so far.
    """

    mat: NDArray[np.uint8]
    grid: GridConfig
    dz: float = 2e-9
    materials: dict[int, Material] = field(default_factory=lambda: dict(_BY_ID))
    history: list = field(default_factory=list)
    #: Extra per-voxel volumes carried alongside ``mat``, keyed by name and
    #: always the same shape. See :meth:`add_field`.
    fields: dict[str, NDArray] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def blank(
        cls,
        grid: GridConfig,
        dz: float = 2e-9,
        substrate: str = "Si",
        substrate_thickness: float = 40e-9,
        headroom: float = 200e-9,
    ) -> Stack:
        """Create a stack containing only a substrate slab plus empty space.

        Parameters
        ----------
        grid : GridConfig
            Lateral grid definition.
        dz : float
            Vertical voxel height [m].
        substrate : str
            Substrate material name.
        substrate_thickness : float
            Substrate slab thickness [m].
        headroom : float
            Vacuum height reserved above the substrate [m].  Grows
            automatically when deposition needs more.
        """
        n_sub = max(int(round(substrate_thickness / dz)), 1)
        n_head = max(int(round(headroom / dz)), 1)
        mat = np.zeros((n_sub + n_head, grid.n_pixels, grid.n_pixels), dtype=np.uint8)
        mat[:n_sub] = get_material(substrate).id
        st = cls(mat=mat, grid=grid, dz=dz)
        st.history.append(f"blank: {substrate} {substrate_thickness*1e9:.0f} nm")
        logger.info(
            "Stack.blank: %dx%dx%d voxels (%.2f MB), dz=%.1f nm",
            *mat.shape, mat.nbytes / 1e6, dz * 1e9,
        )
        return st

    def copy(self) -> Stack:
        """Deep copy, so snapshots taken between process steps stay independent."""
        return replace(
            self,
            mat=self.mat.copy(),
            materials=dict(self.materials),
            history=list(self.history),
            fields={k: v.copy() for k, v in self.fields.items()},
        )

    # ------------------------------------------------------------------
    # Shape / coordinates
    # ------------------------------------------------------------------

    @property
    def nz(self) -> int:
        return int(self.mat.shape[0])

    @property
    def shape_xy(self) -> tuple[int, int]:
        return int(self.mat.shape[1]), int(self.mat.shape[2])

    @property
    def pixel_size(self) -> float:
        return self.grid.pixel_size

    def _n_voxels(self, thickness: float) -> int:
        """Convert a thickness [m] to a whole number of voxels (at least 1)."""
        return max(int(round(thickness / self.dz)), 1)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def solid(self) -> NDArray[np.bool_]:
        """Boolean mask of every non-vacuum voxel."""
        return self.mat != VACUUM

    def surface_index(self) -> NDArray[np.int32]:
        """Index of the topmost solid voxel per column, or ``-1`` if empty.

        Returns
        -------
        NDArray[np.int32]
            Shape ``(ny, nx)``.
        """
        solid = self.solid()
        any_solid = solid.any(axis=0)
        # argmax on the reversed axis finds the first solid from the top.
        top_from_end = np.argmax(solid[::-1], axis=0)
        idx = (self.nz - 1 - top_from_end).astype(np.int32)
        return np.where(any_solid, idx, np.int32(-1))

    def top_height(self) -> NDArray[np.float64]:
        """Height of the top surface per column [m]; 0 where the column is empty."""
        idx = self.surface_index()
        return np.where(idx >= 0, (idx + 1) * self.dz, 0.0)

    def top_material(self) -> NDArray[np.uint8]:
        """Material ID exposed at the top of each column (``0`` where empty)."""
        idx = self.surface_index()
        safe = np.where(idx >= 0, idx, 0)
        ny, nx = self.shape_xy
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        top = self.mat[safe, yy, xx]
        return np.where(idx >= 0, top, np.uint8(VACUUM)).astype(np.uint8)

    def thickness_of(self, ref: str | int | Material) -> NDArray[np.float64]:
        """Total thickness of one material per column [m]."""
        mid = get_material(ref).id
        return (self.mat == mid).sum(axis=0) * self.dz

    def volume_fraction(self, ref: str | int | Material) -> float:
        """Fraction of all voxels occupied by one material."""
        return float((self.mat == get_material(ref).id).mean())

    def present_materials(self) -> list:
        """Materials actually present, bottom-most first, excluding vacuum."""
        ids = [int(i) for i in np.unique(self.mat) if int(i) != VACUUM]
        # Order by mean height so the viewer legend reads bottom-to-top.
        def mean_z(mid: int) -> float:
            zs = np.nonzero((self.mat == mid).any(axis=(1, 2)))[0]
            return float(zs.mean()) if zs.size else 0.0
        ids.sort(key=mean_z)
        return [self.materials.get(i, _BY_ID.get(i, Material(i, f"mat{i}"))) for i in ids]

    def cross_section(self, axis: str = "y", index: int | None = None) -> NDArray[np.uint8]:
        """Extract a vertical slice for 2-D plotting.

        Parameters
        ----------
        axis : str
            ``"y"`` cuts at constant y and returns the x–z plane (the usual
            view for vertical lines); ``"x"`` cuts at constant x.
        index : int, optional
            Row/column to cut at.  Defaults to the centre.

        Returns
        -------
        NDArray[np.uint8]
            Shape ``(nz, n)`` with ``iz=0`` at the bottom.
        """
        ny, nx = self.shape_xy
        if axis == "y":
            return self.mat[:, ny // 2 if index is None else index, :]
        if axis == "x":
            return self.mat[:, :, nx // 2 if index is None else index]
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")

    def height_fields(
        self, ref: str | int | Material
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Bottom and top surfaces of one material, for surface rendering.

        Returns
        -------
        (z_bottom, z_top, present) : tuple of NDArray
            Heights [m] of the lowest and highest voxel of this material in
            each column, and a mask of columns where it appears at all.

        Notes
        -----
        This collapses a column to a single interval.  That is exact for
        ordinary films, and an approximation only where a material occupies
        two disjoint intervals in one column — the freestanding-spacer case.
        Use :meth:`occupancy` with a marching-cubes renderer when that
        matters.
        """
        mid = get_material(ref).id
        occ = self.mat == mid
        present = occ.any(axis=0)
        first = np.argmax(occ, axis=0)
        last = self.nz - 1 - np.argmax(occ[::-1], axis=0)
        z_bot = np.where(present, first * self.dz, np.nan)
        z_top = np.where(present, (last + 1) * self.dz, np.nan)
        return z_bot, z_top, present

    def occupancy(
        self, ref: str | int | Material, smooth: float = 0.0
    ) -> NDArray[np.float32]:
        """Occupancy field of one material, optionally smoothed.

        Smoothing is what lets marching cubes recover a sub-voxel surface
        from a binary field instead of a staircase.
        """
        occ = (self.mat == get_material(ref).id).astype(np.float32)
        if smooth > 0.0:
            from scipy.ndimage import gaussian_filter

            occ = gaussian_filter(occ, sigma=smooth)
        return occ

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def ensure_headroom(self, extra: float) -> None:
        """Pad the top of the stack with vacuum so a deposit has room to grow."""
        need = self._n_voxels(extra)
        free = self.nz - 1 - int(self.surface_index().max(initial=-1))
        if free >= need:
            return
        pad = need - free
        ny, nx = self.shape_xy
        self.mat = np.concatenate(
            [self.mat, np.zeros((pad, ny, nx), dtype=np.uint8)], axis=0
        )
        # Auxiliary volumes must grow with `mat` or they silently desynchronise
        # — same shape, different nz, and every later index is wrong.
        for name, arr in self.fields.items():
            self.fields[name] = np.concatenate(
                [arr, np.zeros((pad, ny, nx), dtype=arr.dtype)], axis=0
            )
        logger.debug("ensure_headroom: padded %d voxels (nz=%d)", pad, self.nz)

    def add_field(
        self, name: str, dtype: DTypeLike = np.float32, initial: float = 0.0
    ) -> NDArray:
        """Attach an extra per-voxel volume, shaped like ``mat``.

        ``mat`` holds a material ID and nothing else, which is enough for
        geometry and not enough for devices: a dopant concentration, a stress
        tensor component or a temperature all need their own volume over the
        same voxels. Rather than widen ``mat`` or bolt on a parallel array at
        the call site, they live here, and the four places that change a
        stack's shape or identity carry them automatically —
        :meth:`ensure_headroom`, :meth:`copy`, :meth:`save` and :meth:`load`.

        Nothing in the engine writes a field yet. It exists so that when
        implant and anneal arrive they do not force a change to ``Stack``'s
        API, its serialisation, or every renderer that reads ``mat``.

        Returns the array, so ``stack.add_field("dopant")[...] = ...`` works.
        """
        if name in self.fields:
            return self.fields[name]
        arr = np.full(self.mat.shape, initial, dtype=dtype)
        self.fields[name] = arr
        return arr

    def fill(self, where: NDArray[np.bool_], ref: str | int | Material) -> None:
        """Set every voxel in *where* to a material."""
        m = get_material(ref)
        self.mat[where] = m.id
        self.materials.setdefault(m.id, m)

    def clear(self, where: NDArray[np.bool_]) -> None:
        """Set every voxel in *where* to vacuum."""
        self.mat[where] = VACUUM

    def _on_ids(self, on: str | int | Material | Sequence | None) -> list | None:
        """Normalise a selective-growth filter to a list of material IDs."""
        if on is None:
            return None
        if isinstance(on, (str, int, np.integer, Material)):
            on = [on]
        return [get_material(o).id for o in on]

    def deposit_blanket(
        self,
        ref: str | int | Material,
        thickness: float,
        planarize: bool = False,
        on: str | int | Material | Sequence | None = None,
    ) -> None:
        """Deposit a film on top of the current surface.

        Parameters
        ----------
        ref : str or int or Material
            Material to deposit.
        thickness : float
            Deposited thickness [m].
        planarize : bool
            If True, fill up to a flat height ``thickness`` above the highest
            existing point (a spin-on / CMP-like fill).  If False, follow the
            existing topography — each column gets the same added thickness.
        on : material ref or sequence of refs, optional
            Grow only over columns whose **exposed** material is one of these.
            ``None`` (the default) grows everywhere, as before.  This is what
            makes epitaxy expressible: source/drain epi nucleates on exposed
            silicon and not on the oxide or nitride beside it, so a blanket
            deposit is the wrong primitive for it entirely.
        """
        m = get_material(ref)
        self.ensure_headroom(thickness * 2)
        n_add = self._n_voxels(thickness)
        surf = self.surface_index()  # (ny, nx), -1 where empty
        zz = np.arange(self.nz)[:, None, None]

        if planarize:
            target = int(surf.max(initial=-1)) + n_add
            where = (zz > surf[None]) & (zz <= target)
        else:
            where = (zz > surf[None]) & (zz <= surf[None] + n_add)

        seeds = self._on_ids(on)
        if seeds is not None:
            grows = np.isin(self.top_material(), seeds)
            if not grows.any():
                self.history.append(
                    f"deposit {m.name}: nothing it can nucleate on is exposed "
                    f"— no-op"
                )
                return
            where &= grows[None]

        self.fill(where & (self.mat == VACUUM), m)
        selective = "" if seeds is None else " on " + _names_of(seeds)
        self.history.append(
            f"deposit {m.name} {thickness*1e9:.0f} nm"
            f"{' (planarized)' if planarize else ''}{selective}"
        )

    def deposit_conformal(
        self,
        ref: str | int | Material,
        thickness: float,
        on: str | int | Material | Sequence | None = None,
    ) -> None:
        """Grow a conformal film of uniform thickness over every exposed surface.

        This is the step that makes self-aligned patterning work: the film
        grows *sideways* off vertical sidewalls as well as upward off
        horizontal ones, so after the mandrel is later removed the sidewall
        film is left standing on its own.

        Implemented as a Euclidean distance transform of the vacuum region
        with anisotropic sampling, so a "20 nm" spacer is 20 nm in every
        direction even when ``dz != pixel_size``.

        Parameters
        ----------
        on : material ref or sequence of refs, optional
            Grow only off surfaces of these materials — selective epitaxy,
            where the film nucleates on exposed silicon and nowhere else.
            ``None`` (the default) grows off everything, as before.

            The distance is straight-line, as it already was, so growth does
            not know about obstacles: a seed material behind a thin wall still
            grows through it.  Fine at the thicknesses this is used for, and
            the same approximation the unselective path has always made.
        """
        m = get_material(ref)
        self.ensure_headroom(thickness * 2)
        px = self.pixel_size
        seeds = self._on_ids(on)
        nucleate = (self.mat != VACUUM) if seeds is None else np.isin(self.mat, seeds)
        if not nucleate.any():
            self.history.append(
                f"deposit conformal {m.name}: nothing it can nucleate on is "
                f"present — no-op"
            )
            return
        dist = distance_transform_edt(~nucleate, sampling=(self.dz, px, px))
        # The transform's distances come out of a square root, so a shell at
        # an exact voxel multiple can land one ULP above the requested
        # thickness — and a "12 nm" film on a 4 nm grid quietly lost its
        # last shell and deposited 8 nm, while 16 nm (a power-of-two
        # multiple, exact in float) deposited 16. One part in a billion is
        # far below any physical tolerance and exactly covers the rounding.
        shell = (self.mat == VACUUM) & (dist <= thickness * (1.0 + 1e-9))
        self.fill(shell, m)
        selective = "" if seeds is None else " on " + _names_of(seeds)
        self.history.append(
            f"deposit conformal {m.name} {thickness*1e9:.0f} nm{selective}"
        )
        logger.debug(
            "deposit_conformal %s: %d voxels added", m.name, int(shell.sum())
        )

    def etch(
        self,
        targets: str | int | Material | Sequence,
        depth: float,
        anisotropy: float = 1.0,
        selectivity: dict[str, float] | None = None,
        open_mask: NDArray[np.bool_] | None = None,
        profile: EtchProfile | None = None,
        line_of_sight: bool = False,
        contact: bool = False,
        blanket: bool = False,
        stop_on: str | int | Material | None = None,
        exposure: str = "surface",
    ) -> None:
        """Etch downward from the exposed surface.

        The etch mask is **emergent**: a column etches only if the material
        currently on top of it is in *targets*.  Anything else — patterned
        resist, a hardmask, a spacer — masks the column simply by being
        there.  That is what lets LELE, SADP, and cut masks all fall out of
        this one primitive without special cases.

        Parameters
        ----------
        targets : material ref or sequence of refs
            Materials this chemistry attacks when exposed.
        depth : float
            Nominal etch depth [m] at unit etch rate.
        anisotropy : float
            ``1.0`` = perfectly directional.  Lower values add lateral
            undercut of ``(1 - anisotropy) * depth``.
        selectivity : dict, optional
            ``{material_name: relative_rate}`` overriding
            :attr:`Material.etch_rate`.  A rate near zero is an etch stop.
        open_mask : NDArray, optional
            Extra ``(ny, nx)`` boolean gate on which columns may etch.
            Rarely needed — prefer letting the stack define the mask.
        exposure : {"surface", "any"}
            Which surfaces the chemistry can attack.

            ``"surface"`` is the emergent-mask rule described above: a column
            opens only where the target is the material on **top**.  Right for
            every top-down patterning etch, and structurally unable to express
            anything else.

            ``"any"`` attacks the target wherever the etchant can reach it,
            **sidewalls included**.  That is what a channel release, an inner
            spacer recess or a source/drain undercut actually is: the layer is
            buried under a cap and exposed only on the wall of a trench, so
            ``"surface"`` correctly finds nothing to open and the process
            cannot be written down at all.  See :meth:`_etch_lateral`.
        """
        if exposure not in ("surface", "any"):
            raise ValueError(
                f"exposure must be 'surface' or 'any', not {exposure!r}"
            )
        if isinstance(targets, (str, int, np.integer, Material)):
            targets = [targets]
        target_ids = {get_material(t).id for t in targets}

        # Per-material time cost to remove one voxel: dz / rate.
        rates = np.zeros(256, dtype=np.float64)
        for mid, m in {**_BY_ID, **self.materials}.items():
            rate = m.etch_rate
            if selectivity and m.name in selectivity:
                rate = selectivity[m.name]
            rates[mid] = rate
        rates[VACUUM] = np.inf  # vacuum costs no time to pass through

        # A hard stop is a rate of nearly zero — the same mechanism selectivity
        # already uses, spelled the way a recipe thinks about it.
        if stop_on is not None:
            rates[get_material(stop_on).id] = 1e-6

        names = ", ".join(sorted(get_material(t).name for t in targets))

        if exposure == "any":
            # A different removal model, not a variation on the one below: the
            # column cost integral cannot express a front that arrives sideways.
            if profile is not None and not profile.is_identity:
                raise ValueError(
                    "an etch profile shapes one mask edge, and a lateral etch "
                    "has no single edge to shape — set exposure='surface' or "
                    "drop the profile"
                )
            if line_of_sight:
                raise ValueError(
                    "line_of_sight models ions travelling in straight lines; a "
                    "lateral etch is the chemistry that reaches where they "
                    "cannot — the two contradict each other"
                )
            self._etch_lateral(
                target_ids=target_ids, rates=rates, depth=depth,
                selectivity=selectivity, open_mask=open_mask,
                blanket=blanket, anisotropy=anisotropy, names=names,
            )
            return

        if blanket:
            # A blanket recess takes the surface down wherever there *is* a
            # surface, whatever material is exposed. That is the one thing the
            # emergent-mask rule cannot express: normally something on top
            # protects what is below, and here nothing does.
            open_col = self.surface_index() >= 0
        else:
            open_col = np.isin(self.top_material(), list(target_ids))
        exposed_cols = open_col.copy()
        if open_mask is not None:
            open_col &= open_mask
        if not open_col.any():
            self.history.append(self._no_target_note(
                target_ids, names, blanket=blanket,
                masked_out=bool(exposed_cols.any()),
            ))
            logger.debug("etch: nothing exposed for targets=%s", sorted(target_ids))
            return

        removed = self._removal_by_cost(rates, open_col, depth)
        if profile is not None and not profile.is_identity:
            removed = self._shape_with_profile(removed, profile, open_col, rates, depth)
        if line_of_sight:
            removed = self._shadow_line_of_sight(removed)
        if contact:
            removed = self._keep_surface_reachable(removed, open_col)
        if anisotropy < 1.0 and removed.any():
            removed = self._spread_undercut(removed, rates, (1.0 - anisotropy) * depth)

        self.clear(removed)
        self.history.append(f"etch {names} {depth*1e9:.0f} nm (aniso={anisotropy:.2f})")
        logger.debug(
            "etch %s: %d voxels removed from %d open columns",
            names, int(removed.sum()), int(open_col.sum()),
        )

    def _removal_by_cost(
        self,
        rates: NDArray[np.float64],
        open_col: NDArray[np.bool_],
        depth: float,
    ) -> NDArray[np.bool_]:
        """Directional removal: burn the depth budget down each open column.

        Time cost to remove one voxel of each material is ``dz / rate``. A slow
        material (low rate) burns more of the budget per voxel, which is what
        makes it an etch stop.
        """
        with np.errstate(divide="ignore"):
            cost = np.where(rates[self.mat] > 0, self.dz / rates[self.mat], np.inf)
        cost[self.mat == VACUUM] = 0.0

        # cum[iz] = cost of everything at or above iz; cum_above[iz] excludes
        # iz itself, so it is exactly the material the etch must chew through
        # before it can reach iz. Taken by shifting rather than as `cum - cost`,
        # which is inf - inf = NaN over a material with a rate of exactly zero —
        # a value the stop-rate control can now be set to. The comparisons below
        # happened to come out right on NaN; they say so on purpose instead.
        cum = np.cumsum(cost[::-1], axis=0)[::-1]
        cum_above = np.empty_like(cum)
        cum_above[:-1] = cum[1:]
        cum_above[-1] = 0.0

        surf = self.surface_index()
        zz = np.arange(self.nz)[:, None, None]
        # Restart the clock at each column's own surface, so a recessed column
        # does not get credit for the empty space above it.
        at_surface = np.take_along_axis(
            cum_above, np.clip(surf, 0, self.nz - 1)[None], axis=0
        )
        elapsed = cum_above - at_surface

        budget = depth  # metres of a unit-rate material
        return (
            open_col[None]
            & (zz <= surf[None])
            & (elapsed < budget)
            & (self.mat != VACUUM)
            & np.isfinite(cost)   # a rate of zero is a wall, not a slow film
        )

    def _shape_with_profile(
        self,
        removed: NDArray[np.bool_],
        profile: EtchProfile,
        open_col: NDArray[np.bool_],
        rates: NDArray[np.float64],
        depth: float,
    ) -> NDArray[np.bool_]:
        """Re-carve the removal to the profile's sidewall shape.

        One signed distance field, thresholded per plane. Negative inside
        the opening, positive outside, so ``signed <= offset(d)`` widens the
        etch where the offset is positive and pulls it in where negative —
        bias, taper, both corner radii and the foot all fall out of the
        same comparison, on a field computed once.
        """
        px = self.pixel_size
        if open_col.all() or not open_col.any():
            # A profile shapes a *sidewall*, and an etch with no mask edge
            # in the field has none to shape. Skipping is not a nicety:
            # `distance_transform_edt` of an all-True array measures the
            # distance to the array edge, not to a masked region, so the
            # profile would carve a bowl out of the field boundary — and
            # with line_of_sight on, shadow everything beneath it.
            # Say so where a user can see it. A shape control that silently
            # does nothing is how a working engine comes to look broken —
            # and `blanket=True` reaches here too, because it opens every
            # column by construction.
            self.history.append(
                "etch: the whole field is open, so there is no mask edge — "
                "bias, wall angle, corner radii and the foot were ignored"
            )
            logger.debug("etch: no mask edge in the field, profile ignored")
            return removed

        # Replicate the edge before the transform so the field boundary
        # is not mistaken for a mask edge either. A feature running off
        # the edge continues outward instead of acquiring a wall there.
        pad = int(np.ceil(
            max(abs(float(np.max(np.abs(profile.offsets(
                np.array([0.0, depth]), depth))))), px) / px
        )) + 2
        padded = np.pad(open_col, pad, mode="edge")
        signed = (
            distance_transform_edt(~padded, sampling=(px, px))
            - distance_transform_edt(padded, sampling=(px, px))
        )[pad:-pad, pad:-pad]

        # How deep the etch actually got, from the cost model — so
        # selectivity still sets the floor that the bottom features are
        # measured up from.
        reached = (
            float(removed.sum(axis=0).max()) * self.dz if removed.any() else 0.0
        )
        if reached <= 0.0:
            reached = depth

        surf = self.surface_index()
        zz = np.arange(self.nz)[:, None, None]
        # Depth below each column's *own* surface, so a recessed column's
        # profile starts where its material does rather than at the highest
        # point on the wafer.
        depth_below = (surf[None] - zz) * self.dz
        offset = profile.offsets(depth_below, reached)

        shaped = (
            (signed[None] <= offset)
            & (depth_below >= 0.0)
            & (depth_below <= reached)
            & (self.mat != VACUUM)
            # An etch cannot remove anything higher than the surface it
            # started from. `depth_below` is measured from each column's
            # own top, which is right for a recessed *open* column and
            # quite wrong for a masked one: there the reference is the top
            # of the mask, so any outward offset — a re-entrant wall above
            # all — carved the mask into the shape of the hole it was
            # supposed to be defining. Anything standing proud of the open
            # surface is, by definition, masking.
            & (zz <= int(surf[open_col].max()))
        )

        # The profile is a *shape*, not a transport model: it does not
        # integrate a cost along the lateral path, so a stop layer has to be
        # protected explicitly. Otherwise a widened front eats sideways into
        # it — exactly the mistake the isotropic pass used to make.
        slow = rates[self.mat] <= _STOP_RATE
        if slow.any():
            below_stop = np.cumsum(slow[::-1], axis=0)[::-1] > 0
            shaped &= ~below_stop
        return shaped

    def _shadow_line_of_sight(
        self, removed: NDArray[np.bool_]
    ) -> NDArray[np.bool_]:
        """Keep only voxels an ion travelling straight down can reach.

        Ion flux travels in straight lines and cannot turn a corner, so a
        voxel is reachable only if nothing *solid and surviving* stands
        above it in its own column. Everything past the first obstruction
        is shadowed.

        Vacuum has to count as transparent, which is the whole subtlety:
        measuring from the top of the array instead of from the surface
        made the empty headroom an obstruction, and shadowed the entire
        wafer.
        """
        blocked = (self.mat != VACUUM) & ~removed
        above = np.cumsum(blocked[::-1], axis=0)[::-1] - blocked
        return removed & (above == 0)

    def _keep_surface_reachable(
        self,
        removed: NDArray[np.bool_],
        open_col: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        """Keep only removal the etchant can physically get to.

        Chemical access: keep material connected to the open surface through
        other removed material — which permits reaching *under* an overhang,
        and forbids opening a sealed void that nothing can reach.
        """
        from scipy.ndimage import label as _label

        lab, n = _label(removed)
        if n == 0:
            return removed
        surf = self.surface_index()
        yy, xx = np.indices(surf.shape)
        at_surface = lab[np.clip(surf, 0, self.nz - 1), yy, xx]
        reachable = np.unique(np.concatenate([
            at_surface[open_col].ravel(), lab[-1].ravel(),
        ]))
        reachable = reachable[reachable != 0]
        return np.isin(lab, reachable)

    def _spread_undercut(
        self,
        removed: NDArray[np.bool_],
        rates: NDArray[np.float64],
        lateral: float,
    ) -> NDArray[np.bool_]:
        """Dilate the removal sideways for a partially isotropic etch.

        An isotropic etch spreads sideways at the *local* rate, so a slow
        material has to resist lateral attack exactly as it resists the
        vertical kind. Scaling each voxel's reach by its own rate is what
        makes selectivity mean something here — without it the dilation
        removed anything within reach regardless of material, and walked
        straight through an etch stop the directional pass had correctly
        stopped on. Measured before the fix: at anisotropy 0.7 a 60 nm
        stop layer with selectivity 0.001 was removed *entirely*.

        Deliberately *not* gated on the open columns: undercut is by
        definition material removed from beneath a mask, so restricting the
        spread to open columns would delete the very effect anisotropy < 1
        exists to produce.
        """
        px = self.pixel_size
        dist = distance_transform_edt(~removed, sampling=(self.dz, px, px))
        reach = lateral * rates[self.mat]
        return removed | ((dist <= reach) & (self.mat != VACUUM))

    def _no_target_note(
        self,
        target_ids: Iterable[int],
        names: str,
        blanket: bool = False,
        masked_out: bool = False,
    ) -> str:
        """Why an etch found nothing to open, specifically enough to act on.

        One sentence used to cover four different situations, so a recipe in
        the wrong order, a buried layer, a closed mask and a missing engine
        capability all read identically from the outside — and the fixes are
        nothing alike.  The reported case was an etch placed *above* the step
        that deposits its target: the engine was right, said so, and the user
        still had no way to tell which of the four they were looking at.
        """
        from scipy.ndimage import binary_dilation

        if masked_out:
            return (f"etch: {names} is exposed, but the open mask closed every "
                    f"column it was exposed in — no-op")
        if blanket:
            return "etch: the wafer has no surface left to recess — no-op"

        ids = list(target_ids)
        is_target = np.isin(self.mat, ids)
        if not is_target.any():
            here = ", ".join(m.name for m in self.present_materials()) or "nothing"
            return (f"etch: no {names} anywhere on this wafer — it holds "
                    f"{here}. Check the step order, or aim at what is "
                    f"here — no-op")

        # Face contact only: a diagonal touch is not an opening.
        cross = np.zeros((3, 3, 3), dtype=bool)
        cross[1, 1, :] = cross[1, :, 1] = cross[:, 1, 1] = True
        if (is_target & binary_dilation(self.mat == VACUUM, structure=cross)).any():
            return (f"etch: {names} is on the wafer and open to the etchant, but "
                    f"only on a sidewall — a top-down etch cannot turn a corner. "
                    f'exposure="any" attacks every surface the etchant can '
                    f"reach — no-op")

        # Nothing of it touches open space, so something is sitting on it.
        # Name the material directly above the topmost target voxel.
        top_of = np.where(
            is_target.any(axis=0),
            self.nz - 1 - np.argmax(is_target[::-1], axis=0),
            -1,
        )
        yy, xx = np.indices(top_of.shape)
        ok = (top_of >= 0) & (top_of + 1 < self.nz)
        above = self.mat[np.clip(top_of + 1, 0, self.nz - 1), yy, xx][ok]
        above = above[above != VACUUM]
        if above.size:
            found, counts = np.unique(above, return_counts=True)
            burier = get_material(int(found[counts.argmax()])).name
            return f"etch: {names} is buried under {burier} — no-op"
        return f"etch: {names} is not exposed anywhere — no-op"

    def _etch_lateral(
        self,
        target_ids: Iterable[int],
        rates: NDArray[np.float64],
        depth: float,
        selectivity: dict[str, float] | None,
        open_mask: NDArray[np.bool_] | None,
        blanket: bool,
        anisotropy: float,
        names: str,
    ) -> None:
        """Etch every surface the etchant reaches, sidewalls included.

        The column model above asks "what is on top of this column, and how
        long does the front take to chew down through it".  A release etch is
        not that shape at all: the sacrificial layer is capped, and the
        etchant arrives horizontally through the wall of a trench.  No amount
        of selectivity makes a top-down integral turn that corner.

        The right statement is the same one development already makes — the
        front moves normal to itself at the local rate — so this reuses
        :func:`~litho_sim.develop.front.arrival_time`, the fast-sweeping
        eikonal solver written for the developer.  Seed it from every open
        surface instead of just the top plane, give each voxel a speed equal
        to its etch rate, and the removed set is the level set ``{T <= depth}``.

        Selectivity stops being a special case and becomes what it physically
        is: a rate contrast.  Loading and the slower arrival at the middle of
        a long undercut fall out of the same solve, and one solve gives every
        etch time at once.

        Under this model *targets* is not a column-opening rule — it is the
        medium the front travels through.  Anything not named, and not given
        an explicit rate by *selectivity*, is a wall.
        """
        # Deferred: `litho_sim.develop` imports `litho_sim.wafer`, so importing
        # the solver at module scope would close the cycle. Same reason as the
        # scipy import in the contact branch above.
        from scipy.ndimage import label as _label

        from litho_sim.develop.front import arrival_time

        # What the chemistry consumes, and how fast.
        attack = np.zeros_like(rates)
        if blanket:
            # A lateral blanket etch is a wet dip: it attacks whatever it
            # touches, each material at its own rate.
            attack[:] = rates
        else:
            ids = list(target_ids)
            attack[ids] = rates[ids]
            # An explicit selectivity entry is the user naming a rate for a
            # material the front may creep into — a liner meant to slow it, not
            # only the target it is meant to consume.
            for name in (selectivity or {}):
                mid = get_material(name).id
                attack[mid] = rates[mid]
        # Open space costs no time to cross. Finite rather than `inf` so the
        # solver's slowness stays a real number.
        attack[VACUUM] = 1e6

        # The etchant comes from outside. Vacuum sealed inside the wafer is not
        # a source — otherwise a void left by a pinched-off conformal film would
        # quietly etch itself open from the inside.
        lab, n = _label(self.mat == VACUUM)
        seed = np.zeros(self.mat.shape, dtype=bool)
        if n > 0:
            outside = np.unique(lab[-1])
            outside = outside[outside != 0]
            seed = np.isin(lab, outside)

        reachable = (attack[self.mat] > 0.0) & (self.mat != VACUUM)
        if not (seed.any() and reachable.any()):
            self.history.append(self._no_target_note(target_ids, names))
            return

        # Nanometres, not metres. `arrival_time`'s convergence tolerance is
        # absolute, and against a 6e-8 budget the default 1e-6 would declare
        # victory before the first sweep did anything.
        nm = 1e9
        spacing = (self.dz * nm, self.pixel_size * nm, self.pixel_size * nm)
        budget = depth * nm

        # Solve on the attackable material and a thin rim, not the whole array.
        #
        # The tempting pad is "however far the front could travel", and it is
        # far larger than necessary. Every open voxel is already a seed at
        # T = 0, so the front never *travels* through vacuum — it only
        # propagates through attackable material, and any seed touching that
        # material is one voxel outside its bounding box. Two voxels of rim is
        # therefore exact, not an approximation: verified bit-identical to the
        # full-array solve, and 10x faster on a full-field release
        # (128x128x140: 6840 ms whole array, 4192 ms padded by the reach,
        # 663 ms like this).
        pad = 2
        box = []
        for axis in range(3):
            hit = np.where(reachable.any(
                axis=tuple(a for a in range(3) if a != axis)))[0]
            lo = max(0, int(hit[0]) - pad)
            hi = min(self.mat.shape[axis], int(hit[-1]) + 1 + pad)
            box.append(slice(lo, hi))
        box = tuple(box)

        arrival = arrival_time(attack[self.mat[box]], spacing=spacing,
                               seed=seed[box])
        removed = np.zeros(self.mat.shape, dtype=bool)
        removed[box] = (arrival <= budget) & (self.mat[box] != VACUUM)
        if open_mask is not None:
            removed &= open_mask[None]

        if not removed.any():
            self.history.append(self._no_target_note(target_ids, names))
            return

        self.clear(removed)
        note = f"etch {names} {depth*1e9:.0f} nm (lateral front)"
        if anisotropy < 1.0:
            # Not ignored — subsumed. Say so rather than let a control the user
            # moved look as though it did nothing.
            note += "; already isotropic, so anisotropy added nothing"
        self.history.append(note)
        logger.debug(
            "etch %s lateral: %d voxels removed, solved on %s",
            names, int(removed.sum()), tuple(s.stop - s.start for s in box),
        )

    def etch_back(
        self,
        ref: str | int | Material,
        thickness: float | None = None,
        overetch: float = 0.2,
    ) -> None:
        """Spacer etch-back: clear the horizontal film, leave sidewalls standing.

        A purely directional etch of one film thickness plus a little
        overetch.  Film lying on a horizontal surface is exactly one
        thickness deep and clears; film on a vertical sidewall presents the
        full sidewall height to a downward etch and survives.  Pulling the
        mandrel afterwards leaves those sidewalls freestanding — which is the
        whole trick behind self-aligned pitch division.

        Parameters
        ----------
        ref : material ref
            The spacer material.
        thickness : float, optional
            Deposited film thickness [m].  Defaults to the thinnest place the
            film covers, which is the flat-field thickness.  Do **not** use
            the maximum column thickness — over a mandrel sidewall that is
            the mandrel height, and etching that deep clears the wafer.
        overetch : float
            Extra etch beyond one thickness, as a fraction.
        """
        m = get_material(ref)
        t_col = self.thickness_of(m)
        if not (t_col > 0).any():
            self.history.append(f"etch_back {m.name}: not present — no-op")
            return
        if thickness is None:
            thickness = float(t_col[t_col > 0].min())
        # The etch chemistry is chosen for this film, so give it unit rate
        # rather than the material's generic (selectivity-oriented) rate.
        self.etch(
            m,
            depth=thickness * (1.0 + overetch),
            anisotropy=1.0,
            selectivity={m.name: 1.0},
        )
        self.history[-1] = (
            f"etch_back {m.name} {thickness*1e9:.0f} nm (overetch={overetch:.0%})"
        )

    def strip(self, ref: str | int | Material) -> None:
        """Remove a material everywhere — a wet strip or ash."""
        m = get_material(ref)
        n = int((self.mat == m.id).sum())
        self.clear(self.mat == m.id)
        self.history.append(f"strip {m.name}")
        logger.debug("strip %s: %d voxels removed", m.name, n)

    def planarize(
        self,
        height: float | None = None,
        depth: float | None = None,
        stop_on: str | int | Material | None = None,
    ) -> None:
        """CMP: take the surface down to a flat plane.

        Three ways to say where the plane goes, and exactly one of them at a
        time. They exist because a real recipe is written in whichever one the
        process engineer can actually control:

        ``height``
            An absolute level above the substrate. What you use when the target
            is a known film thickness.
        ``depth``
            Remove this much, measured **down from the current high point**.
            What you use when the input topography is whatever the last step
            left and you only know how much to take off.
        ``stop_on``
            Polish until the pad reaches this material and stop there. The
            plane lands at the *highest* point that material reaches anywhere
            on the wafer, because that is where the pad touches it first — so a
            polish stop leaves material behind wherever the stop layer is low.

        With none of the three it takes the surface down to the lowest peak,
        which is the least it can remove and still come out flat.

        A guillotine, deliberately: no dishing, no erosion, indifferent to what
        it cuts through. The first thing to revisit if damascene ever needs a
        realistic overpolish.

        Raises
        ------
        ValueError
            If more than one of the three is given, since there would be no
            sensible way to reconcile them.
        """
        given = [n for n, v in
                 (("height", height), ("depth", depth), ("stop_on", stop_on))
                 if v is not None]
        if len(given) > 1:
            raise ValueError(
                f"planarize takes at most one of height/depth/stop_on, got {given}"
            )

        if depth is not None:
            top = float(self.top_height().max())
            height = max(top - float(depth), 0.0)
            note = f"CMP {depth*1e9:.0f} nm down from {top*1e9:.0f} nm"
        elif stop_on is not None:
            m = get_material(stop_on)
            present = self.mat == m.id
            if not present.any():
                raise ValueError(
                    f"CMP cannot stop on '{m.name}': the stack contains none of it."
                )
            # The pad is flat and comes down, so it meets the stop layer at that
            # layer's highest point anywhere — not at its height in some
            # particular column.
            top_index = int(np.nonzero(present.any(axis=(1, 2)))[0].max())
            height = (top_index + 1) * self.dz
            note = f"CMP stopping on {m.name} at {height*1e9:.0f} nm"
        else:
            if height is None:
                height = float(self.top_height().min())
            note = f"CMP to {height*1e9:.0f} nm"

        keep = self._n_voxels(height)
        above = np.arange(self.nz)[:, None, None] >= keep
        self.clear(np.broadcast_to(above, self.mat.shape))
        self.history.append(note)
        logger.debug("%s (kept %d voxels)", note, keep)

    # ------------------------------------------------------------------
    # Metrology
    # ------------------------------------------------------------------

    def column_runs(
        self, ref: str | int | Material, x: int, y: int
    ) -> list[tuple[int, int]]:
        """Contiguous z-runs of one material in the column at (*x*, *y*).

        Returns ``(z_lo, z_hi)`` index pairs, bottom-up, half-open like a
        slice: ``mat[z_lo:z_hi, y, x]`` is the run. Two runs of the same
        material at different heights stay two runs — a liner under a fill,
        or stacked nanosheets — which is the whole point of asking.
        """
        col = self.mat[:, y, x] == get_material(ref).id
        runs: list[tuple[int, int]] = []
        iz = 0
        while iz < self.nz:
            if col[iz]:
                z0 = iz
                while iz < self.nz and col[iz]:
                    iz += 1
                runs.append((z0, iz))
            else:
                iz += 1
        return runs

    def line_profile(
        self, ref: str | int | Material | None = None, z_frac: float = 0.5
    ) -> NDArray[np.bool_]:
        """Horizontal presence profile through the centre row at a fractional height.

        Parameters
        ----------
        ref : material ref, optional
            Material to profile.  Defaults to "anything solid".
        z_frac : float
            Height at which to cut, as a fraction of the feature's own top
            height.  0.5 samples mid-height, the usual place to measure CD.

        Returns
        -------
        NDArray[np.bool_]
            Shape ``(nx,)``, True where the material is present.
        """
        occ = self.solid() if ref is None else (self.mat == get_material(ref).id)
        present_z = np.nonzero(occ.any(axis=(1, 2)))[0]
        if present_z.size == 0:
            return np.zeros(self.shape_xy[1], dtype=bool)
        iz = int(present_z.min() + z_frac * (present_z.max() - present_z.min()))
        ny = self.shape_xy[0]
        return occ[iz, ny // 2, :]

    def measure_features(
        self, ref: str | int | Material | None = None, z_frac: float = 0.5
    ) -> dict[str, list]:
        """Measure line widths and space widths along the centre row.

        Returns
        -------
        dict
            ``{"lines": [...], "spaces": [...]}`` in metres, in x order.
            Runs touching either edge are dropped, since they are truncated
            by the simulation window rather than by the pattern.

        Notes
        -----
        Alternating space widths are the fingerprint of overlay-induced
        pitch walking, so *spaces* is usually the interesting list.
        """
        prof = self.line_profile(ref, z_frac)
        runs, out = [], {"lines": [], "spaces": []}
        if prof.size == 0:
            return out

        start, cur = 0, bool(prof[0])
        for i in range(1, prof.size):
            if bool(prof[i]) != cur:
                runs.append((cur, start, i - 1))
                start, cur = i, bool(prof[i])
        runs.append((cur, start, prof.size - 1))

        for is_line, a, b in runs:
            if a == 0 or b == prof.size - 1:
                continue  # truncated by the window edge
            width = (b - a + 1) * self.pixel_size
            out["lines" if is_line else "spaces"].append(width)
        return out

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a compressed ``.npz``.  Layered uint8 fields compress well."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "dz": self.dz,
            "n_pixels": self.grid.n_pixels,
            "pixel_size": self.grid.pixel_size,
            "history": self.history,
            "materials": {
                str(k): {
                    "id": v.id, "name": v.name, "color": v.color,
                    "opacity": v.opacity, "etch_rate": v.etch_rate, "role": v.role,
                    # complex is not JSON — store as a [re, im] pair
                    "n_index": [v.n_index.real, v.n_index.imag],
                    "se_yield": v.se_yield,
                }
                for k, v in self.materials.items()
            },
        }
        meta["fields"] = {k: str(v.dtype) for k, v in self.fields.items()}
        extra = {f"field_{k}": v for k, v in self.fields.items()}
        np.savez_compressed(path, mat=self.mat, meta=json.dumps(meta), **extra)
        logger.info("Stack saved → %s (%.1f KB)", path, path.stat().st_size / 1e3)

    @classmethod
    def load(cls, path: str | Path) -> Stack:
        """Load a stack written by :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Stack file not found: {path}")
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        materials = {
            int(k): Material(
                id=v["id"], name=v["name"], color=v["color"],
                opacity=v["opacity"], etch_rate=v["etch_rate"], role=v["role"],
                # files written before n_index was persisted fall back to the
                # dataclass default rather than failing to load
                n_index=complex(*v["n_index"]) if "n_index" in v else 1.5 + 0j,
                # Files written before the SE yield existed — every cached
                # device preset — take the library's value for the same ID,
                # so a loaded GAA images with the same contrast as a fresh
                # one rather than as one flat grey.
                se_yield=v.get("se_yield", _BY_ID.get(int(v["id"]), Material(0, "")).se_yield),
            )
            for k, v in meta["materials"].items()
        }
        # Written before auxiliary fields existed? Then there are none.
        fields = {
            name: data[f"field_{name}"] for name in meta.get("fields", {})
        }
        return cls(
            mat=data["mat"],
            grid=GridConfig(n_pixels=meta["n_pixels"], pixel_size=meta["pixel_size"]),
            dz=meta["dz"],
            materials=materials,
            history=list(meta["history"]),
            fields=fields,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        names = ", ".join(m.name for m in self.present_materials()) or "empty"
        return (
            f"Stack({self.nz}x{self.shape_xy[0]}x{self.shape_xy[1]} voxels, "
            f"dz={self.dz*1e9:.1f} nm, px={self.pixel_size*1e9:.1f} nm, [{names}])"
        )

    def ascii_section(self, axis: str = "y", index: int | None = None) -> str:
        """Render a vertical cross-section as text — handy in tests and logs."""
        sec = self.cross_section(axis, index)
        glyphs = {VACUUM: "."}
        for i, m in enumerate(self.present_materials()):
            glyphs[m.id] = "#@%*+=~oxX"[i % 10]
        lines = [
            "".join(glyphs.get(int(v), "?") for v in row)
            for row in sec[::-1]  # print top-down
        ]
        return "\n".join(lines)
