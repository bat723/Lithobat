"""
The print model OPC corrects against, and the edge placement error it measures.

An OPC algorithm is only as good as its model of what prints, and the
model here is *the engine itself*: the same imaging, the same resist
chemistry and the same "field crosses a level" definition of an edge that
:mod:`litho_sim.analysis.process_window` measures Bossung curves with. There
is no second, approximate imaging model inside OPC. What there is, for the
scalar thin-mask process, is the same optics *reassembled*: the
sum-of-coherent-systems kernels of :mod:`litho_sim.expose.hopkins`, built
once per model and exact to a part in 1e-8 of the Abbe sum, which is what
makes an iteration cost a few transforms rather than a few hundred.

Edge placement error
--------------------
The one number OPC minimises. For each fragment's control point, walk
along the fragment's outward normal through the continuous latent field
and find where the printed edge crosses it. The signed distance from the
drawn edge to that crossing is the **EPE**: positive when the printed
feature reaches *beyond* the drawn edge (too big), negative when it stops
short (too small, or pulled back at a line end). It is measured on the
continuous field with sub-pixel interpolation, for the same reason the
process-window CD is — binarising first quantises every error to the grid.

Two things can go wrong at a control point, and both are results rather
than errors: the printed feature can be absent there (``"missing"`` — a
pinched line, a lost contact) or it can extend past the search range
(``"merged"`` — bridged to its neighbour). The loop treats each as a
maximal move in the recovering direction.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from litho_sim.core.config import GridConfig, OpticsConfig, ResistConfig
from litho_sim.develop.resist import develop_field
from litho_sim.expose.aerial_image import compute_aerial_image, normalisation_scale
from litho_sim.expose.hopkins import SOCSKernels
from litho_sim.mask.layout import Layout
from litho_sim.opc.fragments import FragmentedShape

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# What printed
# ---------------------------------------------------------------------------


@dataclass
class PrintedImage:
    """One layout, printed: the mask, the aerial image, and the latent field.

    Attributes
    ----------
    mask : NDArray
        The rasterised mask that was imaged, ``(n, n)`` transmittance.
    aerial : NDArray
        Aerial image at the model's dose.
    field, level : NDArray, float
        The continuous field and the level whose crossing is the resist
        edge — from :func:`litho_sim.develop.resist.develop_field`.
    tone : str
        The mask polarity the layout was printed with. Decides which side
        of the level is the *drawn feature's* image: the bright side for
        ``"clear"``, the dark side for ``"dark"``.
    grid : GridConfig
        The grid everything above is on.
    """

    mask: NDArray[np.float64]
    aerial: NDArray[np.float64]
    field: NDArray[np.float64]
    level: float
    tone: str
    grid: GridConfig

    @property
    def signed(self) -> NDArray[np.float64]:
        """``field − level``, signed so that positive means *inside* the printed feature."""
        s = self.field - self.level
        return s if self.tone == "clear" else -s

    @property
    def printed(self) -> NDArray[np.bool_]:
        """Where the drawn features' image landed, as a boolean map."""
        return self.signed > 0.0

    def contours(self) -> list[NDArray[np.float64]]:
        """The printed feature outlines as ``(N, 2)`` polylines in metres.

        Extracted from the continuous field, so they sit between pixels.
        """
        from contourpy import contour_generator

        n, px = self.grid.n_pixels, self.grid.pixel_size
        axis = (np.arange(n) - n // 2) * px
        gen = contour_generator(axis, axis, self.signed)
        return [np.asarray(line, dtype=np.float64) for line in gen.lines(0.0)]

    def sample(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """The signed field at arbitrary ``(K, 2)`` positions [m], bilinear.

        Periodic, because the imaging is — the FFT wraps the field, so a
        point past the right edge reads the left edge.
        """
        n, px = self.grid.n_pixels, self.grid.pixel_size
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        cols = pts[:, 0] / px + n // 2
        rows = pts[:, 1] / px + n // 2
        return map_coordinates(self.signed, [rows, cols], order=1, mode="grid-wrap")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class PrintModel:
    """Everything needed to turn a layout into a printed edge.

    Parameters
    ----------
    optics, resist, grid
        The process. Whatever ``optics.mask_model`` and ``imaging_model``
        say is what the corrections are computed against.
    dose : float
        Relative dose the layout is printed at — normally the dose-to-size
        of an anchor feature; see :meth:`dose_to_size`.
    tone : str
        Mask polarity, ``"clear"`` or ``"dark"``, as
        :meth:`~litho_sim.mask.layout.Layout.rasterize` takes it.
    model : str
        Resist model — ``"threshold"``, ``"mack"`` or ``"car"``.
    normalisation : str, optional
        Overrides ``optics.normalisation``; defaults to ``"clear"``. Under
        peak normalisation every correction would move the peak and with it
        the meaning of dose, so a bias could never converge. ``None`` keeps
        the optics as given.
    oversample : int
        Rasterisation anti-aliasing factor. 4 resolves a quarter-pixel
        edge move, which is what lets sub-nanometre corrections act on a
        4 nm grid at all.
    imaging : str
        ``"auto"`` images through the SOCS kernels whenever the optics allow
        it (scalar model, thin mask) and through the Abbe sum otherwise.
        The kernels are never more numerous than the source points and each
        costs less than a source point does — on a 128 px field 61 kernels
        replace 197 points at a quarter of the time, and on a 256 px field
        where the TCC's rank reaches the full 197 the print still halves.
        ``"socs"`` insists on the kernels and refuses optics that cannot
        have them; ``"abbe"`` always runs the source loop.
    socs_tol : float
        Smallest kernel kept, relative to the largest — see
        :meth:`litho_sim.expose.hopkins.SOCSKernels.build`.
    """

    optics: OpticsConfig
    resist: ResistConfig
    grid: GridConfig
    dose: float = 1.0
    tone: str = "clear"
    model: str = "threshold"
    normalisation: str | None = "clear"
    oversample: int = 4
    imaging: str = "auto"
    socs_tol: float = 1e-8
    _raw_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tone not in ("clear", "dark"):
            raise ValueError(f"tone must be 'clear' or 'dark', got '{self.tone}'")
        if self.imaging not in ("auto", "socs", "abbe"):
            raise ValueError(f"imaging must be 'auto', 'socs' or 'abbe', got '{self.imaging}'")
        if self.normalisation is not None and self.normalisation != self.optics.normalisation:
            self.optics = dataclasses.replace(self.optics, normalisation=self.normalisation)
        if self.imaging == "socs" and not SOCSKernels.supports(self.optics):
            raise ValueError(
                "imaging='socs' needs the scalar thin-mask process; got "
                f"imaging_model={self.optics.imaging_model!r}, "
                f"mask_model={self.optics.mask_model!r}"
            )

    # ------------------------------------------------------------------

    @property
    def uses_kernels(self) -> bool:
        """Whether prints go through the SOCS kernels rather than the Abbe loop."""
        return self.kernels() is not None

    def kernels(self) -> SOCSKernels | None:
        """The kernels for the current optics, built once and kept.

        Keyed on the optics and grid, so replacing either on the model
        rebuilds them on the next print; ``None`` when the model images
        through the Abbe sum — because it was told to, or because the optics
        cannot be decomposed.
        """
        if self.imaging == "abbe":
            return None
        if self.imaging == "auto" and not SOCSKernels.supports(self.optics):
            return None
        key = (
            json.dumps(dataclasses.asdict(self.optics), sort_keys=True, default=str),
            self.grid.n_pixels, self.grid.pixel_size, self.socs_tol,
        )
        cached = self._raw_cache.get("kernels")
        if cached is None or cached[0] != key:
            cached = (key, SOCSKernels.build(self.optics, self.grid, tol=self.socs_tol))
            self._raw_cache["kernels"] = cached
        return cached[1]

    # ------------------------------------------------------------------

    def rasterize(self, layout: Layout, layer: str | None = None) -> NDArray[np.float64]:
        return layout.rasterize(self.grid, layer=layer, tone=self.tone, oversample=self.oversample)

    def print(
        self, layout: Layout, layer: str | None = None, dose: float | None = None
    ) -> PrintedImage:
        """Image a layout and develop it to the latent field.

        One image — through the kernels or the Abbe sum, see :attr:`uses_kernels`.
        *dose* overrides the model's dose for this print only.
        """
        mask = self.rasterize(layout, layer)
        kernels = self.kernels()
        if kernels is not None:
            raw, peak, clear = kernels.raw(mask)
        else:
            raw, peak, clear = compute_aerial_image(mask, self.optics, self.grid, return_raw=True)
        d = self.dose if dose is None else float(dose)
        aerial = raw * normalisation_scale(self.optics.normalisation, peak, clear, d)
        fld, level = develop_field(aerial, self.resist, self.grid, model=self.model)
        return PrintedImage(mask, aerial, fld, level, self.tone, self.grid)

    def dose_to_size(
        self,
        layout: Layout,
        target_cd_nm: float,
        axis: str = "x",
        layer: str | None = None,
        **kwargs,
    ) -> float:
        """The dose at which the drawn feature on the centre cut prints *target_cd_nm* wide.

        Sizes the *drawn feature* — the bright region of a clear-tone mask,
        the dark region of a dark-tone one — measured along *axis* through
        the field centre. Wraps
        :func:`litho_sim.analysis.calibrate_dose_to_size`; extra keyword
        arguments go through to it.

        Returns the dose without storing it, so a caller can decide whether
        to adopt it (``model.dose = model.dose_to_size(...)``).
        """
        from litho_sim.analysis.process_window import calibrate_dose_to_size

        feature = "above" if self.tone == "clear" else "below"
        return calibrate_dose_to_size(
            self.rasterize(layout, layer), self.optics, self.grid, self.resist,
            target_cd_nm, model=self.model, normalisation=self.normalisation,
            feature=feature, axis=1 if axis == "x" else 0, **kwargs,
        )


# ---------------------------------------------------------------------------
# Edge placement error
# ---------------------------------------------------------------------------


@dataclass
class EPE:
    """Edge placement error at every fragment of a fragmented layout.

    Attributes
    ----------
    values : (M,) array
        EPE per fragment [m], positive when the printed edge lies outside
        the drawn one. NaN where the edge could not be found.
    status : list of str
        ``"ok"``, ``"missing"`` (no printed feature within reach on the
        inside of the drawn edge), ``"merged"`` (the printed feature runs
        past the search range on the outside) or ``"fixed"`` (a fragment
        held outside the imaged field and never measured — NaN here, and
        neither an error nor a failure).
    inside : (M,) array
        The signed field *at* each control point — the raw quantity the
        EPE is a root of; useful for diagnosing a fragment.
    """

    values: NDArray[np.float64]
    status: list[str]
    inside: NDArray[np.float64]

    @property
    def ok(self) -> NDArray[np.bool_]:
        return np.array([s == "ok" for s in self.status], dtype=bool)

    @property
    def fixed(self) -> NDArray[np.bool_]:
        return np.array([s == "fixed" for s in self.status], dtype=bool)

    @property
    def n_failed(self) -> int:
        """Sites where the printed edge was not found — missing or merged."""
        return int(sum(s in ("missing", "merged") for s in self.status))

    @property
    def n_fixed(self) -> int:
        return int(self.fixed.sum())

    @property
    def max_abs(self) -> float:
        """Largest |EPE| [m] over the fragments that have one; NaN if none."""
        v = self.values[self.ok]
        return float(np.abs(v).max()) if v.size else float("nan")

    @property
    def rms(self) -> float:
        v = self.values[self.ok]
        return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")

    def stats(self) -> dict[str, float]:
        """The headline numbers, in nanometres."""
        v = self.values[self.ok]
        return {
            "max_abs_epe_nm": self.max_abs * 1e9,
            "rms_epe_nm": self.rms * 1e9,
            "mean_epe_nm": float(v.mean()) * 1e9 if v.size else float("nan"),
            "n_fragments": int(len(self.values)),
            "n_failed": self.n_failed,
            "n_fixed": self.n_fixed,
        }


def measure_epe(
    printed: PrintedImage,
    fragmented: Sequence[FragmentedShape],
    search: float,
    step: float | None = None,
) -> EPE:
    """Measure the edge placement error at every fragment's control point.

    Parameters
    ----------
    printed : PrintedImage
        What the layout printed as.
    fragmented : sequence of FragmentedShape
        The fragments of the *drawn* layout — the targets, not the
        corrected geometry. The sites and the directions to measure along
        come from here; a retargeted corner fragment measures against the
        rounded corner, along the arc's normal.
    search : float
        How far along the normal to look for the printed edge, either side
        of the drawn edge [m].
    step : float, optional
        Sampling step along the normal [m]; defaults to a quarter pixel.

    Returns
    -------
    EPE
    """
    px = printed.grid.pixel_size
    h = step or 0.25 * px
    k = max(int(np.ceil(search / h)), 2)
    s = np.arange(-k, k + 1) * h          # (2k+1,) distances along the normal
    j0 = k                                 # index of s == 0

    cps = np.concatenate([fs.control_points for fs in fragmented], axis=0)
    nms = np.concatenate([fs.control_normals for fs in fragmented], axis=0)
    held = np.array([f.fixed for fs in fragmented for f in fs.fragments], dtype=bool)
    M = len(cps)
    if M == 0:
        return EPE(np.zeros(0), [], np.zeros(0))

    # (M, K, 2) sample positions, one normal line per fragment.
    pts = cps[:, None, :] + s[None, :, None] * nms[:, None, :]
    g = printed.sample(pts.reshape(-1, 2)).reshape(M, len(s))

    values = np.full(M, np.nan)
    status = ["ok"] * M
    for i in range(M):
        if held[i]:
            # Outside the imaged field: the sample above wrapped to the
            # far side of the periodic image and says nothing about this
            # edge. Not an error, not a failure — simply not measured.
            status[i] = "fixed"
            continue
        gi = g[i]
        if gi[j0] >= 0.0:
            # Inside the printed feature: walk outward to where it ends.
            out = np.nonzero(gi[j0 + 1:] < 0.0)[0]
            if out.size == 0:
                status[i] = "merged"
                continue
            j = j0 + 1 + int(out[0])
            a, b = gi[j - 1], gi[j]
            values[i] = s[j - 1] + (s[j] - s[j - 1]) * (a / (a - b))
        else:
            # Outside it: walk inward to where the feature begins.
            inn = np.nonzero(gi[:j0][::-1] >= 0.0)[0]
            if inn.size == 0:
                status[i] = "missing"
                continue
            j = j0 - 1 - int(inn[0])       # first inside sample, going inward
            a, b = gi[j], gi[j + 1]        # a >= 0 > b
            values[i] = s[j] + (s[j + 1] - s[j]) * (a / (a - b))
    inside = g[:, j0].copy()
    inside[held] = np.nan
    return EPE(values, status, inside)


__all__ = ["PrintedImage", "PrintModel", "EPE", "measure_epe"]
