"""
Hopkins imaging as a sum of coherent systems (SOCS).

The Abbe sum in :mod:`litho_sim.expose.aerial_image` pays one transform per
source point — a few hundred at the default source grid. Hopkins' form of
the same integral moves the source out of the loop::

    I(x) = Σ_{f, f'}  M̃(f) · TCC(f, f') · M̃*(f') · exp(2πi (f − f') · x)

with the transmission cross-coefficient ``TCC(f, f') = Σ_s S(s) P(f − s)
P*(f' − s)`` depending only on the optics.  ``TCC`` is Hermitian, so it has
an eigen-expansion ``Σ_k λ_k φ_k(f) φ_k*(f')``, and the image becomes a sum
of *coherent* images::

    I(x) = Σ_k  λ_k · |F⁻¹[ φ_k(f) · M̃(f) ]|²

That is the sum of coherent systems.  With every eigenvector kept it is the
Abbe sum exactly — same source points, same pupils, reassembled — and the
eigenvalues fall off fast, so a few dozen kernels carry all but a part in
1e-8 of the energy.  The transforms per image drop from one per source
point to one per kernel, which is what makes it the right engine for any
loop that holds the optics fixed and changes the mask: OPC iterations, mask
sweeps, a layout being edited live.

What it costs is a decomposition per optical state — defocus, aberrations
and source included — so a focus sweep gains nothing over Abbe, and it is
built for the scalar thin-mask process only.  A thick mask's spectrum
depends on the source point, which is precisely the dependence Hopkins
factors out, and the vector model would need one TCC per field component.
:meth:`SOCSKernels.build` refuses both rather than approximating.

The band is small.  On a 128-pixel, 4 nm grid at ArF the pupil radius is
under three frequency bins, so the frequencies any source point can pass
number a few dozen; the TCC is a matrix of that side and its
eigendecomposition is a millisecond.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft

from litho_sim.core.config import GridConfig, OpticsConfig
from litho_sim.core.utils import make_freq_grid
from litho_sim.expose.aerial_image import (
    _pupil_phase,
    normalisation_scale,
    source_points,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SOCSKernels:
    """The eigen-kernels of one optical state, ready to image any mask.

    Build with :meth:`build`; image with :meth:`image` or :meth:`raw`.

    Attributes
    ----------
    kernels : (K, n, n) complex array
        ``φ_k(f)`` on the full FFT grid in FFT ordering, zero outside the
        band. Each is unit-norm over the band.
    weights : (K,) array
        ``λ_k``, descending. Their sum is the open-frame intensity the
        kernels reproduce.
    clear : float
        Open-frame throughput — what an unpatterned mask images to — under
        this decomposition; the ``"clear"`` normalisation's denominator.
    energy_kept : float
        Fraction of the trace of the TCC the kept kernels carry.
    n_band : int
        Number of frequencies in the band; the size of the TCC that was
        decomposed, and the most kernels there could have been.
    n_source : int
        Source points the TCC was assembled from — the transforms an Abbe
        sum of the same image would have cost.
    optics, grid
        What the kernels were built for.
    """

    kernels: NDArray[np.complex128]
    weights: NDArray[np.float64]
    clear: float
    energy_kept: float
    n_band: int
    n_source: int
    optics: OpticsConfig
    grid: GridConfig

    # ------------------------------------------------------------------

    @staticmethod
    def supports(optics: OpticsConfig) -> bool:
        """Whether this decomposition applies to *optics* at all."""
        return optics.imaging_model.lower() == "scalar" and optics.mask_model == "thin"

    @classmethod
    def build(
        cls,
        optics: OpticsConfig,
        grid: GridConfig,
        tol: float = 1e-8,
        max_kernels: int | None = None,
    ) -> SOCSKernels:
        """Decompose the optics into kernels.

        Parameters
        ----------
        optics, grid
            The process. ``optics.defocus``, ``zernike_coeffs``, the source
            and the normalisation are all honoured; ``imaging_model`` must
            be ``"scalar"`` and ``mask_model`` ``"thin"``.
        tol : float
            Keep every eigenvalue at least ``tol × λ_max``. ``0`` keeps them
            all, which reproduces the Abbe sum to rounding.
        max_kernels : int, optional
            A hard cap on the count, applied after *tol*.

        Raises
        ------
        ValueError
            For a vector or thick-mask process.
        """
        if not cls.supports(optics):
            raise ValueError(
                "SOCS kernels apply to the scalar thin-mask process only; got "
                f"imaging_model={optics.imaging_model!r}, mask_model={optics.mask_model!r}. "
                "Use compute_aerial_image for those."
            )
        n = grid.n_pixels
        lam, NA = optics.wavelength, optics.NA
        k_max = NA / lam
        src = source_points(optics)

        # The band: every frequency some source point can pass through the
        # pupil. |f − s| ≤ k_max with |s| ≤ k_max puts it inside 2·k_max, and
        # nothing outside can contribute to any TCC element.
        FX, FY = make_freq_grid(n, grid.pixel_size)
        band = np.nonzero(np.hypot(FX, FY) <= 2.0 * k_max + 1e-9 * k_max)
        fx_b, fy_b = FX[band], FY[band]
        n_band = fx_b.size

        need_phi = bool(optics.zernike_coeffs)
        zernike = optics.zernike_coeffs or None

        # A[s, f] = sqrt(S_s) · P(f − s): one row per source point, evaluated
        # on the band alone. Exactly the pupils the Abbe loop would build,
        # restricted to where they are non-zero.
        A = np.zeros((len(src), n_band), dtype=np.complex128)
        for s in range(len(src)):
            dx, dy = fx_b - src.fx[s], fy_b - src.fy[s]
            rho = np.hypot(dx, dy) / k_max
            inside = rho <= 1.0
            if not inside.any():
                continue
            phi = np.arctan2(dy[inside], dx[inside]) if need_phi else None
            W = _pupil_phase(
                rho[inside], phi, lam, NA, optics.defocus, optics.image_index,
                zernike, optics.exact_defocus,
            )
            A[s, inside] = np.sqrt(src.weight[s]) * np.exp(1j * W)

        # TCC(f, f') = Σ_s A[s, f] · conj(A[s, f']) — Hermitian by construction.
        tcc = A.T @ A.conj()
        evals, evecs = np.linalg.eigh(tcc)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        trace = float(np.sum(np.clip(evals, 0.0, None)))

        keep = evals > (tol * evals[0] if evals.size else 0.0)
        keep &= evals > 0.0
        if max_kernels is not None:
            keep[max_kernels:] = False
        lam_k = evals[keep]
        vec_k = evecs[:, keep]
        K = int(lam_k.size)

        kernels = np.zeros((K, n, n), dtype=np.complex128)
        for k in range(K):
            kernels[k][band] = vec_k[:, k]

        # A clear mask has a single DC order, so its image is the kernels'
        # value at f = 0 — the same DC-bin bookkeeping the Abbe sum does.
        dc = np.nonzero((band[0] == 0) & (band[1] == 0))[0]
        clear = float(np.sum(lam_k * np.abs(vec_k[dc[0], :]) ** 2)) if dc.size else 0.0

        kept = float(lam_k.sum() / trace) if trace > 0 else 1.0
        logger.debug(
            "SOCS: %d kernels of %d band frequencies from %d source points, "
            "energy kept %.3e", K, n_band, len(src), kept,
        )
        return cls(kernels, lam_k, clear, kept, n_band, len(src), optics, grid)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.weights.size)

    def raw(self, mask: NDArray) -> tuple[NDArray[np.float64], float, float]:
        """The un-normalised, un-dosed image and its two normalising scalars.

        The same triple :func:`~litho_sim.expose.aerial_image.compute_aerial_image`
        returns with ``return_raw=True``, so the two engines are
        interchangeable behind :func:`normalisation_scale`.
        """
        n = self.grid.n_pixels
        if mask.shape != (n, n):
            raise ValueError(f"mask must be {(n, n)} for these kernels, got {mask.shape}")
        M_fft = np.fft.fft2(np.asarray(mask, dtype=np.complex128))
        fields = sfft.ifft2(self.kernels * M_fft[None], axes=(-2, -1), workers=-1)
        intensity = np.tensordot(self.weights, fields.real ** 2 + fields.imag ** 2, axes=1)
        return intensity, float(intensity.max()), self.clear

    def image(self, mask: NDArray, dose: float = 1.0) -> NDArray[np.float64]:
        """The aerial image, normalised as ``optics.normalisation`` says and dosed."""
        raw, peak, clear = self.raw(mask)
        return raw * normalisation_scale(self.optics.normalisation, peak, clear, dose)


__all__ = ["SOCSKernels"]
