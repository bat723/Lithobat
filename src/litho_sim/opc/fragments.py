"""
Edge fragmentation — the geometry OPC moves.

A drawn shape is a closed polygon. OPC does not move the polygon; it moves
*fragments* of its edges, each along its own outward normal, and rebuilds
the polygon from where the fragments ended up. That decomposition is the
whole representational idea behind edge-based OPC: it turns "correct this
shape" into a vector of scalar unknowns — one offset per fragment — which is
something an iteration can converge.

Fragmentation is where OPC's shape vocabulary comes from. A line end is one
short fragment; the last stretch of each long edge before a corner is its
own fragment (a *corner fragment*). When the loop pushes the end fragment
out and the two corner fragments beside it out as well, the result is a
hammerhead. Nobody drew a hammerhead. Serifs come from the same place. This
is why the fragmentation lengths are the parameters that matter most in
practice, and why they default to a fraction of the optical resolution
``λ/NA`` rather than to a fixed number of nanometres — at EUV the same
rule gives fragments five times shorter than at ArF.

Reconstruction
--------------
Each fragment's offset segment is its original segment shifted by ``δ·n``.
Consecutive fragments are joined by a mitre (the intersection of the two
offset lines) where they meet at a corner, and by a *jog* — a short step
perpendicular to the edge — where they lie on the same straight edge but
were moved by different amounts. Jogs are what a corrected mask looks like
under a microscope: a staircase of small steps along every edge.

Coordinates are metres from field centre, as everywhere in
:mod:`litho_sim.mask`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from litho_sim.mask.geometry import Polygon, Shape
from litho_sim.mask.layout import Layout

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fragments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fragment:
    """One movable piece of a polygon edge.

    Attributes
    ----------
    p0, p1 : (2,) arrays
        Endpoints in metres, in the polygon's counter-clockwise order.
    normal : (2,) array
        Outward unit normal — the direction a positive offset moves it.
    edge : int
        Index of the original polygon edge this fragment came from.
    role : str
        ``"body"`` for the middle of a long edge, ``"corner"`` for the stretch
        beside a vertex, ``"end"`` for an edge short enough to be one
        fragment — a line end, or a side of a contact.
    site, site_normal : (2,) arrays, optional
        Where the edge placement error is measured and along which
        direction — the *target*. ``None`` means the fragment's own midpoint
        and normal. A fragment beside a convex corner is retargeted onto
        the rounded corner the optics can actually print; see
        :func:`fragment_shape`.
    fixed : bool
        Held at the drawn position and never measured. What a fragment
        outside the imaged field becomes when a shape runs out of it (see
        the ``clip`` of :func:`fragment_shape`): the imaging is periodic,
        so a control point past the field edge would sample the wrong
        side of the wrap and read a through-field line as a line end.
    """

    p0: NDArray[np.float64]
    p1: NDArray[np.float64]
    normal: NDArray[np.float64]
    edge: int
    role: str = "body"
    site: NDArray[np.float64] | None = None
    site_normal: NDArray[np.float64] | None = None
    fixed: bool = False

    @property
    def length(self) -> float:
        return float(np.hypot(*(self.p1 - self.p0)))

    @property
    def midpoint(self) -> NDArray[np.float64]:
        """The fragment's centre — what it moves."""
        return 0.5 * (self.p0 + self.p1)

    @property
    def control_point(self) -> NDArray[np.float64]:
        """Where its EPE is measured: the retargeted site, else the midpoint."""
        return self.midpoint if self.site is None else self.site

    @property
    def control_normal(self) -> NDArray[np.float64]:
        return self.normal if self.site_normal is None else self.site_normal

    @property
    def retargeted(self) -> bool:
        return self.site is not None

    @property
    def tangent(self) -> NDArray[np.float64]:
        d = self.p1 - self.p0
        return d / max(float(np.hypot(*d)), 1e-30)


def _ccw(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """The polygon with counter-clockwise winding, so outward is well-defined."""
    x, y = points[:, 0], points[:, 1]
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return points if area2 >= 0 else points[::-1].copy()


def _dedupe(points: NDArray[np.float64], eps: float = 1e-15) -> NDArray[np.float64]:
    """Drop consecutive duplicate vertices (and a closing vertex equal to the first)."""
    if len(points) == 0:
        return points
    keep = [points[0]]
    for p in points[1:]:
        if np.hypot(*(p - keep[-1])) > eps:
            keep.append(p)
    if len(keep) > 1 and np.hypot(*(keep[-1] - keep[0])) <= eps:
        keep.pop()
    return np.asarray(keep, dtype=np.float64)


def _simplify(points: NDArray[np.float64], eps: float = 1e-15) -> NDArray[np.float64]:
    """Drop duplicate and collinear vertices, so a polygon carries only its corners.

    Two fragments moved by the same amount meet at a point that is no
    corner; keeping it would make every corrected rectangle a 30-gon and
    every jog invisible among the clutter.
    """
    pts = _dedupe(points, eps)
    n = len(pts)
    if n < 4:
        return pts
    keep = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        u, v = b - a, c - b
        cross = u[0] * v[1] - u[1] * v[0]
        straight = abs(cross) <= eps * max(np.hypot(*u) * np.hypot(*v), 1e-300) * 1e6
        if not (straight and float(u @ v) > 0.0):
            keep.append(b)
    return np.asarray(keep, dtype=np.float64) if len(keep) >= 3 else pts


def _split_edge(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    edge: int,
    normal: NDArray[np.float64],
    fragment_length: float,
    corner_length: float,
) -> list[Fragment]:
    """Cut one edge into fragments: corner pieces at both ends, body between.

    An edge shorter than one corner piece plus a minimal body is a single
    ``"end"`` fragment — the case a line end and a contact side fall into.
    """
    L = float(np.hypot(*(b - a)))
    if L <= 0.0:
        return []
    # Room for two corner pieces and at least a fragment-length of body?
    if L < 2.0 * corner_length + 0.5 * fragment_length or corner_length <= 0.0:
        n = max(1, int(math.ceil(L / fragment_length))) if fragment_length > 0 else 1
        role = "end" if n == 1 else "body"
        cuts = np.linspace(0.0, 1.0, n + 1)
        return [
            Fragment(a + (b - a) * t0, a + (b - a) * t1, normal, edge, role)
            for t0, t1 in zip(cuts[:-1], cuts[1:])
        ]
    body = L - 2.0 * corner_length
    n_body = max(1, int(math.ceil(body / fragment_length)))
    ts = [0.0, corner_length / L]
    ts += list(corner_length / L + (body / L) * np.linspace(0.0, 1.0, n_body + 1)[1:])
    ts[-1] = 1.0 - corner_length / L
    ts.append(1.0)
    roles = ["corner"] + ["body"] * n_body + ["corner"]
    return [
        Fragment(a + (b - a) * t0, a + (b - a) * t1, normal, edge, role)
        for t0, t1, role in zip(ts[:-1], ts[1:], roles)
    ]


@dataclass
class FragmentedShape:
    """A shape as its ordered fragments, with the offset each has been moved by.

    Attributes
    ----------
    source : Shape
        The drawn shape, kept for its layer and for the record.
    fragments : list of Fragment
        In counter-clockwise order around the polygon.
    offsets : (M,) array
        Current move of each fragment along its normal [m]. Zero is the
        drawn shape.
    """

    source: Shape
    fragments: list[Fragment]
    offsets: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if self.offsets.shape != (len(self.fragments),):
            self.offsets = np.zeros(len(self.fragments), dtype=np.float64)

    def __len__(self) -> int:
        return len(self.fragments)

    @property
    def control_points(self) -> NDArray[np.float64]:
        """``(M, 2)`` measurement sites [m] — midpoints, or retargeted corners."""
        return np.array([f.control_point for f in self.fragments], dtype=np.float64).reshape(-1, 2)

    @property
    def control_normals(self) -> NDArray[np.float64]:
        """``(M, 2)`` directions the EPE is measured along."""
        return np.array([f.control_normal for f in self.fragments], dtype=np.float64).reshape(-1, 2)

    @property
    def normals(self) -> NDArray[np.float64]:
        """``(M, 2)`` outward normals — the directions the fragments move."""
        return np.array([f.normal for f in self.fragments], dtype=np.float64).reshape(-1, 2)

    def shape(self, max_mitre: float | None = None) -> Polygon:
        """Rebuild the polygon from the fragments at their current offsets."""
        return Polygon(self.source.layer, tuple(map(tuple, self._rebuild(max_mitre))))

    def _rebuild(self, max_mitre: float | None) -> NDArray[np.float64]:
        frags, offs = self.fragments, self.offsets
        m = len(frags)
        if m == 0:
            return np.zeros((0, 2))
        out: list[NDArray[np.float64]] = []
        for i in range(m):
            f, g = frags[i], frags[(i + 1) % m]
            d_f, d_g = offs[i], offs[(i + 1) % m]
            end_f = f.p1 + d_f * f.normal
            start_g = g.p0 + d_g * g.normal
            cross = float(f.tangent[0] * g.tangent[1] - f.tangent[1] * g.tangent[0])
            if abs(cross) < 1e-9:
                # Same straight edge (or collinear edges): a jog, which
                # collapses to one point when the two offsets agree.
                out.append(end_f)
                out.append(start_g)
                continue
            # A corner: mitre the two offset lines. Solve end_f + s·t_f =
            # start_g + u·t_g for s.
            r = start_g - end_f
            s = (r[0] * g.tangent[1] - r[1] * g.tangent[0]) / cross
            mitre = end_f + s * f.tangent
            limit = max_mitre if max_mitre is not None else 4.0 * max(abs(d_f), abs(d_g)) + 1e-12
            if np.hypot(*(mitre - f.p1)) > limit:
                # Too acute to mitre sanely; bevel instead.
                out.append(end_f)
                out.append(start_g)
            else:
                out.append(mitre)
        return _simplify(np.asarray(out, dtype=np.float64))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _retarget_corners(
    frags: list[Fragment], pts: NDArray[np.float64], radius: float
) -> list[Fragment]:
    """Move the measurement site of every fragment beside a convex vertex onto
    the arc of *radius* tangent to both edges there.

    A projection lens cannot print a corner sharper than roughly half its
    resolution, so a fragment measuring against the drawn vertex sees an
    EPE no offset can remove — and keeps pushing. Retargeting says what
    the design actually needs: the edge in the right place, and the corner
    as round as it has to be. A convex corner is rounded off; a concave
    one — the inside of an L or a T — fills in, and gets a fillet of the
    same radius on the outside. Without it the inner-corner fragments
    notch the mask to the mask-rule cap and the residual still grows.

    The arc at a vertex whose edges meet at angle φ (interior for convex,
    exterior for concave) touches each edge at distance ``q = r / tan(φ/2)``
    from it. An edge too short for that is rounded with the largest radius
    it has room for, so a narrow line end targets a semicircle rather than
    nothing. A corner fragment's site is its own midpoint projected
    radially onto the arc, so the two fragments meeting at a vertex get two
    distinct sites and do not chase one error twice. Only ``"corner"`` and
    ``"end"`` fragments are retargeted; a body fragment's site is its
    midpoint.
    """
    if radius <= 0.0 or len(frags) == 0:
        return frags
    n = len(pts)
    angle = np.zeros(n)          # the angle the arc spans between: interior or exterior
    convex = np.zeros(n, dtype=bool)
    straight = np.zeros(n, dtype=bool)
    edge_len = np.zeros(n)
    for i in range(n):
        u = pts[i] - pts[i - 1]
        v = pts[(i + 1) % n] - pts[i]
        cross = u[0] * v[1] - u[1] * v[0]
        cosang = float(np.clip((u @ v) / max(np.hypot(*u) * np.hypot(*v), 1e-300), -1, 1))
        turn = math.acos(cosang)                    # 0 on a straight edge
        angle[i] = math.pi - turn                   # the wedge the arc sits in
        convex[i] = cross > 1e-30
        straight[i] = turn < 1e-6
        edge_len[i] = float(np.hypot(*v))

    def radius_at(i: int) -> float:
        """The largest radius ≤ *radius* whose arc fits on both edges at vertex i."""
        r = radius
        for e in (i - 1, i):                      # the edge into i and the edge out of it
            r = min(r, 0.5 * edge_len[e % n] * math.tan(0.5 * angle[i]))
        return r

    out: list[Fragment] = []
    for f in frags:
        if f.role not in ("corner", "end"):
            out.append(f)
            continue
        i0, i1 = f.edge, (f.edge + 1) % n
        a, b = pts[i0], pts[i1]
        t = (b - a) / edge_len[i0]
        mid = f.midpoint
        d0 = float((mid - a) @ t)
        d1 = edge_len[i0] - d0
        # The vertex this fragment belongs to: the nearer one, if convex.
        candidates = sorted(((d0, i0, +1.0), (d1, i1, -1.0)))
        site, snorm = None, None
        for d, i, sign in candidates:
            if straight[i]:
                continue
            r = radius_at(i)
            q = r / math.tan(0.5 * angle[i])
            if d > q * (1.0 + 1e-9) or r <= 0.0:
                continue
            Q = pts[i] + sign * q * t                 # tangent point on this edge
            # The arc centre sits inside the feature at a convex corner and
            # outside it at a concave one; the site is the arc point under
            # the midpoint, and the outward normal always runs from inside
            # to outside — away from the centre for a convex arc, toward
            # it for a fillet.
            inward = -1.0 if convex[i] else +1.0
            C = Q + inward * r * f.normal
            if f.role == "end":
                depth = r - math.sqrt(max(r * r - (q - d) ** 2, 0.0))
                site = mid + inward * depth * f.normal
                radial = site - C
            else:
                radial = mid - C
                site = C + r * radial / max(float(np.hypot(*radial)), 1e-300)
            snorm = -inward * radial / max(float(np.hypot(*radial)), 1e-300)
            break
        out.append(Fragment(f.p0, f.p1, f.normal, f.edge, f.role, site, snorm))
    return out


def fragment_shape(
    shape: Shape,
    fragment_length: float,
    corner_length: float | None = None,
    corner_radius: float | None = None,
    clip: tuple[float, float, float, float] | None = None,
) -> FragmentedShape:
    """Split one shape's outline into fragments.

    Parameters
    ----------
    shape : Shape
        Any mask shape. Its :meth:`~litho_sim.mask.geometry.Shape.polygon`
        is what gets fragmented, so a round contact fragments into its
        32 sides.
    fragment_length : float
        Target length of a body fragment [m]. Edges are divided into equal
        pieces no longer than this.
    corner_length : float, optional
        Length of the fragment beside each vertex [m]. ``None`` uses half
        the fragment length; ``0`` turns corner fragments off.
    corner_radius : float, optional
        Radius the convex corners are retargeted to [m]; see
        :func:`_retarget_corners`. ``None`` uses the fragment length;
        ``0`` keeps the drawn corners as targets — and shows why nobody
        does that.
    clip : (x0, y0, x1, y1), optional
        The imaged field [m]. Fragments whose control point lies outside
        it are marked ``fixed``: never measured, never moved. For a shape
        drawn through the whole field — a line in a periodic array — this
        is what keeps its window edges from being corrected as line ends.

    Returns
    -------
    FragmentedShape
        With all offsets at zero.
    """
    if fragment_length <= 0.0:
        raise ValueError(f"fragment_length must be positive, got {fragment_length}")
    if corner_length is None:
        corner_length = 0.5 * fragment_length
    if corner_radius is None:
        corner_radius = fragment_length if np.isfinite(fragment_length) else 0.0
    pts = _dedupe(_ccw(np.asarray(shape.polygon(), dtype=np.float64)))
    if len(pts) < 3:
        raise ValueError(f"{type(shape).__name__} degenerates to fewer than 3 distinct points")
    frags: list[Fragment] = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        d = b - a
        L = float(np.hypot(*d))
        if L <= 0.0:
            continue
        # Counter-clockwise winding: outward is to the right of travel.
        normal = np.array([d[1], -d[0]], dtype=np.float64) / L
        frags.extend(_split_edge(a, b, i, normal, fragment_length, corner_length))
    frags = _retarget_corners(frags, pts, corner_radius)
    if clip is not None:
        frags = _hold_outside(frags, clip)
    return FragmentedShape(source=shape, fragments=frags)


def _hold_outside(frags: list[Fragment], clip: tuple[float, float, float, float]) -> list[Fragment]:
    """Mark every fragment whose control point lies outside *clip* as fixed.

    The box is ``(x0, y0, x1, y1)`` in metres — normally the imaged field.
    A fragment measured outside it would sample the periodic image on the
    far side of the wrap, which for a line drawn through the whole field
    turns its window edge into a "line end" that reads ``merged`` at every
    iteration and pulls the line apart. Holding such fragments is what a
    simulation window means: the correction is for what is imaged.
    """
    x0, y0, x1, y1 = clip
    out = []
    for f in frags:
        cx, cy = f.control_point
        inside = x0 <= cx <= x1 and y0 <= cy <= y1
        out.append(f if inside else replace(f, fixed=True))
    return out


def fragment_layout(
    layout: Layout,
    fragment_length: float,
    corner_length: float | None = None,
    layer: str | None = None,
    corner_radius: float | None = None,
    clip: tuple[float, float, float, float] | None = None,
) -> list[FragmentedShape]:
    """Fragment every shape of a layout (or of one layer of it), in order.

    *clip* is the imaged field ``(x0, y0, x1, y1)`` [m]; fragments whose
    control point falls outside it are held fixed — see
    :func:`fragment_shape`.
    """
    shapes = layout.shapes if layer is None else layout.on_layer(layer)
    out = [
        fragment_shape(s, fragment_length, corner_length, corner_radius, clip=clip)
        for s in shapes
    ]
    logger.debug(
        "Fragmented %d shapes into %d fragments (%.1f nm body, %s corner, %d held)",
        len(out), sum(len(f) for f in out), fragment_length * 1e9,
        "none" if corner_length == 0 else f"{(corner_length or fragment_length / 2) * 1e9:.1f} nm",
        sum(f.fixed for fs in out for f in fs.fragments),
    )
    return out


def rebuild_layout(
    fragmented: Sequence[FragmentedShape],
    name: str = "corrected",
    max_mitre: float | None = None,
) -> Layout:
    """The layout the fragments describe at their current offsets.

    Every shape comes back as a :class:`~litho_sim.mask.geometry.Polygon`
    on its source shape's layer, so the result rasterises, serialises and
    drops into an :class:`~litho_sim.patterning.steps.Expose` step exactly
    like the drawn layout did.
    """
    return Layout([fs.shape(max_mitre) for fs in fragmented], name=name)


def facing_distances(
    fragmented: Sequence[FragmentedShape],
    direction: float = +1.0,
    probes: int = 3,
) -> NDArray[np.float64]:
    """How far each fragment can travel before it meets the feature it faces [m].

    Casts rays from *probes* points along every fragment, along its outward
    normal (``direction=+1``) or inward (``−1``), and returns the distance
    to the nearest fragment segment of *any* shape whose normal faces back.
    Outward, that is the space to the neighbouring feature — or to the
    other arm of the same concave shape. Inward, it is the feature's own
    width. Fragments with nothing in the way get ``inf``.

    This is the geometry a mask rule check needs: a fragment may take its
    share of that distance and no more.
    """
    segs_p0 = np.concatenate([[f.p0 for f in fs.fragments] for fs in fragmented], axis=0)
    segs_p1 = np.concatenate([[f.p1 for f in fs.fragments] for fs in fragmented], axis=0)
    normals = np.concatenate([fs.normals for fs in fragmented], axis=0)
    M = len(normals)
    out = np.full(M, np.inf)
    if M == 0:
        return out
    e = segs_p1 - segs_p0                                    # (M, 2)
    ts = np.linspace(0.0, 1.0, probes + 2)[1:-1]             # interior probe positions
    for i in range(M):
        d = direction * normals[i]
        # A segment faces this one when its outward normal opposes ours:
        # outward, that is the near side of the neighbouring feature; inward,
        # the far side of our own. Either way the near side of our own
        # feature, whose normal is ours, is excluded.
        facing = (normals @ normals[i]) < -1e-9
        facing[i] = False
        if not facing.any():
            continue
        P0, E = segs_p0[facing], e[facing]
        denom = d[0] * E[:, 1] - d[1] * E[:, 0]              # d × e
        ok = np.abs(denom) > 1e-30
        best = np.inf
        for t in ts:
            o = segs_p0[i] + t * e[i]
            w = P0 - o
            with np.errstate(divide="ignore", invalid="ignore"):
                s_hit = (w[:, 0] * E[:, 1] - w[:, 1] * E[:, 0]) / denom
                u_hit = (w[:, 0] * d[1] - w[:, 1] * d[0]) / denom
            hit = ok & (s_hit > 1e-12) & (u_hit >= -1e-9) & (u_hit <= 1.0 + 1e-9)
            if hit.any():
                best = min(best, float(s_hit[hit].min()))
        out[i] = best
    return out


def bias_layout(
    layout: Layout,
    bias: float,
    layer: str | None = None,
    name: str | None = None,
) -> Layout:
    """Move every edge of every shape outward by *bias* metres (inward if negative).

    The rule-based ancestor of model-based OPC and still the first thing a
    lithographer reaches for: a fixed per-edge bias that makes a feature
    print to size at the anchor condition. Analytic, so a 0.7 nm bias on a
    4 nm grid is represented — the pixel-based
    :func:`~litho_sim.mask.patterns.apply_bias` rounds it to zero.
    """
    fragmented = fragment_layout(layout, fragment_length=np.inf, corner_length=0.0, layer=layer)
    for fs in fragmented:
        fs.offsets[:] = bias
    return rebuild_layout(fragmented, name=name or f"{layout.name}+{bias * 1e9:.1f}nm")


__all__ = [
    "Fragment",
    "FragmentedShape",
    "fragment_shape",
    "fragment_layout",
    "rebuild_layout",
    "bias_layout",
    "facing_distances",
]
