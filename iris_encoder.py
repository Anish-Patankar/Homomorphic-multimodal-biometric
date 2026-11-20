"""
iris_encoder.py
================
Converts a raw iris eye image into a 2048-bit ("256-byte") iris code,
following the pipeline described in:

    J. Daugman, "High Confidence Visual Recognition of Persons by a Test
    of Statistical Independence," IEEE TPAMI, vol. 15, no. 11, 1993.

Pipeline
--------
1. Segmentation       : locate pupil boundary and iris (limbus) boundary
                         as circles (x, y, r), via Hough-circle search.
2. Normalization       : "rubber sheet" model -> map the annular iris
                         region into a fixed-size (radial x angular)
                         rectangular image I(r, theta), independent of
                         pupil dilation / image scale (eqs. 10-12).
3. Noise masking        : flag specular reflections, near-black/near-white
                         pixels, and unreliable radial rings so those bits
                         are excluded from matching.
4. Encoding             : convolve the normalized image with a quadrature
                         pair of 2D Gabor filters (eq. 13) and, at a fixed
                         1024-point sampling grid, quantize the sign of the
                         real and imaginary filter responses into 2 bits
                         each (eqs. 14-17) -> 1024 * 2 = 2048-bit code.
5. Matching             : masked, rotation-tolerant (best-of-n shifts)
                         fractional Hamming distance (eqs. 21, 33-34).

This module produces plaintext iris codes + masks. In the larger project
these bits are what get encrypted under Kyber-AHE before the XOR/Hamming
step is executed on ciphertexts -- this file only needs to guarantee that
the *plaintext* codes separate genuine vs impostor Hamming distances
cleanly, since AHE decryption of XOR(c1, c2) reproduces XOR(code1, code2)
bit-for-bit.

Dependencies: opencv-python, numpy
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Circle:
    x: int
    y: int
    r: int


@dataclasses.dataclass
class IrisCode:
    bits: np.ndarray   # shape (2048,), dtype uint8, values in {0,1}
    mask: np.ndarray   # shape (2048,), dtype uint8, 1 = usable bit, 0 = masked out
    grid_shape: Tuple[int, int]  # (n_rows, n_cols) sampling grid, rows*cols*2 == 2048

    def packed(self) -> bytes:
        """Pack the 2048 bits into 256 bytes (MSB-first), matching the
        paper's '256-byte iris code' representation."""
        return np.packbits(self.bits).tobytes()

    def packed_mask(self) -> bytes:
        return np.packbits(self.mask).tobytes()

    def bit_string(self) -> str:
        """Return the 2048 bits as a binary string of '0's and '1's."""
        return (self.bits + ord('0')).tobytes().decode('ascii')

    def mask_string(self) -> str:
        """Return the 2048 mask bits as a binary string of '0's and '1's."""
        return (self.mask + ord('0')).tobytes().decode('ascii')


class SegmentationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# 1. Segmentation
# --------------------------------------------------------------------------- #

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def preprocess_gray(image: np.ndarray) -> np.ndarray:
    """Grayscale conversion + CLAHE, shared by every segmentation path.

    CLAHE (local/adaptive equalization) is used instead of global
    equalizeHist: global equalization over-amplifies noise in flat regions
    (sclera, skin, background) by an amount that depends on that image's
    overall histogram shape -- which varies shot-to-shot even for the same
    eye, and was found empirically to destabilize circle detection between
    different photos of the same iris. CLAHE boosts local contrast at the
    actual pupil/limbus edges without that global side effect. Applying the
    same preprocessing when boundaries come from an external segmenter keeps
    the downstream encoding identical, so only segmentation differs."""
    gray = _to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_pupil(
    gray: np.ndarray,
    min_radius_frac: float = 0.08,
    max_radius_frac: float = 0.28,
    param1: int = 60,
    param2: int = 30,
) -> Circle:
    """Locate the pupil as the darkest, most circular blob via Hough circles.

    Radii are expressed as a fraction of min(image dimensions) so the same
    call works across datasets with different image sizes.
    """
    h, w = gray.shape
    dim = min(h, w)
    min_r = max(5, int(dim * min_radius_frac))
    max_r = max(min_r + 5, int(dim * max_radius_frac))

    blurred = cv2.medianBlur(gray, 7)
    # Pupil is dark: boost contrast on the low end before Hough.
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=dim,
        param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        # Retry with a looser accumulator threshold.
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=dim,
            param1=param1, param2=max(10, param2 - 12), minRadius=min_r, maxRadius=max_r,
        )
    if circles is None:
        raise SegmentationError("Pupil not found")

    cand = np.round(circles[0, :]).astype(int)
    # Keep candidates inside the image, pick the one with lowest mean interior
    # intensity (pupil is the darkest circular region in the eye).
    best, best_score = None, None
    for x, y, r in cand:
        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            continue
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (x, y), max(1, int(r * 0.6)), 255, -1)
        score = cv2.mean(gray, mask=mask)[0]
        if best_score is None or score < best_score:
            best, best_score = (x, y, r), score
    if best is None:
        x, y, r = cand[0]
        best = (int(x), int(y), int(r))
    return Circle(*best)


def _edge_contrast_score(gray: np.ndarray, x0: float, y0: float, r: float,
                          thetas: np.ndarray, delta: int = 3) -> float:
    """Cheap edge-strength score for a candidate circle: contrast between
    mean intensity just inside vs. just outside radius r, along the given
    arc. Used to validate/rank circle candidates against a genuine
    intensity edge, rather than trusting Hough's own candidate ordering or
    a purely geometric heuristic."""
    inner = _mean_circle_intensity(gray, x0, y0, max(1, r - delta), thetas)
    outer = _mean_circle_intensity(gray, x0, y0, r + delta, thetas)
    return abs(outer - inner)


def detect_iris(
    gray: np.ndarray,
    pupil: Circle,
    min_radius_factor: float = 1.8,
    max_radius_factor: float = 4.2,
    param1: int = 40,
    param2: int = 20,
) -> Circle:
    """Locate the outer (limbus) boundary given the pupil circle.

    Generates a pool of candidates (Hough proposals, plus fallback circles
    concentric with the pupil at a range of radii) and picks the one with
    the strongest genuine edge contrast along the horizontal-meridian arcs
    -- i.e. the arcs actually used for limbus localization -- rather than
    simply the Hough candidate closest to the pupil center. This avoids
    latching onto strong-but-irrelevant edges elsewhere in the image (most
    commonly the eyebrow / eyelash region, which produces a much stronger
    Hough response than the true iris-sclera boundary on many real photos).
    """
    h, w = gray.shape
    abs_max_r = int(min(h, w) / 2 * 0.98)
    min_r = int(pupil.r * min_radius_factor)
    max_r = min(abs_max_r, int(pupil.r * max_radius_factor))
    max_r = max(max_r, min_r + 5)

    # Narrow horizontal-meridian arcs: wide enough to get a stable edge
    # estimate, narrow enough to stay well clear of the eyebrow/eyelid
    # region even when the candidate radius is large.
    thetas = _arc_thetas([(-25, 25), (155, 205)])

    candidates = []

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min(h, w),
        param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r,
    )
    if circles is not None:
        for x, y, r in np.round(circles[0, :]).astype(int):
            r = int(min(r, abs_max_r))
            center_offset = np.hypot(x - pupil.x, y - pupil.y)
            # Anatomical sanity bound: the paper reports pupil/iris
            # non-concentricity is typically at most ~15% of iris radius.
            # We allow a generous margin above that (25%) to tolerate real
            # variation and estimation slack, but reject candidates beyond
            # it outright -- without this, a strong-but-wrong edge
            # elsewhere in the image (eye corner, eyelid fold, skin crease)
            # can win purely on edge contrast despite being anatomically
            # implausible. This was directly observed: one real IITD image
            # pair had the same eye's "limbus" jump by 40px / ~36% of
            # radius between two photos, consistent with exactly this
            # failure mode.
            if center_offset <= 0.25 * r:
                candidates.append(Circle(int(x), int(y), r))

    # Fallback / supplementary candidates: concentric with the pupil at a
    # sweep of plausible radii. Real irises are only mildly non-concentric
    # with the pupil (paper: typically <=15% offset), so this is a strong
    # prior when Hough is unreliable, and it also protects against Hough
    # returning nothing at all.
    for factor in np.arange(min_radius_factor, max_radius_factor + 0.01, 0.25):
        r = min(int(pupil.r * factor), abs_max_r)
        if r > min_r:
            candidates.append(Circle(pupil.x, pupil.y, r))

    if not candidates:
        raise SegmentationError("No iris/limbus candidates found")

    best_circle, best_score = None, -1.0
    for c in candidates:
        score = _edge_contrast_score(gray, c.x, c.y, c.r, thetas)
        if score > best_score:
            best_circle, best_score = c, score

    return best_circle


def _arc_thetas(arcs_deg, n_samples_per_90deg: int = 90) -> np.ndarray:
    """Build a theta sample array (radians) covering the given list of
    (start_deg, end_deg) arcs."""
    thetas = []
    for a0, a1 in arcs_deg:
        n = max(4, int(n_samples_per_90deg * (a1 - a0) / 90.0))
        thetas.append(np.deg2rad(np.linspace(a0, a1, n, endpoint=False)))
    return np.concatenate(thetas)


def _mean_circle_intensity(gray: np.ndarray, x0: float, y0: float, r: float,
                            thetas: np.ndarray) -> float:
    h, w = gray.shape
    xs = x0 + r * np.cos(thetas)
    ys = y0 + r * np.sin(thetas)
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    return float(gray[yi, xi].mean())


def _parabolic_peak(f_minus: float, f_center: float, f_plus: float, center: float) -> float:
    """Sub-pixel peak location via parabolic interpolation of three
    equally-spaced samples straddling a discrete maximum. Standard
    technique for recovering sub-pixel precision from an integer-grid
    search -- without it, two photos of the same eye whose true boundary
    sits at, say, radius 107.6px can snap to different integer estimates
    (107 vs 108), which is enough to introduce a small radial-scale
    mismatch between their rubber-sheet mappings that angular rotation
    correction cannot fix."""
    denom = f_minus - 2 * f_center + f_plus
    if denom == 0:
        return float(center)
    offset = 0.5 * (f_minus - f_plus) / denom
    offset = float(np.clip(offset, -1.0, 1.0))
    return center + offset


def refine_circle(
    gray: np.ndarray,
    init: Circle,
    thetas: np.ndarray,
    center_search: int = 4,
    r_search: int = 10,
    edge_delta: int = 3,
    subpixel: bool = True,
) -> Circle:
    """Local refinement of a circle estimate: search a small neighborhood
    of (x0, y0, r) around `init` for the circle with the strongest
    inside-vs-outside intensity edge contrast along the given arc
    (`thetas` restricts the arc, e.g. two narrow cones at the horizontal
    meridian for the limbus, or the upper 270 deg for the pupil, to avoid
    eyelids / specular reflection, per the paper). Uses the same
    edge-contrast metric as candidate selection in detect_iris, so scoring
    is consistent across the whole segmentation pipeline.

    If `subpixel` is True (default), a parabolic-interpolation pass over
    (r, then x, then y) refines the integer-grid winner to sub-pixel
    precision, which materially improves consistency of the boundary
    estimate across different photos of the same eye.
    """
    best = None
    rs = np.arange(max(1, init.r - r_search), init.r + r_search + 1)
    for dx in range(-center_search, center_search + 1):
        for dy in range(-center_search, center_search + 1):
            x0, y0 = init.x + dx, init.y + dy
            for r in rs:
                score = _edge_contrast_score(gray, x0, y0, int(r), thetas, delta=edge_delta)
                if best is None or score > best[0]:
                    best = (score, x0, y0, int(r))

    best_score, x0, y0, r = best
    if not subpixel:
        return Circle(x0, y0, r)

    # Sub-pixel refinement, sequentially over r, x, y (each conditioned on
    # the current best estimate of the others -- a standard, cheap
    # approximation to a full joint sub-pixel optimization).
    r_m = _edge_contrast_score(gray, x0, y0, r - 1, thetas, delta=edge_delta)
    r_p = _edge_contrast_score(gray, x0, y0, r + 1, thetas, delta=edge_delta)
    r_sub = _parabolic_peak(r_m, best_score, r_p, r)

    xc = _edge_contrast_score(gray, x0, y0, r_sub, thetas, delta=edge_delta)
    x_m = _edge_contrast_score(gray, x0 - 1, y0, r_sub, thetas, delta=edge_delta)
    x_p = _edge_contrast_score(gray, x0 + 1, y0, r_sub, thetas, delta=edge_delta)
    x_sub = _parabolic_peak(x_m, xc, x_p, x0)

    yc = _edge_contrast_score(gray, x_sub, y0, r_sub, thetas, delta=edge_delta)
    y_m = _edge_contrast_score(gray, x_sub, y0 - 1, r_sub, thetas, delta=edge_delta)
    y_p = _edge_contrast_score(gray, x_sub, y0 + 1, r_sub, thetas, delta=edge_delta)
    y_sub = _parabolic_peak(y_m, yc, y_p, y0)

    return Circle(x_sub, y_sub, r_sub)


def segment_iris(image: np.ndarray, refine: bool = True) -> Tuple[Circle, Circle, np.ndarray]:
    """Full segmentation: returns (pupil_circle, iris_circle, gray_image).

    `refine=True` (default) runs a local integrodifferential refinement
    pass after the initial Hough-circle estimate, restricted to the same
    angular arcs the paper uses (upper 270 deg for the pupil to avoid the
    lower-mounted illuminator's corneal reflection; two 90 deg cones at
    the horizontal meridian for the limbus to avoid eyelid occlusion).
    This materially improves consistency of the circle estimates across
    different images of the *same* eye, which is what genuine-pair
    Hamming distance separation depends on.
    """
    gray = preprocess_gray(image)

    pupil = detect_pupil(gray)
    iris = detect_iris(gray, pupil)
    if iris.r <= pupil.r:
        raise SegmentationError("Iris radius must exceed pupil radius")

    if refine:
        # Upper-270-degree arc, correctly excluding the 90 deg cone centered
        # at theta=90 (downward, +y) where the below-mounted illuminator's
        # corneal reflection falls. NOTE: an earlier version of this arc
        # was ((-135,45),(45,135)), which -- despite its comment's stated
        # intent -- actually combines into one continuous span through
        # theta=0 and excludes a cone centered at theta=180 (left) instead.
        # A single arc from 135 to 405 degrees (wrapping through 180/270/0)
        # covers exactly the 270 deg region excluding [45,135] as intended.
        pupil_thetas = _arc_thetas([(135, 405)])
        # NOTE: excludes the cone centered at theta=90 (downward, +y), where
        # the below-mounted illuminator's corneal reflection typically falls.
        pupil = refine_circle(gray, pupil, pupil_thetas, center_search=3, r_search=6)

        limbus_thetas = _arc_thetas([(-25, 25), (155, 205)])  # narrow cones at horizontal meridian
        iris = refine_circle(gray, iris, limbus_thetas, center_search=4, r_search=8)

        # Refinement searches a neighborhood relative to its *own* starting
        # point, so it is not automatically bounded by the image-size cap
        # applied during initial detection. Re-enforce it here -- without
        # this, refinement can walk the iris radius past the edge of the
        # frame (e.g. toward a strong-but-wrong edge like an eyebrow).
        h, w = gray.shape
        abs_max_r = int(min(h, w) / 2 * 0.98)
        iris = Circle(iris.x, iris.y, min(iris.r, abs_max_r))

        if iris.r <= pupil.r:
            raise SegmentationError("Iris radius must exceed pupil radius after refinement")

        # Sanity check: reject segmentations with no real edge evidence
        # along the arcs actually used, rather than silently returning a
        # circle that latched onto noise.
        limbus_score = _edge_contrast_score(gray, iris.x, iris.y, iris.r, limbus_thetas)
        if limbus_score < 3.0:
            raise SegmentationError(
                f"No reliable limbus edge found (edge contrast={limbus_score:.2f})"
            )

        # Second sanity check, after refinement: enforce the same
        # anatomical non-concentricity bound used when filtering candidates
        # in detect_iris. Refinement's own search window is small, but a
        # borderline-plausible starting candidate could still drift toward
        # an implausible final position; this is the last line of defense
        # against exactly the failure mode observed on real IITD data
        # (limbus center jumping ~36% of iris radius between two photos of
        # the same eye, consistent with locking onto an eye-corner or
        # eyelid-fold edge instead of the true iris-sclera boundary).
        center_offset = float(np.hypot(iris.x - pupil.x, iris.y - pupil.y))
        if center_offset > 0.25 * iris.r:
            raise SegmentationError(
                f"Implausible pupil/iris non-concentricity "
                f"({center_offset:.1f}px, {center_offset / iris.r:.0%} of iris radius)"
            )

    return pupil, iris, gray


# --------------------------------------------------------------------------- #
# 2. Rubber-sheet normalization (Daugman eqs. 10-12)
# --------------------------------------------------------------------------- #

def normalize_iris(
    gray: np.ndarray,
    pupil: Circle,
    iris: Circle,
    radial_res: int = 32,
    angular_res: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map the annular iris region to a fixed-size (radial_res x angular_res)
    rectangular image using the doubly-dimensionless polar coordinate
    (homogeneous rubber-sheet) model. Pupil and iris circles are allowed to
    be non-concentric, as in the paper.

    Uses bilinear interpolation (via cv2.remap) rather than nearest-
    neighbor sampling: with nearest-neighbor, a fraction-of-a-pixel
    difference in the segmented boundary between two photos of the same
    eye can flip which discrete source pixel gets sampled, injecting
    avoidable noise into exactly the comparison that needs to be most
    stable (genuine pairs). Bilinear sampling makes the normalized image
    vary smoothly as the boundary estimate varies slightly, instead of by
    discrete jumps.

    Returns
    -------
    normalized : float64 array, shape (radial_res, angular_res)
    valid_mask : uint8 array, same shape; 0 where the sample fell outside
                 the source image bounds.
    """
    theta = np.linspace(0, 2 * np.pi, angular_res, endpoint=False)

    xp = pupil.x + pupil.r * np.cos(theta)
    yp = pupil.y + pupil.r * np.sin(theta)
    xs = iris.x + iris.r * np.cos(theta)
    ys = iris.y + iris.r * np.sin(theta)

    return _rubber_sheet(gray, xp, yp, xs, ys, radial_res)


def _rubber_sheet(
    gray: np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    radial_res: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Core of eqs. (11)-(12): given the pupil boundary point (xp[j], yp[j])
    and limbus boundary point (xs[j], ys[j]) at each of `angular_res`
    angles, sample the image along the straight line between them at
    `radial_res` fractional radii. Shared by the circle-based and the
    boundary-point-based (e.g. external segmenter) entry points, so both
    produce normalized images by exactly the same procedure."""
    h, w = gray.shape
    angular_res = len(xp)
    r_vals = (np.arange(radial_res) + 0.5) / radial_res  # sample ring centers

    map_x = np.empty((radial_res, angular_res), dtype=np.float32)
    map_y = np.empty((radial_res, angular_res), dtype=np.float32)
    for i, r in enumerate(r_vals):
        map_x[i, :] = (1 - r) * xp + r * xs
        map_y[i, :] = (1 - r) * yp + r * ys

    inside = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)

    normalized = cv2.remap(
        gray.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(np.float64)
    valid_mask = inside.astype(np.uint8)

    return normalized, valid_mask


def resample_boundary(
    points,
    angular_res: int,
    max_harmonics: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a closed boundary, given as N points equally spaced in angle,
    onto the module's angular grid theta_j = 2*pi*j/angular_res.

    Point-ordering convention (verified on Neurotechnology MegaMatcher/
    VeriEye output): point 0 lies at the right-hand side (theta = 0) and the
    points run counter-clockwise *as seen on screen*, one every 360/N
    degrees -- i.e. in this module's y-down image convention, point k sits
    at theta_k = -2*pi*k/N.

    Interpolation is trigonometric (FFT zero-padding): it is exact for a
    band-limited closed curve, which circles and ellipses essentially are,
    and avoids the systematic inward bias of straight chords between
    sparse points (for r=100 and N=32 chords cut ~0.5 px inside the curve).
    `max_harmonics`, if given, additionally low-pass filters the contour
    (keeps harmonics 0..max_harmonics) to suppress the +-0.5 px integer
    rounding of the input points; the default (None) interpolates exactly.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise SegmentationError("Boundary must be an (N, 2) array of points")
    n = len(pts)
    if n < 8:
        raise SegmentationError(f"Boundary has too few points ({n})")

    # Index of the (fractional) source point for each output angle.
    # theta_j = 2*pi*j/angular_res  <->  k = -theta_j * N / (2*pi)  (mod N)
    out = []
    for coord in (pts[:, 0], pts[:, 1]):
        spec = np.fft.rfft(coord)                      # N//2 + 1 coefficients
        if max_harmonics is not None:
            spec[max_harmonics + 1:] = 0
        full = np.zeros(angular_res // 2 + 1, dtype=complex)
        m = min(len(spec), len(full))
        full[:m] = spec[:m]
        if n % 2 == 0 and n // 2 < len(full):
            full[n // 2] *= 0.5                        # split the Nyquist bin
        # irfft returns samples at parameter u_j = j/angular_res of the
        # *source* parameterization; scale amplitude for the longer length.
        resampled = np.fft.irfft(full, n=angular_res) * (angular_res / n)
        out.append(resampled)

    # `resampled[j]` is the curve at source parameter k = j * N/angular_res,
    # i.e. at theta = -2*pi*k/N. Re-index onto the theta_j grid: theta_j
    # corresponds to source parameter -j (mod angular_res).
    idx = (-np.arange(angular_res)) % angular_res
    return out[0][idx], out[1][idx]


# --------------------------------------------------------------------------- #
# 3. Noise masking (specular reflections, saturated pixels, eyelid zones,
#    and invalid samples)
# --------------------------------------------------------------------------- #

def _candidate_occlusion_columns(
    angular_res: int,
    top_band_deg: float = 90.0,
    bottom_band_deg: float = 60.0,
) -> np.ndarray:
    """Boolean array of length angular_res: True for columns where eyelid
    intrusion or specular reflection is *plausible* (near the top / bottom
    of the iris, in this module's convention where theta=0 is +x/right and
    theta=+90 is +y/down). This is only a candidate region to examine
    adaptively (see build_noise_mask) -- it is intentionally wider than
    the actual expected occlusion, since the real decision of which pixels
    to mask is made per-pixel from local image content, not by blanket
    zone exclusion. A blanket exclusion was tried and made results worse:
    it discards genuinely good iris texture in the (common) case where an
    eye is fairly open in a given photo, and shrinking every code's usable
    bit count increases the variance of every Hamming-distance estimate,
    widening both distributions rather than narrowing them."""
    theta_deg = np.degrees(np.linspace(0, 2 * np.pi, angular_res, endpoint=False))
    theta_deg = ((theta_deg + 180) % 360) - 180

    def angular_dist(a, b):
        return np.abs(((a - b + 180) % 360) - 180)

    top = angular_dist(theta_deg, -90.0) < (top_band_deg / 2)
    bottom = angular_dist(theta_deg, 90.0) < (bottom_band_deg / 2)
    return top | bottom


def _local_variance(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Fast local variance via box filters: Var(X) = E[X^2] - E[X]^2."""
    img = img.astype(np.float64)
    mean = cv2.boxFilter(img, ddepth=-1, ksize=(ksize, ksize))
    sqmean = cv2.boxFilter(img * img, ddepth=-1, ksize=(ksize, ksize))
    return np.clip(sqmean - mean ** 2, 0.0, None)


def build_noise_mask(
    normalized: np.ndarray,
    valid_mask: np.ndarray,
    low_thresh: float = 20.0,
    high_thresh: float = 235.0,
    top_band_deg: float = 90.0,
    bottom_band_deg: float = 60.0,
    variance_percentile: float = 25.0,
) -> np.ndarray:
    """Flag pixels likely corrupted by eyelash shadow (very dark), specular
    reflection (very bright), samples that fell outside the source image
    during normalization, or eyelid occlusion.

    Eyelid occlusion is detected *adaptively*, per image, rather than by
    excluding a fixed zone from every code: within a candidate region near
    the top/bottom of the iris (where eyelids can plausibly intrude), a
    pixel is masked only if its local texture variance is low relative to
    this image's own iris texture (eyelid skin is comparatively smooth;
    genuine iris texture -- crypts, furrows, the trabecular meshwork -- is
    not). This means an image with wide-open eyes keeps nearly all of its
    bits, while an image with genuine eyelid intrusion has only the
    actually-occluded pixels removed. The threshold is set per-image, as
    a percentile of that image's own variance distribution over valid
    pixels, so it adapts to each photo's contrast/focus/noise level rather
    than relying on one fixed constant across the whole dataset.
    """
    ok = valid_mask.astype(bool)
    ok &= normalized > low_thresh
    ok &= normalized < high_thresh

    variance = _local_variance(normalized)
    candidate = _candidate_occlusion_columns(normalized.shape[1], top_band_deg, bottom_band_deg)

    valid_var = variance[ok]
    if valid_var.size > 0:
        var_thresh = np.percentile(valid_var, variance_percentile)
        smooth = variance < var_thresh
        ok[candidate & smooth] = False

    return ok.astype(np.uint8)


# --------------------------------------------------------------------------- #
# 4. Encoding: quadrature 2D Gabor filtering -> 2048-bit code
# --------------------------------------------------------------------------- #

def _make_gabor_pair(ksize: int, sigma: float, wavelength: float, gamma: float = 1.0):
    """Quadrature pair of 2D Gabor kernels (cos / sin phase) applied in the
    (radial, angular) normalized-image domain, corresponding to the real
    2-D Gabor wavelet family of eq. (13). theta=0 orients the modulation
    along the angular axis, which carries the bulk of discriminative iris
    texture."""
    real = cv2.getGaborKernel((ksize, ksize), sigma, theta=0, lambd=wavelength,
                               gamma=gamma, psi=0, ktype=cv2.CV_64F)
    imag = cv2.getGaborKernel((ksize, ksize), sigma, theta=0, lambd=wavelength,
                               gamma=gamma, psi=-np.pi / 2, ktype=cv2.CV_64F)
    # Zero the DC component of each kernel (Daugman notes the real Gabor
    # filters are adjusted to have zero volume / no DC response so that
    # code bits do not depend on illumination strength).
    real -= real.mean()
    imag -= imag.mean()
    return real, imag


def _filter_and_sample(
    normalized: np.ndarray,
    noise_mask: np.ndarray,
    ksize: int,
    sigma: float,
    wavelength: float,
    n_rows: int,
    n_cols: int,
    radial_band: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one Gabor scale over the normalized image and sample it on an
    (n_rows x n_cols) grid, returning (bits, mask) blocks of shape
    (n_rows, n_cols, 2)."""
    rad_res, ang_res = normalized.shape
    real_k, imag_k = _make_gabor_pair(ksize, sigma, wavelength)

    pad = ksize // 2
    wrapped = np.pad(normalized, ((pad, pad), (0, 0)), mode="reflect")
    wrapped = np.pad(wrapped, ((0, 0), (pad, pad)), mode="wrap")

    resp_real = cv2.filter2D(wrapped, ddepth=cv2.CV_64F, kernel=real_k, borderType=cv2.BORDER_REFLECT)
    resp_imag = cv2.filter2D(wrapped, ddepth=cv2.CV_64F, kernel=imag_k, borderType=cv2.BORDER_REFLECT)
    resp_real = resp_real[pad:pad + rad_res, pad:pad + ang_res]
    resp_imag = resp_imag[pad:pad + rad_res, pad:pad + ang_res]

    lo, hi = radial_band
    row_idx = np.linspace(lo, hi, n_rows) * (rad_res - 1)
    row_idx = np.round(row_idx).astype(int)
    col_idx = np.linspace(0, ang_res, n_cols, endpoint=False)
    col_idx = np.round(col_idx).astype(int) % ang_res

    grid_r = resp_real[np.ix_(row_idx, col_idx)]
    grid_i = resp_imag[np.ix_(row_idx, col_idx)]
    grid_mask = noise_mask[np.ix_(row_idx, col_idx)]

    bits = np.empty((n_rows, n_cols, 2), dtype=np.uint8)
    bits[..., 0] = (grid_r >= 0).astype(np.uint8)
    bits[..., 1] = (grid_i >= 0).astype(np.uint8)

    mask = np.empty((n_rows, n_cols, 2), dtype=np.uint8)
    mask[..., 0] = grid_mask
    mask[..., 1] = grid_mask

    return bits, mask


# Default multi-scale filter bank: (ksize, sigma, wavelength) per scale.
# Fine scale resolves the dense trabecular-meshwork detail near the pupil;
# coarse scale resolves broader furrows/crypts and is more robust for
# irises where fine texture is low-contrast (common with lighter irises,
# softer focus, or more compression/noise). Encoding both, rather than a
# single fixed wavelength, is what the paper's own multi-scale Gabor
# family (eq. 7) is doing -- a single scale is a simplification that
# works but hits a real accuracy ceiling as population diversity grows.
DEFAULT_SCALES: Tuple[Tuple[int, float, float], ...] = (
    (17, 3.5, 7.0),   # fine: resolves trabecular detail without sub-Nyquist aliasing
    (27, 6.0, 14.0),  # coarse: resolves broader crypts/furrows
)


def encode_iris(
    normalized: np.ndarray,
    noise_mask: np.ndarray,
    scales: Tuple[Tuple[int, float, float], ...] = DEFAULT_SCALES,
    n_rows: int = 4,
    n_cols: int = 128,
    radial_band: Tuple[float, float] = (0.15, 0.85),
) -> IrisCode:
    """Encode a normalized iris image into a 2048-bit iris code using one
    or more Gabor scales.

    Each scale contributes an (n_rows x n_cols) sample grid -> 2 bits per
    sample. `len(scales) * n_rows * n_cols` must equal 1024 so the total
    code length is exactly 2048 bits (256 bytes), matching the paper.
    Scale blocks are stacked along the row axis, so all rows still share
    the same angular (column) indexing -- circular shifts for rotation
    tolerance (see `_shift_code`) remain valid across the whole code.
    """
    n_scales = len(scales)
    assert n_scales * n_rows * n_cols == 1024, (
        "len(scales) * n_rows * n_cols must equal 1024 for a 2048-bit code"
    )

    bit_blocks, mask_blocks = [], []
    for ksize, sigma, wavelength in scales:
        bits, mask = _filter_and_sample(
            normalized, noise_mask, ksize, sigma, wavelength, n_rows, n_cols, radial_band
        )
        bit_blocks.append(bits)
        mask_blocks.append(mask)

    bits = np.concatenate(bit_blocks, axis=0)   # (n_scales*n_rows, n_cols, 2)
    mask = np.concatenate(mask_blocks, axis=0)

    return IrisCode(
        bits=bits.reshape(-1),
        mask=mask.reshape(-1),
        grid_shape=(n_scales * n_rows, n_cols),
    )


# --------------------------------------------------------------------------- #
# 5. Matching: masked, rotation-tolerant fractional Hamming distance
# --------------------------------------------------------------------------- #

def _shift_code(code: IrisCode, shift: int) -> Tuple[np.ndarray, np.ndarray]:
    """Circularly shift a code along the angular axis by `shift` columns.
    Returns (bits, mask) reshaped back to flat 2048-length arrays."""
    n_rows, n_cols = code.grid_shape
    bits = code.bits.reshape(n_rows, n_cols, 2)
    mask = code.mask.reshape(n_rows, n_cols, 2)
    bits = np.roll(bits, shift, axis=1)
    mask = np.roll(mask, shift, axis=1)
    return bits.reshape(-1), mask.reshape(-1)


def hamming_distance(
    code1: IrisCode, code2: IrisCode, max_shift: int = 16, min_valid_bits: int = 300
) -> float:
    """Fractional Hamming distance (eq. 21) between two iris codes,
    excluding bits masked out in either code, minimized over a small
    range of angular shifts (best-of-n test, eqs. 33-34) to tolerate head
    tilt / cyclorotation.

    `min_valid_bits` guards against a statistically noisy estimate: with
    very few unmasked overlapping bits, the fractional Hamming distance
    has high variance and can land near 0 or 1 by chance even for
    genuinely independent codes (e.g. with only ~20 valid bits, a result
    as extreme as 0.13 is unremarkable; with 1000+ valid bits it is
    essentially impossible). Comparisons that never reach this many valid
    bits at any tested shift return NaN rather than a misleadingly
    confident number.

    Returns a value in [0, 1]; NaN if no shift ever has enough unmasked
    overlapping bits to trust.
    """
    assert code1.grid_shape == code2.grid_shape, "Codes must share the same grid shape"
    best_hd, best_n_valid = None, 0
    for shift in range(-max_shift, max_shift + 1):
        b2, m2 = _shift_code(code2, shift)
        combined_mask = (code1.mask.astype(bool)) & (m2.astype(bool))
        n_valid = combined_mask.sum()
        if n_valid < min_valid_bits:
            continue
        disagreements = np.logical_xor(code1.bits.astype(bool), b2.astype(bool)) & combined_mask
        hd = disagreements.sum() / n_valid
        if best_hd is None or hd < best_hd:
            best_hd, best_n_valid = hd, n_valid
    return float(best_hd) if best_hd is not None else float("nan")


def hamming_distance_debug(
    code1: IrisCode, code2: IrisCode, max_shift: int = 16, min_valid_bits: int = 300
):
    """Same as hamming_distance, but also returns the number of valid
    overlapping bits used at the winning shift, for diagnostics."""
    assert code1.grid_shape == code2.grid_shape
    best = None  # (hd, n_valid, shift)
    for shift in range(-max_shift, max_shift + 1):
        b2, m2 = _shift_code(code2, shift)
        combined_mask = (code1.mask.astype(bool)) & (m2.astype(bool))
        n_valid = int(combined_mask.sum())
        if n_valid < min_valid_bits:
            continue
        disagreements = np.logical_xor(code1.bits.astype(bool), b2.astype(bool)) & combined_mask
        hd = disagreements.sum() / n_valid
        if best is None or hd < best[0]:
            best = (hd, n_valid, shift)
    if best is None:
        return float("nan"), 0, None
    return float(best[0]), best[1], best[2]


# --------------------------------------------------------------------------- #
# Convenience: end-to-end
# --------------------------------------------------------------------------- #

def image_to_iris_code(
    image: np.ndarray,
    radial_res: int = 32,
    angular_res: int = 256,
    scales: Tuple[Tuple[int, float, float], ...] = DEFAULT_SCALES,
    n_rows: int = 4,
    n_cols: int = 128,
) -> IrisCode:
    """Full pipeline: raw image -> segmentation -> normalization ->
    noise mask -> 2048-bit IrisCode."""
    pupil, iris, gray = segment_iris(image)
    normalized, valid_mask = normalize_iris(gray, pupil, iris, radial_res, angular_res)
    noise_mask = build_noise_mask(normalized, valid_mask)
    return encode_iris(normalized, noise_mask, scales=scales, n_rows=n_rows, n_cols=n_cols)


# --------------------------------------------------------------------------- #
# External segmentation: use boundary points from another segmenter
# (e.g. Neurotechnology MegaMatcher / VeriEye) instead of our own circle fit.
# Everything downstream of segmentation -- CLAHE, rubber-sheet sampling,
# Gabor encoding, masking, matching -- is the same code path as above, so a
# comparison between the two isolates the effect of segmentation alone.
# --------------------------------------------------------------------------- #

def normalize_iris_from_boundaries(
    gray: np.ndarray,
    inner_points,
    outer_points,
    radial_res: int = 32,
    angular_res: int = 256,
    max_harmonics: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rubber-sheet normalization (eqs. 11-12) from explicit pupil (inner)
    and limbus (outer) boundary point sets rather than fitted circles. The
    boundaries need not be circular or concentric. See `resample_boundary`
    for the point-ordering convention."""
    xp, yp = resample_boundary(inner_points, angular_res, max_harmonics)
    xs, ys = resample_boundary(outer_points, angular_res, max_harmonics)

    # Sanity: the pupil contour must lie inside the limbus contour. Measured
    # as mean distance from each contour's own centroid, which is robust to
    # the outer contour extending past the image edge.
    r_in = np.hypot(xp - xp.mean(), yp - yp.mean()).mean()
    r_out = np.hypot(xs - xs.mean(), ys - ys.mean()).mean()
    if not (0 < r_in < r_out):
        raise SegmentationError(
            f"Inner boundary (r~{r_in:.1f}) is not inside outer boundary (r~{r_out:.1f})"
        )
    return _rubber_sheet(gray, xp, yp, xs, ys, radial_res)


def image_to_iris_code_from_boundaries(
    image: np.ndarray,
    inner_points,
    outer_points,
    radial_res: int = 32,
    angular_res: int = 256,
    scales: Tuple[Tuple[int, float, float], ...] = DEFAULT_SCALES,
    n_rows: int = 4,
    n_cols: int = 128,
    max_harmonics: Optional[int] = None,
) -> IrisCode:
    """Like `image_to_iris_code`, but segmentation is supplied: `inner_points`
    / `outer_points` are (N, 2) arrays of (x, y) pixel coordinates in the
    same pixel frame as `image`."""
    gray = preprocess_gray(image)
    normalized, valid_mask = normalize_iris_from_boundaries(
        gray, inner_points, outer_points, radial_res, angular_res, max_harmonics
    )
    noise_mask = build_noise_mask(normalized, valid_mask)
    return encode_iris(normalized, noise_mask, scales=scales, n_rows=n_rows, n_cols=n_cols)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python iris_encoder.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    code = image_to_iris_code(img)
    print(f"Iris code: {len(code.bits)} bits, {code.mask.sum()} usable "
          f"({code.mask.sum() / len(code.mask):.1%} unmasked)")
    print(f"Packed (hex, first 32 bytes): {code.packed()[:32].hex()}")
