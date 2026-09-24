"""The verification numeric contract: comparing against an asset stored as float32.

Why this file exists
--------------------
Two earlier verification rounds on RPE1 failed, and neither failure was a defect in
the build.

Round one compared a float32 accumulation against the build's float64 accumulation
and failed at 2.4e-06.  That measured numpy's accumulation dtype, not agreement.

Round two accumulated in float64 on both sides and still failed, at 4.77e-07 and
1.19e-07 against a 1e-08 bound.  Those residuals are exactly one float32 unit in the
last place at the magnitudes involved.  The derivation behind 1e-08 reasoned about
float64 summation order and forgot that the product is *stored* as float32: two
float64 values differing by 1e-12 can straddle a float32 rounding boundary and come
out a whole ULP apart.  No verifier can resolve below the precision of the artifact
it is checking.

The contract
------------
The independent recomputation is done in float64 from the original cell data, then
cast to the dtype the asset is actually stored in, and only then compared:

    |x_stored - x_expected32| <= 2 * eps32 * max(|x_expected32|, 1)

The ``max(., 1)`` floor is load bearing.  A response is a difference of two
quantities of order one, so where they nearly cancel the result is tiny while its
absolute error is still inherited from the operands; a purely relative or purely
ULP-based bound would demand a precision that the subtraction itself destroyed.  The
floor makes the bound absolute at 2.4e-07 there and relative above magnitude one.

This bound is not a tolerance chosen to fit an observed residual, and it is not
adjusted again.  It is two ULP of the storage format.  A real defect, a wrong control
batch or a wrong incidence weight, moves a response by 1e-02 to 1e+00, four to six
orders of magnitude above this bound; ``corrupt_for_witness`` and the protocol tests
demonstrate that rather than asserting it.
"""
from __future__ import annotations

import numpy as np


EPS32 = float(np.finfo(np.float32).eps)
TOLERANCE_ULPS = 2.0
TOLERANCE_EXPRESSION = "abs(stored - expected32) <= 2 * eps32 * max(abs(expected32), 1)"
CONTRACT = "RESPONSE_ASSET_VERIFIER_CONTRACT_V1_FLOAT32_STORAGE_AWARE"


def _monotonic_key(values: np.ndarray) -> np.ndarray:
    """Order-preserving integer key for float32, so ULP distance is a subtraction.

    Negative floats are reflected so the mapping is monotonic across zero, and the
    two zeros map to the same key.
    """
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.int64)
    return np.where(bits >= 0x80000000, np.int64(0x80000000) - bits, bits)


def ulp_distance(stored: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """How many representable float32 values lie between the two, elementwise."""
    return np.abs(_monotonic_key(stored) - _monotonic_key(np.asarray(expected, dtype=np.float32)))


def compare(stored: np.ndarray, expected: np.ndarray, label: str) -> dict:
    """The frozen gate.  ``expected`` may be float64; it is cast before comparison."""
    stored = np.ascontiguousarray(stored, dtype=np.float32)
    expected32 = np.ascontiguousarray(expected, dtype=np.float32)
    if stored.shape != expected32.shape:
        raise ValueError(f"{label}: shapes differ, {stored.shape} against {expected32.shape}")
    if not (np.isfinite(stored).all() and np.isfinite(expected32).all()):
        return {"label": label, "passed": False, "reason": "a compared value is not finite"}
    difference = np.abs(stored.astype(np.float64) - expected32.astype(np.float64))
    bound = TOLERANCE_ULPS * EPS32 * np.maximum(np.abs(expected32.astype(np.float64)), 1.0)
    worst = int(np.argmax(difference - bound)) if difference.size else 0
    flat_stored, flat_expected = stored.reshape(-1), expected32.reshape(-1)
    # ULP distance is informative only where float32 spacing is the limiting factor,
    # which is where the value exceeds one; below that the floor in the bound binds
    # instead, and a large ULP distance there is subtractive cancellation rather than
    # disagreement.  The RPE1 round-three record shows 359 ULP at values near 1e-03
    # with an absolute error of 1.2e-07, which is exactly that situation.
    resolved = np.abs(flat_expected) >= 1.0
    ulps = ulp_distance(flat_stored[resolved], flat_expected[resolved]) if resolved.any() \
        else np.zeros(0, dtype=np.int64)
    return {
        "label": label,
        "passed": bool((difference <= bound).all()),
        "elements": int(stored.size),
        "max_abs_error": float(difference.max()) if difference.size else 0.0,
        "bound_at_worst_element": float(bound.reshape(-1)[worst]) if difference.size else 0.0,
        "value_at_worst_element": float(flat_expected[worst]) if difference.size else 0.0,
        "violations": int((difference > bound).sum()),
        "max_ratio_to_bound": float((difference / bound).max()) if difference.size else 0.0,
        "max_ulp_distance_where_ulp_binds": int(ulps.max()) if ulps.size else 0,
        "elements_where_ulp_binds": int(resolved.sum()),
        "tolerance": TOLERANCE_EXPRESSION,
    }


def corrupt_for_witness(values: np.ndarray, magnitude: float = 1e-3,
                        seed: int = 20260918) -> np.ndarray:
    """Introduce one realistic magnitude error, to witness that the gate still fires.

    The perturbation is the size of a genuine pipeline defect, not of a rounding
    difference: a response moved by 1e-03 is four orders of magnitude above the
    storage bound and two to three below a typical response.

    The caller is responsible for handing in a block the verifier will actually
    inspect.  Round three of the RPE1 verification failed precisely because it did
    not: a single element was corrupted anywhere in a 1483-identity asset while the
    verifier samples twelve identities, so the witness missed its target about 99
    times in 100 and reported that a corrupted asset had passed.  That was a defect
    in the witness, and it is the reason the witness is itself gated.
    """
    generator = np.random.default_rng(seed)
    # ascontiguousarray, not np.array: a fancy-indexed block such as
    # ``responses[:, sampled]`` is not C-contiguous, ``np.array(..., order="K")``
    # preserves that layout, and ``reshape(-1)`` on it returns a copy rather than a
    # view.  Writing through that copy left the returned array untouched, so round
    # four of the RPE1 verification reported that a corrupted asset had passed while
    # nothing had in fact been corrupted.  The post-condition below is the reason a
    # repeat of that failure is impossible rather than unlikely.
    corrupted = np.ascontiguousarray(values, dtype=np.float32).copy()
    flat = corrupted.reshape(-1)
    index = int(generator.integers(0, flat.size))
    before = float(flat[index])
    flat[index] = np.float32(before + magnitude)
    changed = int((np.asarray(corrupted, dtype=np.float32)
                   != np.ascontiguousarray(values, dtype=np.float32)).sum())
    if changed != 1:
        raise RuntimeError(
            f"the corruption changed {changed} elements rather than one; a witness that "
            "does not corrupt its target verifies nothing")
    return corrupted
