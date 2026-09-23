"""
Unit tests for FFT-new (`fft_fxp`'s self-adjoint-decomposition forward FFT,
see `fxp/fft_fxp.py`'s "Forward FFT" section docstring).

Three layers, each catching a different class of bug:

  1. `_fft_selfadj_at` against the float64 reference, on self-adjoint inputs
     built the same way `_fft_new_at` builds its own a/b halves — pins the
     R[] table and the real-only recursion.
  2. `fft_fxp` (the public entry point) against `falcon_ref.fft.fft`, across
     both `certified` modes and a spread of n (including the two used in
     production, 512 and 1024) — pins the a+X^(n/2)b decomposition and the
     ±i sign pattern used to recombine â, b̂.
  3. Same, with `check_modulus()` enabled, so every FxC built along the way
     (including the internal `bodd` widen-and-reabsorb step) is checked
     against the STRICT |z| < 2^m invariant, not just the component-wise
     |x| < 2^p backstop — this is what actually certifies the "no widen"
     claims in the module docstring.
"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "falcon_ref"))
sys.path.insert(0, str(ROOT / "fxp"))

from fft import fft as fft_ref, adj as adj_ref            # noqa: E402

from fxtypes import FxR, retag_fxr, check_modulus          # noqa: E402
from fft_fxp import fft_fxp, _fft_selfadj_at                # noqa: E402


def _smallest_m(max_abs) -> int:
    m = 1
    while max_abs >= (1 << m):
        m += 1
    return m


def _to_fxr(f, p=63, m=None):
    max_abs = max((abs(int(x)) for x in f), default=0)
    m = m if m is not None else _smallest_m(max_abs)
    return [FxR.from_int(int(x), m=m, p=p) for x in f], m


def _max_abs_diff_complex(a, b):
    return max(abs(ca - cb) for ca, cb in zip(a, b))


# --------------------------------------------------------------------- #
# Layer 1: `_fft_selfadj_at` vs. the float64 reference.
# --------------------------------------------------------------------- #

def _random_selfadj_int(n, bound, rng):
    """Build an integral self-adjoint poly directly from Remark 1: pick the
    free half a[0..n/2-1] at random, force a[n/2]=0 and reflect the rest,
    a[n/2+j]=-a[n/2-j] — exactly representable, no rounding either way."""
    half = n // 2
    a = [0] * n
    a[0] = rng.randint(-bound, bound)
    for j in range(1, half):
        a[j] = rng.randint(-bound, bound)
        a[n - j] = -a[j]
    a[half] = 0
    return a


def test_fft_selfadj_matches_float_reference():
    """`_fft_selfadj_at(a, m)` requires m to bound ‖FFT(a)‖_∞, not just a's
    own coefficients (same structural contract as `fft_fxp`'s uncertified
    mode: m = m_coef + log2(n), since ‖FFT‖ ≤ (n/√2)·2^m_coef)."""
    rng = random.Random(0)
    for n in (4, 8, 16, 32, 64, 128, 256, 512):
        for _ in range(10):
            a_int = _random_selfadj_int(n, 1 << 20, rng)
            a_fxr, m_coef = _to_fxr(a_int)
            m = m_coef + (n.bit_length() - 1)
            a_fxr = [retag_fxr(c, m) for c in a_fxr]
            got = [z.to_float() for z in _fft_selfadj_at(a_fxr, m)]
            ref = fft_ref([float(c) for c in a_int])
            assert max(abs(r.imag) for r in ref) < 1e-6, \
                f"n={n}: reference FFT of a self-adjoint input has nonzero Im"
            err = max(abs(g - r.real) for g, r in zip(got, ref))
            tol = 1e-6 * max(1.0, max(abs(c) for c in a_int))
            assert err < tol, f"n={n}: selfadj err={err:.3e} tol={tol:.3e}"


# --------------------------------------------------------------------- #
# Layer 2: `fft_fxp` vs. the float64 reference (both `certified` modes).
# --------------------------------------------------------------------- #

def test_fft_fxp_matches_float_reference_uncertified():
    rng = random.Random(1)
    for n in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
        for _ in range(15):
            f_int = [rng.randint(-(1 << 20), 1 << 20) for _ in range(n)]
            f_fxr, _ = _to_fxr(f_int)
            got = [z.to_complex() for z in fft_fxp(f_fxr, certified=False)]
            ref = fft_ref([float(c) for c in f_int])
            err = _max_abs_diff_complex(got, ref)
            tol = 1e-6 * max(1.0, max(abs(c) for c in f_int))
            assert err < tol, f"n={n}: err={err:.3e} tol={tol:.3e}"


def test_fft_fxp_matches_float_reference_certified():
    """`certified=True`: caller loads coefficients at a tag it has already
    PROVEN bounds ‖FFT(f)‖ — exercised the way target_construction.py and
    sign_tweak.py actually call it (fixed m, no per-call retag). The bound
    a real caller relies on (e.g. an NTRUGen γ filter) is tighter than the
    generic structural one; here we just use that same structural bound
    (m = m_coef + log2(n)) directly as the certified tag, which trivially
    satisfies the certified contract (‖FFT‖ ≤ (n/√2)·2^m_coef < 2^m)."""
    rng = random.Random(2)
    for n in (8, 64, 512):
        m_coef = 3
        m = m_coef + (n.bit_length() - 1)
        for _ in range(15):
            bound = (1 << m_coef) - 1
            f_int = [rng.randint(-bound, bound) for _ in range(n)]
            f_fxr = [retag_fxr(FxR.from_int(c, m=m_coef, p=63), m) for c in f_int]
            got = [z.to_complex() for z in fft_fxp(f_fxr, certified=True)]
            ref = fft_ref([float(c) for c in f_int])
            err = _max_abs_diff_complex(got, ref)
            tol = 1e-8 * max(1.0, max(abs(c) for c in f_int))
            assert err < tol, f"n={n}: err={err:.3e} tol={tol:.3e}"


def test_fft_fxp_edge_polys():
    """Zero, single-spike, and alternating-sign polys (worst case for the
    a+X^(n/2)b decomposition's cancellation)."""
    for n in (4, 8, 64, 512):
        cases = [
            [0] * n,
            [1] + [0] * (n - 1),
            [(-1) ** i for i in range(n)],
            [(1 << 18) * (-1 if i % 3 == 0 else 1) for i in range(n)],
        ]
        for f_int in cases:
            f_fxr, _ = _to_fxr(f_int)
            got = [z.to_complex() for z in fft_fxp(f_fxr)]
            ref = fft_ref([float(c) for c in f_int])
            err = _max_abs_diff_complex(got, ref)
            tol = 1e-6 * max(1.0, max(abs(c) for c in f_int))
            assert err < tol, f"n={n}: edge case err={err:.3e} tol={tol:.3e}"


def test_fft_fxp_adjoint_consistency():
    """FFT(adj(f))[j] == conj(FFT(f)[j]) — an independent algebraic check
    (not derived the same way as the a/b decomposition) that would catch a
    sign error in the ±i recombination that the reference-comparison tests
    above might not exercise as sharply."""
    rng = random.Random(3)
    for n in (8, 64, 512):
        for _ in range(5):
            f_int = [rng.randint(-(1 << 16), 1 << 16) for _ in range(n)]
            fstar_int = adj_ref([float(c) for c in f_int])
            f_fxr, m = _to_fxr(f_int)
            fstar_fxr = [FxR.from_int(round(c), m=m, p=63) for c in fstar_int]
            got_f = [z.to_complex() for z in fft_fxp(f_fxr)]
            got_fstar = [z.to_complex() for z in fft_fxp(fstar_fxr)]
            err = max(abs(a - b.conjugate()) for a, b in zip(got_fstar, got_f))
            tol = 1e-6 * max(1.0, max(abs(c) for c in f_int))
            assert err < tol, f"n={n}: adj/conj mismatch err={err:.3e} tol={tol:.3e}"


# --------------------------------------------------------------------- #
# Layer 3: strict |z| < 2^m modulus check.
# --------------------------------------------------------------------- #

def test_fft_fxp_modulus_strict():
    """Every FxC built by fft_fxp (including the transient `bodd` branch's
    reabsorption) satisfies |z| < 2^m exactly, not just componentwise."""
    rng = random.Random(4)
    with check_modulus(True):
        for n in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
            for _ in range(15):
                f_int = [rng.randint(-(1 << 18), 1 << 18) for _ in range(n)]
                f_fxr, _ = _to_fxr(f_int)
                fft_fxp(f_fxr, certified=False)   # raises AssertionError on violation


# --------------------------------------------------------------------- #
# Standalone runner (mirrors test_fxtypes.py / test_nr_fxp.py style)
# --------------------------------------------------------------------- #

ALL_TESTS = [
    test_fft_selfadj_matches_float_reference,
    test_fft_fxp_matches_float_reference_uncertified,
    test_fft_fxp_matches_float_reference_certified,
    test_fft_fxp_edge_polys,
    test_fft_fxp_adjoint_consistency,
    test_fft_fxp_modulus_strict,
]


def main() -> int:
    for t in ALL_TESTS:
        t()
        print(f"{t.__name__}: PASS")
    print(f"\nAll {len(ALL_TESTS)} fft_new tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
