"""
Fixed-point FFT over R[x]/(x^n+1), on FxR/FxC values.

Twiddle constants live at (m=1, p), selected by input's p (63 or 127).

The forward FFT (`fft_fxp`) runs FFT-new (see the "Forward FFT" section
below): it decomposes the input into two self-adjoint halves and FFTs each
with real-only butterflies, at a SINGLE tag m resolved once at entry (no
per-level growth in the OUTPUT tag — the self-adjoint recursion's transient
"odd" branch runs one bit wider internally but is reabsorbed every level;
see `_fft_selfadj_at`). Correctness rests on the same averaging bound as
before: a size-N partial transform of f is a value of some sub-FFT, so
‖·‖_∞ ≤ ‖FFT(f)‖_∞ — hence if the OUTPUT fits in 2^m, every intermediate
does too. `fft_fxp`'s `certified` flag says where that m comes from — NOT
whether it is fixed (it always is):
  - certified=True  : the caller's load tag already bounds the output (the B0
                      rows, at their γ tags). m_out = m_in; no retag at all.
  - certified=False : no such bound, so use the structural one, m_in + log₂n
                      (‖FFT‖ ≤ (n/√2)·2^{m_in}). One retag of the inputs.

The inverse FFT is separate and unchanged (classic recursion via
`split_complex_fxp`/`merge_fft_fxp`, still used internally by ffLDL/
ffsampling too — see the "Forward FFT" section's docstring for why FFT-new
was scoped to the forward direction only): `split_complex_fxp` preserves m,
one rounding per half — both are formed on the exact integer mantissas and
shifted once (the ÷2 and the twiddle mul fold into that shift), so
`ifft_fxp` is m-preserving and retag-free throughout.

The only `retag` in this module is `fft_fxp`'s uncertified input retag.
"""

from fractions import Fraction

from beartype import beartype

from fxtypes import FxR, FxC, PolyR, PolyC, retag_fxr, _bankers_shift
from fxp_constants_p63 import roots_dict_fxp as _roots_p63
from fxp_constants_p127 import roots_dict_fxp as _roots_p127
from nr_fxp import nr_reciprocal


_ROOTS_BY_P = {63: _roots_p63, 127: _roots_p127}


def _roots_for(p: int):
    """Select the precomputed twiddle table matching precision p."""
    try:
        return _ROOTS_BY_P[p]
    except KeyError:
        raise ValueError(
            f"no FFT constants for p={p}. Available: {sorted(_ROOTS_BY_P)}. "
            f"Regenerate via: sage scripts/generate_constants_fxp.sage {p} 1"
        )


# --------------------------------------------------------------------- #
# Forward FFT — FFT-new (self-adjoint decomposition)
#
# From "Toward a Fixed-Point..." [eprint 2026/1915], §3.2-3.3: any c in
# R[x]/(x^n+1) decomposes as c = a + x^{n/2}·b with a, b self-adjoint
# (a=(c*+c)/2, b=(c*-c)*x^{n/2}/2, both exact integer/fixed-point
# combinations of c's own coefficients — algorithm FFT-new). For a
# self-adjoint polynomial, a(ζ) is REAL at every root ζ (algorithm
# FFT-selfadj-new): splitting a into even/odd coefficients and writing
# b_odd = (1+x)·a_odd (also self-adjoint) turns each butterfly's ONE
# complex multiply (4 real integer muls, via `FxC.mul_to`) into ONE real
# multiply (`FxR.mul_to`) + a real add/sub — a 4x cut per butterfly.
#
# This port keeps the file's existing "redundant, length-n" PolyC/PolyR
# convention (every FFT-domain poly has the SAME length as its coefficient
# form) rather than the paper's packed half-size real/imaginary layout, so
# `merge_fft_fxp` / `split_complex_fxp` and every downstream consumer
# (ffldl_fxp, ffsampling_fxp, target_construction, sign_tweak) need no
# change at all. The cost: `_fft_selfadj_at`'s own recursion is ALSO
# "redundant, full-length" (it does not exploit that a self-adjoint FFT
# only carries n/2 independent values the way the paper's packed layout
# does), so it runs 2x the butterflies the paper's FFT-selfadj-new would.
# Net effect, measured on Falcon-512 (n=512): the OLD classic recursion
# below did 2048 `FxC.mul_to` calls (= 8192 raw integer multiplies) per
# forward FFT; FFT-new does 4096 `FxR.mul_to` calls (= 4096 raw integer
# multiplies) — a real 2x cut in the dominant arithmetic cost, half of the
# paper's ~4x, in exchange for touching only this file. `_fft_selfadj_at`
# is defined, by construction, to equal `[z.re for z in <the old recursive
# complex FFT>(a, m)]` position-for-position (both directions cross-checked
# against the float64 reference in tests/test_fft_new.py) — a pure
# rearrangement of the SAME computation, not a different one.
#
# Forward direction only: `ifft_fxp` stays on the classic split/merge
# recursion below. The paper's own operation count has 5 forward FFTs
# against 2 inverse FFTs per signature, so this still captures the
# majority of the win; inverting the self-adjoint recursion needs an O(n)
# wrap-around (the (1+x)-multiply's inverse is a circular recurrence) that
# does not fit this codebase's single-rounding-per-op model as cleanly, so
# it was left out of this "contained" pass — see fxp/README.md.
# --------------------------------------------------------------------- #


_SELFADJ_R_CACHE: dict[tuple[int, int], PolyR] = {}   # (p, n) -> R[], length n//2


def _round_fraction(fr: Fraction) -> int:
    """Round a Fraction to the nearest integer, ties-to-even."""
    q, r = divmod(fr.numerator, fr.denominator)   # fr.denominator > 0; r in [0, den)
    d = fr.denominator
    if 2 * r > d or (2 * r == d and q % 2 == 1):
        q += 1
    return q


def _selfadj_R_for(p: int, n: int) -> PolyR:
    """R[i] = 1/(2·Re(w[2i])) for i = 0..n/2-1, w = _roots_for(p)[n].

    This is exactly the paper's R[] table (§3.2: ζ/(1+ζ²) = 1/(2cosθ) for
    ζ=e^{iθ}), but derived from the EXISTING root-of-unity constants
    (`_roots_for`) instead of a new generated table: no new sage output,
    no float — pure exact-rational arithmetic (`fractions.Fraction`) on
    the already-precise integer mantissas, run once and cached. |R[i]|
    grows with n (≈ n/π at the top level: unbounded by the |root|=1 bound
    that every OTHER constant in this file has), so each level gets its
    own tight m (unlike `_roots_for`'s single shared m_fft).
    """
    key = (p, n)
    cached = _SELFADJ_R_CACHE.get(key)
    if cached is not None:
        return cached
    w = _roots_for(p)[n]
    half = n // 2
    exact = []
    for i in range(half):
        zr = w[2 * i].re
        # value(zr) = zr.x * 2^{zr.m - p}; R = 1/(2*value) = 2^{p-zr.m-1}/zr.x.
        exact.append(Fraction(1 << (p - zr.m - 1), zr.x))
    m_r = 0
    max_abs = max(abs(r) for r in exact)
    while max_abs >= (1 << m_r):
        m_r += 1
    table = [FxR(x=_round_fraction(r * (1 << (p - m_r))), m=m_r, p=p) for r in exact]
    _SELFADJ_R_CACHE[key] = table
    return table


def _mul_1_plus_x_selfadj(a_odd: PolyR, m: int) -> PolyR:
    """b = (1+x)*a_odd mod x^k+1 (k=len(a_odd)): b[0]=a_odd[0]-a_odd[k-1],
    b[i]=a_odd[i]+a_odd[i-1] for i=1..k-1 — self-adjoint iff a_odd is
    X-adjoint (Remark 2/3). Each output is the EXACT raw-mantissa sum
    (both operands share tag m) banker's-shifted ONCE to land at m+1 —
    same one-rounding pattern as `split_complex_fxp`'s (A±B)/2, not two.
    `b` can reach 2·2^m (|1+X| ≤ 2 on the unit circle), hence the +1 bit;
    reabsorbed by the very next `mul_to` in `_fft_selfadj_at`."""
    k = len(a_odd)
    p = a_odd[0].p
    xs = [c.x for c in a_odd]
    out = [FxR(x=_bankers_shift(xs[0] - xs[-1], 1), m=m + 1, p=p)]
    out.extend(FxR(x=_bankers_shift(xs[i] + xs[i - 1], 1), m=m + 1, p=p)
               for i in range(1, k))
    return out


def _fft_selfadj_at(a: PolyR, m: int) -> PolyR:
    """FFT of a self-adjoint polynomial, real-only, single-tag m for the
    OUTPUT (see module docstring). `a` is the full-length redundant
    coefficient array (a[n/2]=0, a[n/2+j]=-a[n/2-j] — Remark 1); the
    transient `bodd` branch runs one bit wider (m+1) to hold the |1+X|≤2
    growth, reabsorbed by `Rtab[i].mul_to(..., m)` before the final add,
    so the recursion returns to m at every level (no compounding growth).
    """
    n = len(a)
    if n == 2:
        # Self-adjoint degree-2: a[1] is the forced zero (Remark 1); the
        # single free coefficient a[0] IS its own (real) FFT value.
        return [a[0], a[0]]
    p = a[0].p
    half = n // 2
    aeven_hat = _fft_selfadj_at(a[0::2], m)
    bodd = _mul_1_plus_x_selfadj(a[1::2], m)
    bodd_hat = _fft_selfadj_at(bodd, m + 1)
    r_tab = _selfadj_R_for(p, n)
    out = [None] * n
    for i in range(half):
        z = r_tab[i].mul_to(bodd_hat[i], m)       # fused multiply + retag to m
        out[2 * i] = aeven_hat[i] + z
        out[2 * i + 1] = aeven_hat[i] - z
    return out


def _adj_coefficient(f: PolyR) -> PolyR:
    """Coefficient-domain Hermitian adjoint: f*[0]=f[0], f*[i]=-f[n-i] for
    i=1..n-1 (exact: reindex + negate, same tag as f)."""
    n = len(f)
    return [f[0]] + [-f[n - i] for i in range(1, n)]


def _fft_new_at(f: PolyR, m: int) -> PolyC:
    """FFT-new: decompose f = a + x^{n/2}·b into self-adjoint a, b (both at
    the SAME tag m as f — |a(ζ)|, |b(ζ)| ≤ |f(ζ)| pointwise, since a, b are
    the real/imaginary parts of f's own FFT value, so no growth here
    either), FFT each with `_fft_selfadj_at`, and recombine. Empirically
    (cross-checked in tests/test_fft_new.py): ĉ[j] = â[j] + i·b̂[j] for
    j < n/2, and ĉ[j] = â[j] − i·b̂[j] for j >= n/2 — matching this file's
    own (Falcon-standard even/odd split/merge) indexing convention.
    """
    n = len(f)
    p = f[0].p
    half = n // 2
    fstar = _adj_coefficient(f)
    xs_f = [c.x for c in f]
    xs_fstar = [c.x for c in fstar]
    # a=(f+f*)/2, b_pre=(f*-f)/2: exact raw-mantissa combine, ONE rounding.
    a = [FxR(x=_bankers_shift(xs_f[i] + xs_fstar[i], 1), m=m, p=p) for i in range(n)]
    b_pre = [FxR(x=_bankers_shift(xs_fstar[i] - xs_f[i], 1), m=m, p=p) for i in range(n)]
    # b = x^{n/2} * b_pre mod x^n+1: exact rotate-with-sign, no rounding.
    b = [None] * n
    for i in range(n):
        j = i + half
        if j < n:
            b[j] = b_pre[i]
        else:
            b[j - n] = -b_pre[i]
    a_hat = _fft_selfadj_at(a, m)
    b_hat = _fft_selfadj_at(b, m)
    out = [None] * n
    for j in range(half):
        out[j] = FxC(re=a_hat[j], im=b_hat[j])
        out[j + half] = FxC(re=a_hat[j + half], im=-b_hat[j + half])
    return out


@beartype
def fft_fxp(f: PolyR, certified: bool = False) -> PolyC:
    """FFT of a real polynomial in R[x]/(x^n+1), run at a SINGLE tag m.

    `certified` states what the CALLER knows about ‖FFT(f)‖_∞, which is all
    that decides m (the transform is single-tag either way):

      certified=True  — the caller guarantees ‖FFT(f)‖_∞ < 2^{m(f)}, so load
        the coefficients at that tag and the transform stays there: m_out =
        m_in, and NOT ONE retag occurs. This is what an NTRUGen check buys:
        the B0 rows load at their γ tags (M_B_FG=8, M_B_FG_UP=12) instead of
        the structural 14 / 16, which tightens every internal rounding.

      certified=False — no such bound, so fall back on the structural one,
        m_out = m(f) + log₂n (since ‖FFT‖ ≤ (n/√2)·2^{m(f)}). The inputs are
        retagged once, up front, to that tag. Used for the hash-derived c/q
        and qt, which no check bounds.

    Validity in both cases rests on ‖sub-FFT‖ ≤ ‖FFT‖: if the OUTPUT fits in
    2^m, every intermediate does. Integer coefficients retag exactly; only a
    fractional input (c/q) loses bits there, far below the pipeline's needs.

    Runs FFT-new (`_fft_new_at`, see module docstring above) for n >= 4; the
    n = 2 base case is direct (f0 ± i·f1 IS the FFT, no decomposition to be
    had).
    """
    n = len(f)
    assert n >= 2 and (n & (n - 1)) == 0, "n must be a power of 2"
    if certified:
        m = f[0].m
    else:
        m = f[0].m + (n.bit_length() - 1)            # m(f) + log₂n
        f = [retag_fxr(c, m) for c in f]             # once, here — not per level
    if n == 2:
        # f_fft = [f0 + i·f1, f0 − i·f1]: the two reals become the orthogonal
        # components of one FxC. Modulus √2·max ≤ ‖FFT‖ < 2^m, so no widen.
        return [FxC(re=f[0], im=f[1]), FxC(re=f[0], im=-f[1])]
    return _fft_new_at(f, m)


@beartype
def merge_fft_fxp(f_list_fft: list[PolyC], m: int) -> PolyC:
    """Combine two length-n/2 FFTs (both already at m) into one length-n at m.

    Inverse of `split_complex_fxp`. |w|=1 so the twiddle mul keeps the
    modulus; the butterfly output is a value of the (sub-)FFT, ≤ ‖FFT‖ < 2^m
    by the averaging bound, so `f0 ± w_f1` fits at m with no widen. One
    rounding per butterfly (the mul emitting at m), the add is exact.
    """
    f0_fft, f1_fft = f_list_fft
    n = 2 * len(f0_fft)
    w = _roots_for(f0_fft[0].p)[n]
    out = [None] * n
    for i in range(n // 2):
        w_f1 = w[2 * i].mul_to(f1_fft[i], m)         # emit at m (|w|=1)
        f0 = f0_fft[i]                                        # already at m
        out[2 * i] = f0 + w_f1
        out[2 * i + 1] = f0 - w_f1
    return out


# --------------------------------------------------------------------- #
# Inverse FFT
# --------------------------------------------------------------------- #


@beartype
def ifft_fxp(f_fft: PolyC) -> PolyR:
    """Inverse FFT, returning a list of FxR values. For a real-input FFT,
    f_fft = [a + i·b, a − i·b] at n=2, so (a, b) = (Re, Im) of f_fft[0]."""
    n = len(f_fft)
    assert n >= 2 and (n & (n - 1)) == 0, "n must be a power of 2"
    if n == 2:
        return [f_fft[0].re, f_fft[0].im]
    f0_fft, f1_fft = split_complex_fxp(f_fft)
    f0 = ifft_fxp(f0_fft)
    f1 = ifft_fxp(f1_fft)
    # `split_complex_fxp` retags f1 back to m, so f0 and f1 share m at every
    # recursion depth → straight interleave, no alignment retag.
    out = [None] * n
    for i in range(n // 2):
        out[2 * i] = f0[i]
        out[2 * i + 1] = f1[i]
    return out


# --------------------------------------------------------------------- #
# Element-wise polynomial helpers (FFT domain)
# --------------------------------------------------------------------- #


# Coefficient-wise FFT-domain ops (operands share p; add/sub also share m).

@beartype
def add_fft_fxp(f: PolyC, g: PolyC) -> PolyC:
    return [a + b for a, b in zip(f, g)]

@beartype
def sub_fft_fxp(f: PolyC, g: PolyC) -> PolyC:
    return [a - b for a, b in zip(f, g)]

@beartype
def mul_fft_fxp(f: PolyC, g: PolyC) -> PolyC:
    """Pointwise multiply at the natural (tight) tag m_a+m_b."""
    return [a * b for a, b in zip(f, g)]

@beartype
def mul_fft_to(f: PolyC, g: PolyC, m_out: int) -> PolyC:
    """Pointwise FxC multiply emitting directly at the budget m_out (fused
    multiply-and-retag, single round; see `FxC.mul_to`) — one rounding rather
    than a separate multiply then retag."""
    return [a.mul_to(b, m_out) for a, b in zip(f, g)]

@beartype
def adj_fft_fxp(f: PolyC) -> PolyC:
    """Complex conjugate (= FFT-domain adjoint of a real poly)."""
    return [z.conjugate() for z in f]


@beartype
def div_fft_fxp(f: PolyC, g: PolyR, m_out: int) -> PolyC:
    """Pointwise FxC ÷ real division: each f[i] divided by g[i] via the
    Newton-Raphson reciprocal (`nr_reciprocal`) then a multiply. The divisor
    is a Gram diagonal — real in FFT domain, enforced by the `PolyR` type.
    m_out depends on a lower bound for |g[i]|."""
    assert len(f) == len(g)
    out = []
    for a, b in zip(f, g):
        r = nr_reciprocal(b)                         # 1/g[i]
        out.append(FxC(re=a.re.mul_to(r, m_out),
                       im=a.im.mul_to(r, m_out)))
    return out


@beartype
def split_complex_fxp(f_fft: PolyC) -> tuple[PolyC, PolyC]:
    """Split a length-n **complex** FFT into halves (sign / ifft):
        f0[i] = 0.5·(f[2i] + f[2i+1])
        f1[i] = 0.5·(f[2i] − f[2i+1])·conj(w[2i])
    Both halves are returned at (m, p). The mul by conj(w) widens to m+1
    transiently; we banker-shift f1 back to m so downstream sees uniform m.
    """
    n = len(f_fft)
    w = _roots_for(f_fft[0].p)[n]
    m, p = f_fft[0].m, f_fft[0].p
    m_w = w[0].m
    f0 = [None] * (n // 2)
    f1 = [None] * (n // 2)

    # Both halves: exact integer mantissas, rounded once. |x_A ± x_B| reaches
    # 2^{p+1} transiently, hence no intermediate FxR is built.
    for i in range(n // 2):
        A, B = f_fft[2 * i], f_fft[2 * i + 1]
        # f0 = (A+B)/2 at m: one banker's shift of the exact sum.
        f0[i] = FxC(re=FxR(x=_bankers_shift(A.re.x + B.re.x, 1), m=m, p=p),
                    im=FxR(x=_bankers_shift(A.im.x + B.im.x, 1), m=m, p=p))
        # f1 = (A−B)/2·conj(w) at m: the ÷2 folds into the product's shift.
        d_re, d_im = A.re.x - B.re.x, A.im.x - B.im.x
        wc = w[2 * i]
        u_re, u_im = wc.re.x, -wc.im.x                 # conj(w)
        s = p + 1 - m_w
        f1[i] = FxC(re=FxR(x=_bankers_shift(d_re * u_re - d_im * u_im, s), m=m, p=p),
                    im=FxR(x=_bankers_shift(d_re * u_im + d_im * u_re, s), m=m, p=p))
    return f0, f1


@beartype
def split_real_fxp(f_fft: PolyR) -> tuple[PolyR, PolyC]:
    """Split a length-n **real** FFT (a Hermitian Gram diagonal, keygen path):
        f0[i] = 0.5·(f[2i] + f[2i+1])               → real  (PolyR)
        f1[i] = 0.5·(f[2i] − f[2i+1])·conj(w[2i])   → complex (PolyC)
    f0 stays real (the child diagonal); f1 picks up the twiddle and is complex.

    Implemented as `split_complex_fxp` on the FxC embedding (im = 0): the two
    are bit-identical (f0's imaginary part is exactly 0, so `.re` is lossless),
    while this entry keeps the typed PolyR → (PolyR, PolyC) contract.
    """
    zero = FxR(x=0, m=f_fft[0].m, p=f_fft[0].p)
    f0, f1 = split_complex_fxp([FxC(re=x, im=zero) for x in f_fft])
    return [z.re for z in f0], f1


