import functools
import itertools
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn, optim
try:
    import numba as _nb
except ModuleNotFoundError:
    _nb = None
__all__ = [
    # Core helpers
    "add_loss", "generate_distributions", "fock_to_indices",
    "construct_submatrix", "compute_permanent_torch", "permanent",
    "modes_to_fock",
    # Non-linear parameters
    "compute_nonlinear_parameters", "kerr_phase_factor",
    # Probability utilities
    "boson_sampling_probability", "boson_sampling_probability_advanced",
    # Simulation
    "boson_sampling_simulation",
    # Torch / GAN wrappers
    "BosonSamplerTorch", "BosonLatentGenerator",
    "Generator", "Discriminator",
]


# ------------------------------------------------------------------
# Pure‑Python Ryser permanent (used as fallback when Numba fails)
# ------------------------------------------------------------------
def _ryser_permanent_py(mat: np.ndarray) -> complex:      # noqa: N802
    """
    Pure-Python implementation of Ryser's algorithm for the matrix permanent.

    Parameters
    ----------
    mat : np.ndarray
        Square complex matrix (n x n).

    Returns
    -------
    complex
        Permanent of `mat`.
    """
    n   = len(mat)
    perm = 0j
    rng  = range(1, 1 << n)          # exclude empty subset
    for subset in rng:
        # parity = (‑1)^(|S|)
        parity = 1 - 2 * (subset.bit_count() & 1)
        prod = 1+0j
        for r in mat:
            acc, bit = 0j, 1
            for c in r:
                if subset & bit:
                    acc += c
                bit <<= 1
            prod *= acc
        perm += parity * prod
    return perm

# ------------------------------------------------------------------
def _ryser_permanent(mat: np.ndarray) -> complex:
    """
    Compute the permanent of a square matrix using Rysers formula.
    
    Rysers algorithm runs in O(n*2**n) time and works well for
    small matrices up to about 7x7, handling real and complex entries.

    Parameters:
        mat (np.ndarray): An n x n array whose permanent is computed.

    Returns:
        complex: The permanent of the input matrix.
    """
    n   = mat.shape[0]
    rng = range(1, 1 << n) # exclude empty subset
    perm = 0j

    for subset in rng:
        # --- Numba‑friendly popcount & parity ---
        k = 0
        s = subset
        while s: # Brian Kernighan trick
            s &= s - 1
            k += 1
        parity = 1 - 2 * (k & 1) # +1 if k even, −1 if k odd

        prod = 1+0j
        for r in mat: # iterate rows
            acc, bit = 0j, 1
            for c in r:
                if subset & bit:
                    acc += c
                bit <<= 1
            prod *= acc
        perm += parity * prod
    return perm


# Optional JIT compile and fallback
if _nb:
    try:
        _ryser_permanent = _nb.njit(cache=True)(
            _ryser_permanent)  # type: ignore[assignment]
    except Exception as exc:               # fall back if Numba errors
        import warnings
        warnings.warn(
            f"Numba JIT of _ryser_permanent failed ({exc}); "
            "using pure-Python implementation instead.",
            RuntimeWarning)
        _ryser_permanent = _ryser_permanent_py  # type: ignore[assignment]
else:
    _ryser_permanent = _ryser_permanent_py      # type: ignore[assignment]

# ──────────────────────────────────────────────────────────
# Source loss
# ──────────────────────────────────────────────────────────
def add_loss(state: np.ndarray, loss: float) -> np.ndarray:
    """
    Simulate photon loss on a Fock state vector via binomial draws.

    Each photon in each mode survives with probability (1 - loss).

    Parameters:
        state (np.ndarray): 1D array of photon counts per mode.
        loss (float): Probability that an individual photon is lost.

    Returns:
        np.ndarray: New Fock state vector after loss is applied.
    """
    lossy = np.zeros_like(state)
    for i, n in enumerate(state):
        lossy[i] = np.random.binomial(int(n), 1.0 - loss)
    return lossy


# ──────────────────────────────────────────────────────────
# Non‑linear QD model (Nielsen et al. 2024)
# ──────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=None)
def _paper_params(detuning: float, pulse_bw: float, gamma: float
                  ) -> tuple[float, float, float]:
    """
    Lookup scattering parameters for a nonlinear quantum dot model.

    Based on Nielsen et al 2024, returns eta, ell, and phi_nl:
      - eta: single photon transmission amplitude squared
      - ell: single photon loss probability
      - phi_nl: nonlinear phase shift for two photons

    Parameters:
        detuning (float): Frequency detuning Delta.
        pulse_bw (float): Photon pulse bandwidth sigma.
        gamma (float): Intrinsic linewidth gamma.

    Returns:
        tuple[float, float, float]: (eta, ell, phi_nl).
    """
    x, y = detuning / gamma, pulse_bw / gamma
    eta   = (x**2 + 1) / (x**2 + 1 + (2*y) ** 2)
    ell   = 0.30 / ((1 + x**2) * (1 + y**2))
    phi_nl = (math.pi / 4) / ((1 + x**2) * (1 + y**2))
    return float(eta), float(ell), float(phi_nl)


def _amp_factor(n: int, *, eta: float, ell: float,
                phi_lin: float, phi_nl: float) -> complex:
    """
    Calculate the amplitude factor for n photons scattering off the QD.

    Supports n up to n photons:
      - 0 photons: amplitude = 1 (trivial mapping)
      - 1 photon: amplitude = sqrt(eta) * (1 - ell) * exp(i * phi_lin)
      - 2 photons: amplitude = eta * exp(i * (2 * phi_lin + phi_nl))
      - >2 photons: amplitude = 1 (trivial mapping)

    Parameters:
        n (int): Number of photons (0, 1, or 2).
        eta (float): Transmission parameter.
        ell (float): Single photon loss probability.
        phi_lin (float): Linear phase shift.
        phi_nl (float): Nonlinear phase shift for two photons.

    Returns:
        complex: The complex amplitude factor.
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return math.sqrt(eta) * (1 - ell) * complex(
            math.cos(phi_lin), math.sin(phi_lin)
        )
    elif n == 2:
        return eta * complex(
            math.cos(2 * phi_lin + phi_nl),
            math.sin(2 * phi_lin + phi_nl),
        )
    return 1


# ──────────────────────────────────────────────────────────
# Fock utilities + permanent (Ryser's, Glynn)
# ──────────────────────────────────────────────────────────
def generate_distributions(M: int, N: int):
    """
    Generate all ways to distribute N photons into M modes.

    Yields tuples of length M, each summing to N, representing photon counts per mode.

    Parameters:
        M (int): Number of modes.
        N (int): Total photons.

    Yields:
        tuple[int, ...]: One distribution per iteration.
    """
    if M == 1:
        yield (N,)
        return
    # Iterative stack avoids Python call‑overhead
    stack = [(0, [])] # (k, prefix)
    while stack:
        k, prefix = stack.pop()
        if len(prefix) == M - 1:
            yield tuple(prefix) + (N - k,)
        else:
            for next_k in range(N - k, -1, -1): # Reverse for FIFO order
                stack.append((k + next_k, prefix + [next_k]))


def fock_to_indices(fock: Sequence[int]) -> list[int]:
    """
    Flatten a Fock vector into a list of mode indices.

    Example: [2,1,0] -> [0,0,1]

    Parameters:
        fock (Sequence[int]): Photon counts per mode.

    Returns:
        list[int]: Mode index for each photon.
    """
    idx = []
    for m, c in enumerate(fock):
        idx.extend([m] * c)
    return idx


def construct_submatrix(U: np.ndarray, rows: Sequence[int], cols: Sequence[int]) -> np.ndarray:
    """
    Extract a submatrix of U using specified rows and columns.

    Parameters:
        U (np.ndarray): An m x m matrix.
        rows (Sequence[int]): Rows to keep.
        cols (Sequence[int]): Columns to keep.

    Returns:
        np.ndarray: The submatrix U[np.ix_(rows, cols)].
    """
    return U[np.ix_(rows, cols)]


def compute_permanent_torch(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the permanent of a square tensor A using CPU or GPU.

    Uses Rysers formula on CPU for n <= 12,
    otherwise uses a vectorized Glynn algorithm on GPU.
    """
    # Disable PyTorch deterministic‑algorithms guard for the fast GPU kernel
    det_flag = torch.are_deterministic_algorithms_enabled()
    n = A.shape[0]
    if n > 12 and det_flag:
        torch.use_deterministic_algorithms(False)
    if n <= 12:
        # --- small‑n path: Ryser on CPU ---
        if A.device.type == "cuda":
            A_np = A.detach().cpu().numpy()
        else:
            A_np = A.numpy()
        try:
            val = _ryser_permanent(A_np)
        except Exception:
            # fallback to pure‑Python path if the compiled version blows up
            val = _ryser_permanent_py(A_np)
        if n > 12 and det_flag:
            torch.use_deterministic_algorithms(True)
        return A.new_tensor(val)

    # ------------------------------------------------------------------
    # Glynn formula (vectorised)  P = 2^{1‑n} Σ_δ (∏_i δ_i) ∏_j Σ_i δ_i A_{ij}
    # Works on CPU or GPU and for real/complex dtypes.
    # ------------------------------------------------------------------
    device, dtype = A.device, A.dtype
    # build matrix of ±1 (shape 2^n × n) using bit‑arithmetic
    exp = torch.arange(1 << n, device=device, dtype=torch.int64).unsqueeze(1)
    bits = (exp >> torch.arange(n, device=device, dtype=torch.int64)) & 1
    signs = bits.to(dtype).mul_(2).sub_(1)            # {‑1, +1}

    # S = signs @ A   (broadcasted matmul)
    S = signs @ A                                    # (2^n, n)
    prod = torch.prod(S, dim=1)                      # complex product
    coeff = torch.prod(signs, dim=1)                 # ∏_i δ_i  (real ±1)
    perm = (coeff * prod).sum() * (2.0 ** (1 - n))   # 2^{1‑n} prefactor
    if det_flag:
        torch.use_deterministic_algorithms(True)
    return perm

@functools.lru_cache(maxsize=4096)
def _permanent_squared_from_bytes(mat_bytes: bytes, shape: tuple[int, int]) -> float:
    """
    Memoised helper returning |Perm(A)|² from raw bytes.

    Parameters
    ----------
    mat_bytes : bytes
        Matrix encoded as contiguous complex64 bytes.
    shape : tuple[int, int]
        Original 2-D shape of the matrix.

    Returns
    -------
    float
        Squared modulus of the permanent.
    """
    arr = np.frombuffer(mat_bytes, dtype=np.complex64).reshape(shape).copy()
    p   = compute_permanent_torch(torch.tensor(arr, dtype=torch.complex64))
    return float((p.conj() * p).real)

def _boson_prob(U: np.ndarray, in_idx, out_idx) -> float:
    """
    Ideal boson-sampling probability for one input/output pattern.

    Builds the required sub-matrix of `U`, hashes it, and looks up / caches
    the squared permanent.

    Parameters
    ----------
    U : np.ndarray
        Global m x m interferometer unitary.
    in_idx : Sequence[int]
        Flattened list of input mode indices.
    out_idx : Sequence[int]
        Flattened list of output mode indices.

    Returns
    -------
    float
        Probability value |Perm|².
    """
    sub = U[np.ix_(in_idx, out_idx)].astype(np.complex64)     # convert here
    b   = sub.tobytes()
    return _permanent_squared_from_bytes(b, sub.shape)

# ──────────────────────────────────────────────────────────
# Ideal Clifford sampling
# ──────────────────────────────────────────────────────────

def clifford_sample_ideal(U: np.ndarray,
                          input_state: np.ndarray,
                          n_samples: int = 1) -> np.ndarray:
    m = U.shape[0]
    in_rows = fock_to_indices(input_state)
    samples = []
    for _ in range(n_samples):
        R = list(in_rows)
        C: list[int] = []
        for _ in range(len(in_rows)):
            probs = np.zeros(m, float)
            for j in range(m):
                sub = construct_submatrix(U, R, C + [j])
                p = permanent(sub)
                probs[j] = abs(p)**2
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones(m)/m
            choice = int(np.random.choice(m, p=probs))
            C.append(choice)
            R.pop()
        samples.append(modes_to_fock(C, m))
    return np.array(samples, dtype=int)


# ──────────────────────────────────────────────────────────
# Exact linear sampler (small N) with disk cache
# ──────────────────────────────────────────────────────────
def _linear_distribution(U: np.ndarray, input_state: np.ndarray):
    """
    Compute or load the exact output distribution for a linear unitary.
    """
    n      = int(input_state.sum())
    modes  = len(input_state)
    in_idx = fock_to_indices(input_state)

    states, probs = [], []
    for fock in generate_distributions(modes, n):
        raw = _boson_prob(U, in_idx, fock_to_indices(fock))
        denom = math.prod(math.factorial(nj) for nj in fock)
        p = raw / denom
        if p > 1e-12:
            states.append(fock)
            probs.append(p)
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()

    return states, probs

def clifford_sample(U: np.ndarray,
                    input_state: np.ndarray,
                    n_samples: int = 1,
                    input_loss: float = 0.0) -> np.ndarray:
    """
    Draw samples from the exact linear boson sampling distribution. Uses Clifford & Clifford sampling if ideal conditions

    Optionally apply source loss, then select n_samples outcomes by probability.

    Parameters:
        U (np.ndarray): m x m interferometer unitary.
        input_state (np.ndarray): Input Fock state.
        n_samples (int): Number of samples.
        input_loss (float): Photon loss probability.

    Returns:
        np.ndarray: Sampled Fock states of shape (n_samples, m).
    """
    # Removed special-case shortcut for input_loss == 0.0
    if input_loss > 0.0:
        input_state = add_loss(input_state, input_loss)
    states, probs = _linear_distribution(U, input_state)
    draws = np.random.choice(len(states), size=n_samples, p=probs)
    return np.array([states[i] for i in draws], dtype=int)



# ──────────────────────────────────────────────────────────
# Distributions
# ──────────────────────────────────────────────────────────
def _sample_output_state(input_fock: np.ndarray,
                         U: np.ndarray,
                         *,
                         nonlinear: bool,
                         detuning: float,
                         pulse_bw: float,
                         QD_linewidth: float,
                         phi: float,
                         g2_target: float) -> tuple[np.ndarray, float]:
    """
    Sample one output Fock state under linear or nonlinear model.

    For nonlinear, applies advanced scattering routines; otherwise uses permanents.

    Parameters:
        input_fock (np.ndarray): Input Fock vector.
        U (np.ndarray): Unitary matrix.
        nonlinear (bool): Include nonlinear effects if True.
        detuning (float): Frequency detuning.
        pulse_bw (float): Pulse bandwidth.
        QD_linewidth (float): Quantum dot linewidth.
        phi (float): Linear phase.
        g2_target (float): Target g2.

    Returns:
        (np.ndarray, float): The output Fock vector and its probability.
    """
    n_tot = int(input_fock.sum())
    m     = len(input_fock)

    states, probs = [], []
    for out_fock in generate_distributions(m, n_tot):
        if nonlinear:
            p = boson_sampling_probability_advanced(
                    input_fock, out_fock, U,
                    detuning, pulse_bw, QD_linewidth,
                    phi=phi, g2_target=g2_target)
        else:
            # Linear model
            p = _boson_prob(U,
                            fock_to_indices(input_fock),
                            fock_to_indices(out_fock))
        if p > 0:
            states.append(out_fock)
            probs .append(p)

    if not probs: # Nothing survived
        return input_fock.copy(), 1.0

    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    choice = np.random.choice(len(states), p=probs)
    return np.asarray(states[choice], dtype=int), float(probs[choice])

def sample_distribution(
        U: np.ndarray,
        input_state: np.ndarray,
        *,
        n_samples: int = 10_000,
        input_loss: float = 0.0,
        return_exact: bool = False
) -> (
        dict[tuple[int, ...], float]
        | tuple[dict[tuple[int, ...], float],
                list[tuple[int, ...]],
                np.ndarray]
):
    """
    Monte‑Carlo estimate of the *linear* boson‑sampling output distribution.

    If `input_loss == 0` the exact distribution is computed once and samples
    are drawn i.i.d.; otherwise the Clifford sampler is invoked.

    Parameters
    ----------
    U : np.ndarray
        m × m interferometer unitary.
    input_state : np.ndarray
        Length‑m input Fock vector.
    n_samples : int, default 10 000
        Number of Monte‑Carlo draws.
    input_loss : float, default 0.0
        Per‑photon source‑loss probability.
    return_exact : bool, default False
        Also return the exact `(states, probs)` alongside the histogram.

    Returns
    -------
    dict
        Histogram mapping Fock tuples to empirical probabilities, or
        `(histogram, exact_states, exact_probs)` if `return_exact=True`.
    """
    # For ideal linear sampling, draw i.i.d. samples from the exact distribution
    if input_loss == 0.0:
        states, probs = _linear_distribution(U, input_state)
        # Graceful handling when the distribution is empty
        if len(states) == 0:
            if return_exact:
                return {}, (), np.array([])
            raise RuntimeError(
                "No output states with non‑zero probability – "
                "check the unitary matrix and the input Fock state."
            )
        # Monte Carlo draws from the exact probabilities
        draws = np.random.choice(len(states), size=n_samples, p=probs)
        hist = {}
        for idx in draws:
            fock = states[idx]
            hist[fock] = hist.get(fock, 0) + 1
        # Normalize counts
        hist_normalized = {k: v / n_samples for k, v in hist.items()}
        if return_exact:
            # Also return the exact distribution
            return hist_normalized, states, probs
        return hist_normalized
    # For nonzero loss, fall back to standard sampling
    raw = clifford_sample(U, input_state, n_samples, input_loss)
    hist = {}
    for s in raw:
        k = tuple(int(x) for x in s)
        hist[k] = hist.get(k, 0) + 1
    states_exact, probs_exact = _linear_distribution(U, np.array(input_state))
    if return_exact:
        return {k: v / n_samples for k, v in hist.items()}, states_exact, probs_exact
    return {k: v / n_samples for k, v in hist.items()}


def sample_distribution_nl(
        U: np.ndarray,
        input_state: np.ndarray,
        *,
        n_samples: int = 10_000,
        detuning: float = 0.0,
        pulse_bw: float = 0.0,
        qd_linewidth: float = 1.0,
        phi_linear: float = 0.0,
        input_loss: float = 0.0,
        dark_count_rate: float = 0.0
) -> dict[tuple[int, ...], float]:
    """
    Empirical distribution under the nonlinear QD model.

    We draw from the linear sampler then reweight each sample by its nonlinear amplitude.

    Returns:
        dict[tuple,int→float]
    """
    raw = clifford_sample(U, input_state, n_samples, input_loss)
    eta, ell, phi_nl = _paper_params(detuning, pulse_bw, qd_linewidth)

    hist, norm = {}, 0.0
    for s in raw:
        if dark_count_rate:
            noise = (np.random.rand(len(s)) < dark_count_rate).astype(int)
            s = s + noise
        amp = 1.0 + 0j
        for n in s:
            amp *= _amp_factor(n, eta=eta, ell=ell,
                               phi_lin=phi_linear, phi_nl=phi_nl)
        w = abs(amp)**2
        k = tuple(int(x) for x in s)
        hist[k] = hist.get(k, 0.0) + w
        norm += w
    return {k: w / norm for k, w in hist.items()}


def sample_distribution_nl_torch(
        *args,
        device: str = "cpu",
        **kwargs
) -> tuple[torch.IntTensor, torch.FloatTensor]:
    """
    Torch-compatible wrapper around sample_distribution_nl.

    Returns:
        states: torch.IntTensor, shape (K, m)
        probs : torch.FloatTensor, shape (K,)
    """
    dist = sample_distribution_nl(*args, **kwargs)
    states = torch.tensor(list(dist.keys()), dtype=torch.int32, device=device)
    probs  = torch.tensor(list(dist.values()), dtype=torch.float32, device=device)
    return states, probs
# =============================================================================
# Boson Sampling Simulation Functions
# =============================================================================

def generate_unitary(
        m: int,
        bs_loss: float,
        bs_jitter: float,
        phase_noise_std: float = 0.0,
        systematic_phase_offset: float = 0.0,
        mode_loss: Sequence[float] | None = None
    ) -> np.ndarray:
    """
    Build a Haar-like unitary with loss, jitter, and phase noise.

    Parameters:
        m (int): Number of modes.
        bs_loss (float): Loss factor.
        bs_jitter (float): Amplitude noise.
        phase_noise_std (float): Phase jitter standard deviation.
        systematic_phase_offset (float): Fixed phase offset.
        mode_loss (Sequence[float] or None): Per-mode loss.

    Returns:
        np.ndarray: The m x m modified unitary matrix.
    """
    z = (np.random.randn(m, m) + 1j * np.random.randn(m, m)) / np.sqrt(2)
    q, _ = np.linalg.qr(z)
    U = q
    U *= math.sqrt(bs_loss)
    U += bs_jitter * (np.random.randn(*U.shape) +
                      1j * np.random.randn(*U.shape))
    phases = (np.random.normal(0.0, phase_noise_std, size=m) +
              systematic_phase_offset)
    U = U * np.exp(1j * phases)
    if mode_loss is not None:
        U = U * np.sqrt(np.array(mode_loss))[:, None]
    return U


def permanent(numpy_matrix):
    """
    Compute the permanent of a numpy matrix via the torch routine.

    Parameters:
        numpy_matrix (np.ndarray): Square array.

    Returns:
        float: The permanent value.
    """
    A_torch = torch.tensor(numpy_matrix, dtype=torch.complex64)
    return compute_permanent_torch(A_torch).detach().cpu().numpy()

def modes_to_fock(mode_indices, m):
    """
    Convert a list of single-photon mode indices to a Fock vector of length m.

    E.g. [0,2,2] → [1,0,2,...].

    Parameters:
        mode_indices: Sequence[int]
        m: int

    Returns:
        np.ndarray of length m
    """
    return np.bincount(mode_indices, minlength=m)

# =============================================================================
# Nonlinear Parameter Functions
# =============================================================================

def compute_nonlinear_parameters(detuning, pulse_bw, QD_linewidth):
    """
    Translate physical QD parameters (detuning, pulse_bw, QD_linewidth) into the
    effective nonlinear trio (η, φ_NL, ℓ_NL).

    All inputs are positive real numbers in the same frequency units.
    Outputs are dimensionless amplitudes (η, ℓ_NL) and a phase in radians (φ_NL).

    Returns
    -------
    tuple[float, float, float]
        (η, φ_NL, ℓ_NL)
    """
    rel_detuning = detuning / QD_linewidth
    rel_bandwidth = pulse_bw / QD_linewidth
    phiNL_max = np.pi / 4  
    phiNL = phiNL_max / ((1 + rel_detuning**2) * (1 + rel_bandwidth**2))
    lNL_max = 0.3  
    lNL = lNL_max / ((1 + rel_detuning**2) * (1 + rel_bandwidth**2))
    eta_min = 0.7  
    eta = 1 - (1 - eta_min) / ((1 + rel_detuning**2) * (1 + rel_bandwidth**2))
    return eta, phiNL, lNL

def kerr_phase_factor(n, detuning, pulse_bw, QD_linewidth, eta=1.0, phi=0.0, g2_target=0):
    """
    Complex amplitude factor for *n* photons traversing a Kerr‑type element.

    When `g2_target == 1` the element behaves linearly
    (η = 1, φ_NL = 0, ℓ_NL = 0).

    Raises
    ------
    ValueError
        If `n` is negative.
    """
    if g2_target == 1:
        eta_eff, phiNL_eff, lNL_eff = 1.0, 0.0, 0.0
    else:
        eta_eff, phiNL_eff, lNL_eff = compute_nonlinear_parameters(detuning, pulse_bw, QD_linewidth)
    if n == 0:
        return 1.0
    elif n == 1:
        return np.sqrt(eta_eff * (1 - lNL_eff)) * np.exp(1j * phi)
    elif n == 2:
        return np.sqrt(eta_eff**2) * np.exp(1j * (2 * phi + phiNL_eff))
    else:
        single_factor = np.sqrt(eta_eff * (1 - lNL_eff)) * np.exp(1j * phi)
        return single_factor ** n

# =============================================================================
# Boson Sampling Probability Functions
# =============================================================================

def boson_sampling_probability(
        U: np.ndarray,
        input_modes: Sequence[int],
        output_modes: Sequence[int],
        effective_mu: float,
        m: int
) -> float:
    """
    Transition probability for partially distinguishable linear bosons.

    P = mu |Perm|^2 + (1-mu) prod|U_ij|^2.
    """
    if len(input_modes) == 0 or len(output_modes) == 0:
        return 0
    U_sub = construct_submatrix(U, input_modes, output_modes)
    A_sub_torch = torch.tensor(U_sub, dtype=torch.complex64)
    perm_val = compute_permanent_torch(A_sub_torch)
    classical_probability = np.prod(np.abs(U_sub)**2)
    perm_val_np = perm_val.detach().cpu().numpy()
    probability = effective_mu * np.abs(perm_val_np)**2 + (1 - effective_mu) * classical_probability
    return probability

def boson_sampling_probability_advanced(input_state, output_state, U_linear, detuning, pulse_bw, QD_linewidth, phi=0.0, g2_target=0):
    """
    Exact probability including one nonlinear QD scatterer in a Trotter step.

    Sums over intermediate Fock layers and applies kerr_phase_factor.
    """
    N = sum(input_state)
    M = len(input_state)
    if g2_target == 1:
        input_indices = fock_to_indices(input_state)
        output_indices = fock_to_indices(output_state)
        submat = construct_submatrix(U_linear, input_indices, output_indices)
        perm_val = permanent(submat)
        norm = np.prod([math.factorial(n) for n in input_state]) * np.prod([math.factorial(n) for n in output_state])
        return (abs(perm_val)**2) / norm

    U1 = np.eye(M)
    U2 = U_linear
    total_amplitude = 0.0 + 0.0j
    for intermediate in generate_distributions(M, N):
        # Convert Fock vectors to explicit lists of indices.
        input_indices = fock_to_indices(input_state)
        intermediate_indices = fock_to_indices(intermediate)
        output_indices = fock_to_indices(output_state)
        
        submat1 = construct_submatrix(U1, input_indices, intermediate_indices)
        
        if np.allclose(submat1, np.eye(submat1.shape[0])):
            amp_in_to_int = 1.0 + 0.0j
        else:
            amp_in_to_int = permanent(submat1)
        
        submat2 = construct_submatrix(U2, intermediate_indices, output_indices)
        amp_int_to_out = permanent(submat2)
        if amp_in_to_int == 0 or amp_int_to_out == 0:
            continue
        amp_nl = 1.0 + 0.0j
        for mode_idx, n_mode in enumerate(intermediate):
            amp_nl *= kerr_phase_factor(n_mode, detuning, pulse_bw, QD_linewidth, phi=phi, g2_target=g2_target)
        total_amplitude += amp_in_to_int * amp_nl * amp_int_to_out
    return abs(total_amplitude)**2

# =============================================================================
# Other Simulation Functions
# =============================================================================
def draw_n_from_g2(g2: float) -> int:
    """
    Sample photon number n e {0,1,2} for a source with <n>=1 and given g2.

    Supports g2 e [0,2].
    """
    if not 0.0 <= g2 <= 2.0:
        raise ValueError("g2 must be in [0,2]")

    # --- Anti-/sub‑Poissonian (g2 ≤ 0.5) ---
    if g2 <= 0.5:
        p2 = 2.0 * g2 / (1.0 + 2.0 * g2)
        return 2 if np.random.rand() < p2 else 1
    
    # --- Intermediate sub‑Poissonian 0.5 < g2 < 1 ---
    if g2 < 1.0:
        p2 = 0.5 * g2
        return 2 if np.random.rand() < p2 else 1


    # --- Poisson special case  g2 == 1 ---
    if abs(g2 - 1.0) < 1e-12:
        return min(np.random.poisson(1.0), 2)

    # --- Super‑Poissonian / thermal (1 < g2 ≤ 2) ---
    k = 1.0 / (g2 - 1.0)            # cluster size
    p = k / (k + 1.0)               # NB parameter
    n = np.random.negative_binomial(int(round(k)), p)
    return min(n, 2)                # **truncate to ≤2 photons**



def compute_g2(output_modes, m):
    """
    Compute g(2) = <n(n-1)>/<n>^2 from a flat list of output mode indices.
    """
    counts = np.bincount(output_modes, minlength=m)
    total_counts = counts.sum()
    if total_counts < 2:
        return 0
    return np.sum(counts * (counts - 1)) / (total_counts * (total_counts - 1))

def sample_input_state(m: int, num_sources: int, *,
                       g2_target: float,
                       input_loss: float,
                       coupling_efficiency: float,
                       detector_inefficiency: float) -> tuple[np.ndarray, int]:
    """
    Sample which sources fire and how many photons each emits, then assign them to modes.

    Returns (mode_indices_array, total_photons).
    """
    # Effective trigger probability: survive loss then coupling and detection
    p_eff = (1.0 - input_loss) * coupling_efficiency * detector_inefficiency
    modes: list[int] = []

    for _ in range(num_sources):
        if np.random.rand() < p_eff:                 # this source triggers
            n_emit = draw_n_from_g2(g2_target)
            modes.extend(np.random.choice(m, n_emit, replace=True).tolist())

    return np.array(modes, dtype=int), len(modes)

def apply_dark_counts(probability, m, dark_count_rate):
    """
    Add uniform dark-count probability m*dark_count_rate to an event probability.
    """
    return probability + dark_count_rate * m

def boson_sampling_simulation(
        m, num_sources, num_loops,
        input_loss, coupling_efficiency, detector_inefficiency,
        mu, temporal_mismatch, spectral_mismatch, arrival_time_jitter,
        bs_loss, bs_jitter, phase_noise_std, systematic_phase_offset,
        mode_loss, dark_count_rate,
        *,
        precomputed_U=None,
        use_advanced_nonlinearity=False,
        detuning=0.0, pulse_bw=0.5, QD_linewidth=1.0, phi=0.0,
        g2_target=0.0) -> tuple[float, float, np.ndarray]:
    """
    Monte Carlo simulate many boson sampling shots.

    Repeats source sampling, state evolution, and output sampling
    to compute average probability, average g2, and mode distribution.

    Parameters:
        m (int): Number of modes.
        num_sources (int): Number of photon sources.
        num_loops (int): Number of trials.
        input_loss (float): Source loss probability.
        coupling_efficiency (float): Source coupling efficiency.
        detector_inefficiency (float): Detector inefficiency.
        mu (float): Degree of indistinguishability.
        temporal_mismatch (float): Temporal mismatch fraction.
        spectral_mismatch (float): Spectral mismatch fraction.
        arrival_time_jitter (float): Arrival time jitter fraction.
        bs_loss (float): Interferometer loss factor.
        bs_jitter (float): Interferometer jitter.
        phase_noise_std (float): Phase noise standard deviation.
        systematic_phase_offset (float): Fixed phase offset.
        mode_loss (Sequence[float]): Per-mode loss factors.
        dark_count_rate (float): Dark count probability per mode.
        use_advanced_nonlinearity (bool): Include QD nonlinearity.
        detuning (float): Frequency detuning.
        pulse_bw (float): Pulse bandwidth.
        QD_linewidth (float): QD intrinsic linewidth.
        phi (float): Linear phase shift.
        g2_target (float): Target second-order correlation.
        precomputed_U (np.ndarray or None): If supplied, reuse this unitary
            matrix instead of drawing a new Haar‑random one each call.

    Returns:
        (float, float, np.ndarray):
            avg_prob: Average event probability.
            avg_g2: Average measured g2.
            mode_dist: Empirical mode occupation distribution.
    """
    # ------------------------------------------------------------------
    # Interferometer unitary: generate once or reuse cached copy
    # ------------------------------------------------------------------
    if precomputed_U is None:
        U = generate_unitary(m, bs_loss, bs_jitter,
                             phase_noise_std, systematic_phase_offset,
                             mode_loss)
    else:
        U = precomputed_U
    eff_mu = mu*(1-temporal_mismatch)*(1-spectral_mismatch)*(1-arrival_time_jitter)
    probs_list, g2_list = [], []
    counts = np.zeros(m)
    shots = 0
    for _ in range(num_loops):
        raw_in, n_phot = sample_input_state(
            m, num_sources,
            g2_target=g2_target,
            input_loss=input_loss,
            coupling_efficiency=coupling_efficiency,
            detector_inefficiency=detector_inefficiency)
        if n_phot==0: continue
        input_fock = modes_to_fock(raw_in, m)
        if use_advanced_nonlinearity:
            out_fock, p_drawn = _sample_output_state(
                input_fock, U,
                nonlinear=True,
                detuning=detuning, pulse_bw=pulse_bw,
                QD_linewidth=QD_linewidth,
                phi=phi, g2_target=g2_target)
        else:
            # ideal-check
            ideal = (
                input_loss==0.0
                and coupling_efficiency==1
                and detector_inefficiency==1
                and mu==1.0
                and temporal_mismatch==0.0
                and spectral_mismatch==0.0
                and arrival_time_jitter==0.0
                and dark_count_rate==0.0
                and bs_loss==1.0
                and bs_jitter==0.0
                and phase_noise_std==0.0
                and systematic_phase_offset==0.0
                and (mode_loss is None or np.allclose(mode_loss, 1))
            )
            if ideal:
                out_fock = clifford_sample_ideal(U, input_fock, 1)[0]
                p_drawn = boson_sampling_probability(
                    U,
                    fock_to_indices(input_fock),
                    fock_to_indices(out_fock),
                    effective_mu=1.0,
                    m=m)
            else:
                out_fock, p_drawn = _sample_output_state(
                    input_fock, U,
                    nonlinear=False,
                    detuning=detuning, pulse_bw=pulse_bw,
                    QD_linewidth=QD_linewidth,
                    phi=phi, g2_target=g2_target)
        raw_out = fock_to_indices(out_fock)
        if not use_advanced_nonlinearity and not ideal:
            p_drawn = (eff_mu * p_drawn
                       + (1-eff_mu)
                       * np.prod(np.abs(
                           construct_submatrix(
                               U,
                               fock_to_indices(input_fock),
                               raw_out))**2))
        p_drawn = apply_dark_counts(p_drawn, m, dark_count_rate)
        probs_list.append(p_drawn)
        g2_list.append(compute_g2(raw_out, m))
        counts += out_fock
        shots += 1
    if shots>0:
        mode_dist = counts/shots; mode_dist/=mode_dist.sum()
    else:
        mode_dist = np.zeros(m)
    return (float(np.mean(probs_list)) if probs_list else 0.0,
            float(np.mean(g2_list))    if g2_list    else 0.0,
            mode_dist)


# =============================================================================
# PyTorch Module: Wrap Boson Sampling Simulation
# =============================================================================

class BosonSamplerTorch(nn.Module):
    """
    Wrap the Monte-Carlo boson-sampling loop in a torch.nn.Module.
    The sole trainable parameter is a systematic phase offset.
    Forward -> (prob_tensor, latent_tensor).
    """
    def __init__(self,
                 early_late_pairs: int,
                 input_state: list[int] | None,
                 num_loops: int,
                 input_loss: float,
                 coupling_efficiency: float,
                 detector_inefficiency: float,
                 mu: float,
                 temporal_mismatch: float,
                 spectral_mismatch: float,
                 arrival_time_jitter: float,
                 bs_loss: float,
                 bs_jitter: float,
                 phase_noise_std: float,
                 systematic_phase_offset: float,
                 mode_loss: np.ndarray,
                 dark_count_rate: float,
                 use_advanced_nonlinearity: bool = False,
                 detuning: float = 0.0,
                 pulse_bw: float = 0.0,
                 QD_linewidth: float = 1.0,
                 phi: float = 0.0,
                 g2_target: float = 0.0,
                 fixed_unitary: np.ndarray | None = None,
                 ):
        super(BosonSamplerTorch, self).__init__()
        # Encode early-late photon pairs and optional fixed input
        self.early_late_pairs = early_late_pairs
        self.input_state = input_state
        # Derive number of modes and number of sources
        if input_state is not None:
            # Use provided Fock vector
            self.m = len(input_state)
            self.num_sources = int(sum(input_state))
        else:
            # Fall back to early-late pairs
            self.m = early_late_pairs * 2
            self.num_sources = early_late_pairs
        self.num_loops = num_loops
        self.input_loss = input_loss
        self.coupling_efficiency = coupling_efficiency
        self.detector_inefficiency = detector_inefficiency
        self.mu = mu
        self.temporal_mismatch = temporal_mismatch
        self.spectral_mismatch = spectral_mismatch
        self.arrival_time_jitter = arrival_time_jitter
        self.bs_loss = bs_loss
        self.bs_jitter = bs_jitter
        self.phase_noise_std = phase_noise_std
        self.systematic_phase_offset = nn.Parameter(torch.tensor(systematic_phase_offset, dtype=torch.float32))
        self.mode_loss = mode_loss
        self.dark_count_rate = dark_count_rate
        self.use_advanced_nonlinearity = use_advanced_nonlinearity
        self.detuning = detuning
        self.pulse_bw = pulse_bw
        self.QD_linewidth = QD_linewidth
        self.phi = phi
        self.g2_target = g2_target
        self.fixed_U = fixed_unitary

    def forward(self, batch_size: int):
        """
        Run simulation under torch.no_grad() and return:

        - prob_tensor: (batch_size,) with avg_prob replicated.
        - dist_tensor: (m,) the mode-occupation distribution.
        """
        phase_offset = self.systematic_phase_offset.detach().cpu().numpy().item()

        with torch.no_grad():
            if self.input_state is not None:
                # Fixed Fock input distribution
                input_arr = np.array(self.input_state)
                # Pick the unitary for this forward pass
                if self.fixed_U is not None:
                    U_tmp = self.fixed_U
                else:
                    U_tmp = generate_unitary(
                        self.m,
                        self.bs_loss, self.bs_jitter,
                        self.phase_noise_std, phase_offset,
                        self.mode_loss)
                if self.use_advanced_nonlinearity:
                    states, probs = sample_distribution_nl_torch(
                        self.fixed_U if self.fixed_U is not None else U_tmp, input_arr,
                        device=self.systematic_phase_offset.device,
                        n_samples=self.num_loops,
                        detuning=self.detuning,
                        pulse_bw=self.pulse_bw,
                        qd_linewidth=self.QD_linewidth,
                        phi_linear=self.phi,
                        input_loss=self.input_loss,
                        dark_count_rate=self.dark_count_rate
                    )
                else:
                    dist = sample_distribution(
                        U_tmp, input_arr,
                        n_samples=self.num_loops,
                        input_loss=self.input_loss
                    )
                    states = torch.tensor(
                        list(dist.keys()), dtype=torch.int32,
                        device=self.systematic_phase_offset.device
                    )
                    probs = torch.tensor(
                        list(dist.values()), dtype=torch.float32,
                        device=self.systematic_phase_offset.device
                    )
                # --- draw `batch_size` Fock states and use them as the latent code ---
                draws = torch.multinomial(
                    probs, num_samples=batch_size, replacement=True
                )                  # shape (batch_size,)
                sampled_fock = states[draws]          # shape (batch_size, m)
                avg_prob = float(probs.mean().item())
            else:
                avg_prob, measured_g2, mode_dist = boson_sampling_simulation(
                    self.m, self.num_sources, self.num_loops,
                    self.input_loss, self.coupling_efficiency, self.detector_inefficiency,
                    self.mu, self.temporal_mismatch, self.spectral_mismatch,
                    self.arrival_time_jitter,
                    self.bs_loss, self.bs_jitter, self.phase_noise_std,
                    phase_offset, self.mode_loss,
                    self.dark_count_rate,
                    use_advanced_nonlinearity=self.use_advanced_nonlinearity,
                    detuning=self.detuning, pulse_bw=self.pulse_bw,
                    QD_linewidth=self.QD_linewidth, phi=self.phi,
                    g2_target=self.g2_target
                )

        #print(f"Request g2: {self.g2_target}   Measured g2: {measured_g2}")

        # —— convert numpy to torch tensors ——
        prob_tensor  = torch.tensor(avg_prob , dtype=torch.float32,
                                    device=self.systematic_phase_offset.device)
        dist_tensor  = sampled_fock.to(torch.float32)

        # `prob_tensor` is still useful e.g. as a training signal,
        # but the latent space will now be the distribution
        return prob_tensor.expand(batch_size), dist_tensor    # <- 2nd return value

# =============================================================================
# Boson Latent Generator: Wrap the boson sampler to produce a latent vector.
# =============================================================================

class BosonLatentGenerator(nn.Module):
    """
    Wrap BosonSamplerTorch → latent vector.

    Uses the mode-distribution as the latent embedding, replicated over batch.
    """
    def __init__(self, latent_dim: int, boson_sampler_module: BosonSamplerTorch):
        super(BosonLatentGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.boson_sampler_module = boson_sampler_module

    def forward(self, batch_size: int):
        """
        Returns latent_tensor of shape (batch_size, m).
        """
        _, latent_tensor = self.boson_sampler_module(batch_size)
        return latent_tensor      # already shape (batch_size, m)


# =============================================================================
# Simple GAN Components
# =============================================================================

class Generator(nn.Module):
    r"""A simple generator network for a GAN.
    
    Maps a latent vector to an output image (flattened).
    """
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    r"""A simple discriminator network for a GAN.
    
    Maps an input image (flattened) to a probability of being real.
    """
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# =============================================================================
# Training Loop
# =============================================================================

if __name__ == "__main__":
    """
    Main training loop for a GAN that uses boson sampling simulation for latent space generation.
    
    This script initializes the boson sampling simulation, wraps it in a latent generator,
    and trains a GAN using generated latent vectors and dummy real images.
    """
    num_loops             = 100
    # Test specific 50/50 beam splitter circuit with two modes
    input_state = [4,0,0,0,0,0]  # one photon in each input mode
    m = len(input_state)
    num_sources = int(sum(input_state))

    # --- Source/per‐mode perfect transmission:
    input_loss            = 0.0
    coupling_efficiency   = 1.0
    detector_inefficiency = 1.0

    # --- Indistinguishability:
    mu                    = 1.0
    temporal_mismatch     = 0.0
    spectral_mismatch     = 0.0
    arrival_time_jitter   = 0.0

    # --- Interferometer:
    bs_loss               = 1.0
    bs_jitter             = 0.0
    phase_noise_std       = 0.0
    systematic_phase_offset = 0.0
    mode_loss             = np.ones(m)

    # --- Dark Counts:
    dark_count_rate       = 0.0

    # --- Linear model only:
    use_advanced_nonlinearity = True

    # --- Ideal single‐photon sources (optional):
    g2_target             = 0.0
    
    # --- QD Nonlinearity
    detuning              = 0.0
    pulse_bw              = 0.0
    QD_linewidth          = 1
    phi                   = 0

    # Latent space parameters.
    latent_dim  = m         # Dimension for the GAN latent space.
    output_dim = 28 * 28    # For example, generating MNIST-like flattened images.

    # Instantiate the boson sampler module.
    boson_sampler_module = BosonSamplerTorch(
        m, input_state, num_loops,
        input_loss, coupling_efficiency, detector_inefficiency,
        mu, temporal_mismatch, spectral_mismatch, arrival_time_jitter,
        bs_loss, bs_jitter, phase_noise_std, systematic_phase_offset, mode_loss,
        dark_count_rate,
        use_advanced_nonlinearity,
        detuning, pulse_bw, QD_linewidth, phi, g2_target,
    )

    # Wrap the boson sampler module in a latent generator.
    latent_generator = BosonLatentGenerator(latent_dim, boson_sampler_module)

    # Instantiate GAN components.
    generator = Generator(latent_dim, output_dim)
    discriminator = Discriminator(output_dim)

    # Optimizers and loss function.
    opt_G = optim.Adam(generator.parameters(), lr=0.0002)
    opt_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    num_epochs = 100
    batch_size = 16
    for epoch in range(num_epochs):
        z = latent_generator(batch_size)
        # Generate latent vectors using the boson latent generator.
        fake_images = generator(z)

        # Dummy real images (in practice, replace with your dataset).
        real_images = torch.randn(batch_size, output_dim)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train Discriminator.
        discriminator.zero_grad()
        outputs_real = discriminator(real_images)
        loss_real = criterion(outputs_real, real_labels)

        outputs_fake = discriminator(fake_images.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        opt_D.step()

        # Train Generator.
        generator.zero_grad()
        outputs_fake = discriminator(fake_images)
        loss_G = criterion(outputs_fake, real_labels)
        loss_G.backward()
        opt_G.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")