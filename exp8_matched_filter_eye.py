"""
Experiment 8 - Matched filtering, ISI and eye diagrams
BPSK + root-raised-cosine (RRC) pulse shaping + AWGN + matched filter + eye diagrams.

Pure NumPy + Matplotlib. No communications toolbox / scikit-commpy / etc.
Everything (RRC design, AWGN, matched filter, eye diagram, BER) is implemented by hand.

Conventions
-----------
Bit -> symbol mapping : bit 0 -> -1, bit 1 -> +1
Detection rule        : r > 0 -> bit 1, else bit 0
RRC normalisation     : sum(h**2) = 1  (unit energy) => Es = 1, cascade peak = 1
SNR definition        : Es/N0 in dB (Es = energy per symbol, N0 = noise PSD).
                        For BPSK Es = Eb.  Real noise variance per sample:
                        sigma^2 = N0/2 = Es / (2 * 10**(EsN0_dB/10)).
Time unit             : T = symbol period = SPS samples.
Delay                 : TX = (L-1)/2, RX = (L-1)/2, total = L-1 = SPAN*SPS samples.
Indexing              : Python is 0-based, so the cascade peak (= total delay D)
                        is at array index D, and decision k (k = 0..N-1) is at
                        index D + k*SPS (+ timing offset).

Run:   python exp8_matched_filter_eye.py            (shows figures, saves PNGs)
       python exp8_matched_filter_eye.py --no-show  (headless, only saves PNGs)
"""

# %% [markdown]
# # Experiment 8 - Matched filtering, ISI and eye diagrams
# BPSK + RRC pulse shaping + AWGN + matched filter, implemented from scratch with NumPy.
#
# **Conventions:** bit 0 -> -1, bit 1 -> +1; RRC has unit energy (so Es = 1);
# SNR is Es/N0 in dB (= Eb/N0 for BPSK); noise variance per real sample is
# sigma^2 = Es / (2 * 10^(EsN0/10)); total filter delay D = span*sps samples.

# %% Imports and parameters
import os
import sys
import math
import numpy as np
import matplotlib

SHOW_FIGS = "--no-show" not in sys.argv
IN_NOTEBOOK = "ipykernel" in sys.modules
if not SHOW_FIGS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- PARAMETERS (edit here) ----------------
RNG_SEED = 1                 # seed for the data bits
NOISE_SEED = 100             # seed re-applied before every noise realisation
N = 100_000                  # number of BPSK symbols (>= 10000 required)
SPS = 8                      # samples per symbol (must be even)
SPAN = 10                    # RRC span in symbols
BETA = 0.35                  # reference roll-off factor
ESN0_REF = 10                # dB, waveform demonstrations
ESN0_VAL = 10                # dB, delay-validation experiment
SNR_EYES = [0, 5, 10, 20]    # dB, eye diagrams (SNR study)
SNR_SWEEP = np.arange(0, 21, 2)   # dB, eye height vs SNR
SNR_BER = np.arange(0, 10, 1)     # dB, BER vs SNR (validates AWGN scaling)
BETA_LIST = [0.2, 0.5, 0.8]       # roll-off study
SPAN_LIST = [2, 4, 6, 10]         # span study
ESN0_ROLLSPAN = 20           # dB, roll-off and span eyes
ESN0_TIMING = 7              # dB, BER vs timing offset
ESN0_EHOFF = 20              # dB, eye height vs timing offset
CH_TAPS = np.array([1.0, 0.5, 0.25])      # multipath tap amplitudes
CH_DELAY = np.array([0, SPS // 2, SPS])   # multipath delays in samples (0, T/2, T)
ESN0_MP = 20                 # dB, multipath eyes
SNR_MP = np.arange(0, 10, 1)      # dB, BER with / without multipath
NUM_TRACES = 300             # traces overlaid per eye diagram
SAVE_FIGS = True
OUT_DIR = "exp8_figures"

assert SPS % 2 == 0, "SPS must be even so that T/2 is a whole number of samples"
if SAVE_FIGS:
    os.makedirs(OUT_DIR, exist_ok=True)


# %% Functions (RRC, link, AWGN, sampling, eye diagram, metrics)
def rrc_filter(beta, span, sps):
    """Root-raised-cosine impulse response, unit energy, T = 1.

    t in units of T: t = n/sps, n = -(L-1)/2 .. (L-1)/2, L = span*sps + 1.
    h(t) = [sin(pi t (1-b)) + 4 b t cos(pi t (1+b))] / [pi t (1 - (4 b t)^2)]
    Special values (the general formula is 0/0 there):
        t = 0         : h = 1 - b + 4b/pi
        |t| = 1/(4b)  : h = (b/sqrt2) [(1+2/pi) sin(pi/4b) + (1-2/pi) cos(pi/4b)]
    Normalisation: sum(h^2) = 1 so the TX*RX cascade peaks at exactly 1 and
    +/-1 BPSK has Es = 1.
    """
    L = span * sps + 1
    t = np.arange(-(L - 1) // 2, (L - 1) // 2 + 1) / sps
    h = np.zeros(L)
    for n in range(L):
        tn = t[n]
        if abs(tn) < 1e-12:
            h[n] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(tn) - 1 / (4 * beta)) < 1e-9:
            h[n] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                          + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[n] = (np.sin(np.pi * tn * (1 - beta)) + 4 * beta * tn * np.cos(np.pi * tn * (1 + beta))) / \
                   (np.pi * tn * (1 - (4 * beta * tn) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def add_awgn(x, esn0_db, nsym, seed):
    """Real AWGN for a given Es/N0. Es = measured energy per symbol of x.
    N0 = Es / 10^(EsN0/10), sigma^2 = N0/2 per real sample.
    The generator is re-seeded so every SNR uses the same unit-variance noise
    shape, scaled differently (smooth sweeps)."""
    rng = np.random.default_rng(seed)
    es = np.sum(x ** 2) / nsym
    n0 = es / 10 ** (esn0_db / 10)
    sigma2 = n0 / 2
    return x + np.sqrt(sigma2) * rng.standard_normal(x.shape), sigma2


def run_link(sym, h, sps, hch, esn0_db, seed):
    """BPSK -> upsample -> RRC TX -> [channel] -> AWGN -> RRC matched filter.
    Returns y (full-length matched-filter output, delay NOT yet removed), tx, r, sigma2.
    hch=None means no channel; esn0_db=np.inf means noise-free."""
    n = len(sym)
    up = np.zeros(n * sps)
    up[::sps] = sym                                  # explicit upsampling
    tx = np.convolve(up, h)                          # pulse shaping
    x = tx if hch is None else np.convolve(tx, hch)  # multipath BEFORE noise
    if np.isinf(esn0_db):
        r, sigma2 = x, 0.0
    else:
        r, sigma2 = add_awgn(x, esn0_db, n, seed)
    y = np.convolve(r, h[::-1])                      # matched filter h_mf[n] = h_tx[-n]
    return y, tx, r, sigma2


def sample_symbols(y, d, sps, n, off=0):
    """Decision samples at index d + k*sps + off (k = 0..n-1). `valid` marks the
    symbols that have an in-range sample, so TX and RX sequences used for BER
    always have equal length."""
    idx = d + np.arange(n) * sps + off
    valid = (idx >= 0) & (idx < len(y))
    return y[idx[valid]], valid


def eye_height_3sigma(rk, s):
    """Statistical eye height [mean(+1 grp) - 3 std] - [mean(-1 grp) + 3 std]."""
    a, b = rk[s > 0], rk[s < 0]
    return (a.mean() - 3 * a.std(ddof=1)) - (b.mean() + 3 * b.std(ddof=1))


def eye_height_worst(rk, s):
    """Worst-case eye height of a noise-free sample set: min(+1 grp) - max(-1 grp)."""
    return rk[s > 0].min() - rk[s < 0].max()


def eye_height_analytic(gp, d, sps, off=0):
    """Noise-free worst-case eye height from the overall pulse gp (TX*channel*RX):
    EH = 2 * (gp[main] - sum |gp[other symbol instants]|), main sample at d+off."""
    main = gp[d + off]
    k = int(np.ceil(len(gp) / sps))
    m = np.r_[-k:0, 1:k + 1]
    idx = d + off + m * sps
    ok = (idx >= 0) & (idx < len(gp))
    return 2 * (main - np.sum(np.abs(gp[idx[ok]])))


def plot_eye(ax, y, d, sps, num_traces, k_start, title, mark_t=(0, 1), yl=3):
    """Overlay two-symbol-long segments of the matched-filter output.

    y        : matched-filter output (delay not yet removed)
    d        : total cascade delay in samples; removed here: yc = y[d:], so
               yc[(k-1)*sps] is the decision sample of symbol k (k = 1, 2, ...)
    Each trace has 2*sps+1 samples spanning t = -0.5T .. +1.5T around the
    decision instants (t = 0 and t = T): two eye openings. Traces start one
    symbol apart (overlapping windows).
    mark_t   : times (in T) for the red vertical sampling markers.
    """
    yc = y[d:]
    starts = ((k_start - 1) + np.arange(num_traces)) * sps - sps // 2
    idx = starts[:, None] + np.arange(2 * sps + 1)[None, :]
    assert idx.min() >= 0 and idx.max() < len(yc), "eye traces exceed signal length"
    seg = yc[idx]                                    # num_traces x (2*sps+1)
    t = np.arange(-sps // 2, int(1.5 * sps) + 1) / sps
    ax.plot(t, seg.T, color="#4f7fe0", lw=0.5, alpha=0.35)
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.axhline(-1, color="k", ls="--", lw=0.8)
    for mt in np.atleast_1d(mark_t):
        ax.axvline(mt, color="r", lw=1.3)
    ax.set_ylim(-yl, yl)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("Time relative to decision instant (T)")
    ax.set_ylabel("Matched-filter output (amplitude)")
    ax.set_title(title, fontsize=9)


def q_ber(esn0_db):
    """Theoretical BPSK BER = 0.5 erfc(sqrt(Es/N0))."""
    return np.array([0.5 * math.erfc(math.sqrt(10 ** (v / 10))) for v in np.atleast_1d(esn0_db)])


_figs = []


def finish(fig, name):
    """Save the figure (and show it immediately in a notebook)."""
    if SAVE_FIGS:
        fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=150, bbox_inches="tight")
    _figs.append(fig)
    if IN_NOTEBOOK:
        plt.show()


# %% BPSK generation
rng = np.random.default_rng(RNG_SEED)
bits = rng.integers(0, 2, N)           # random bits
sym = 2.0 * bits - 1.0                 # 0 -> -1, 1 -> +1
print(f"BPSK: {N} symbols, mapping 0->-1, 1->+1, seed {RNG_SEED}")

# %% RRC filter and EXPLICIT delay calculation
h = rrc_filter(BETA, SPAN, SPS)        # transmit filter
h_mf = h[::-1]                         # matched filter h_mf[n] = h_tx[-n]
print(f"RRC check: any NaN/Inf = {int(not np.all(np.isfinite(h)))}, energy = {np.sum(h**2):.6f}")

Ltx = len(h);    Dtx = (Ltx - 1) // 2      # symmetric FIR -> group delay (L-1)/2
Lrx = len(h_mf); Drx = (Lrx - 1) // 2
Dtot = Dtx + Drx
print("\n---------------- FILTER DELAY (calculated) ----------------")
print(f"span = {SPAN} symbols, sps = {SPS}, beta = {BETA:.2f}")
print(f"TX filter length       = {Ltx} taps  (span*sps + 1)")
print(f"TX group delay         = {Dtx} samples = {Dtx / SPS:.2f} symbols")
print(f"RX filter length       = {Lrx} taps")
print(f"RX group delay         = {Drx} samples = {Drx / SPS:.2f} symbols")
print(f"Total cascade delay    = {Dtot} samples")
print(f"Total cascade delay    = {Dtot / SPS:.2f} symbols")

# numerical proof (three independent measurements; 0-based argmax = delay in samples)
g = np.convolve(h, h_mf)                   # cascade impulse response
Dtot_pk = int(np.argmax(g))                # (1) peak of cascade pulse
Dtx_pk = int(np.argmax(h))                 # (2) peak of one RRC filter
imp = np.zeros(6 * Ltx); imp[0] = 1        # (3) unit impulse through the chain
Dtot_imp = int(np.argmax(np.convolve(np.convolve(imp, h), h_mf)))
print(f"\nMeasured: TX peak index = {Dtx_pk}, cascade peak = {Dtot_pk}, impulse-through-chain peak = {Dtot_imp}")
print("DELAY CHECK: PASS (formula = measurement)" if (Dtot_pk == Dtot and Dtot_imp == Dtot and Dtx_pk == Dtx)
      else "DELAY CHECK: FAIL")
print(f"Cascade value at the sampling instant = {g[Dtot]:.6f} (should be 1)")
isi_taps = g[Dtot + np.arange(-SPAN, SPAN + 1) * SPS].copy(); isi_taps[SPAN] = 0
print(f"Max |cascade| at other symbol instants (truncation ISI) = {np.max(np.abs(isi_taps)):.2e}")

# %% Transmitter
up = np.zeros(N * SPS); up[::SPS] = sym    # explicit upsampling
tx = np.convolve(up, h)                    # pulse shaping, length N*sps + L - 1
print(f"\nTX waveform length = {len(tx)} samples, measured Es = {np.sum(tx**2) / N:.4f} (target 1)")

# %% AWGN + matched filter (reference link, no channel)
y_ref, _, r_ref, s2_ref = run_link(sym, h, SPS, None, ESN0_REF, NOISE_SEED)
print(f"Reference link: Es/N0 = {ESN0_REF} dB -> sigma^2 = {s2_ref:.4f} per sample")
y_clean = np.convolve(np.convolve(up, h), h_mf)     # noise-free matched-filter output

# %% Figure 1: filter, cascade pulse and delay
fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
ax = axs[0]
ax.stem(np.arange(Ltx) / SPS, h, basefmt=" ", linefmt="C0-", markerfmt="C0o")
ax.plot(Dtx / SPS, h[Dtx], "rs", ms=9, mfc="none", label=f"centre tap = group delay {Dtx} samples")
ax.set(xlabel="Time (symbol periods, T)", ylabel="Amplitude (unit-energy filter taps)",
       title=f"RRC filter, beta={BETA:.2f}, span={SPAN}, sps={SPS}, L={Ltx} taps")
ax.grid(alpha=0.4); ax.legend(["h_tx[n]", f"centre tap = group delay {Dtx} samples"], loc="upper right")
ax = axs[1]
ax.plot(np.arange(len(g)) / SPS, g, "b-", label="cascade pulse")
ax.plot((Dtot + np.arange(-SPAN, SPAN + 1) * SPS) / SPS, g[Dtot + np.arange(-SPAN, SPAN + 1) * SPS],
        "ko", ms=4, label="values at symbol instants")
ax.axvline(Dtot / SPS, color="r", ls="--", label=f"total delay = {Dtot} samples = {Dtot // SPS} T")
ax.set(xlabel="Time (symbol periods, T)", ylabel="Amplitude", title="Cascade h_tx * h_mf (raised cosine)")
ax.grid(alpha=0.4); ax.legend(loc="upper right")
finish(fig, "fig01_filter_and_delay")

# %% Figure 2: transmitted signal and matched-filter output
n_show = 16
t_tx = np.arange((n_show + SPAN) * SPS) / SPS
t_dec = Dtot / SPS + np.arange(n_show)
fig, axs = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
ax = axs[0]
ax.stem(np.arange(n_show), sym[:n_show], basefmt=" ", linefmt="C1-", markerfmt="C1o", label="BPSK symbols a_k (at kT)")
ax.plot(t_tx, tx[:len(t_tx)], "b-", label="s(t) after RRC TX filter")
ax.set(xlabel="Time (T)", ylabel="Amplitude",
       title="Transmitter: BPSK symbols and RRC pulse-shaped waveform (delayed by span/2 symbols)")
ax.grid(alpha=0.4); ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax = axs[1]
ax.plot(t_tx, y_clean[:len(t_tx)], "b-", label="MF output y(t)")
ax.plot(t_dec, y_clean[Dtot + np.arange(n_show) * SPS], "ro", label="samples at D_tot + k*sps")
ax.vlines(t_dec, 0, sym[:n_show], color="k", lw=0.8, label="a_k placed at the decision instants")
ax.set(xlabel="Time (T)", ylabel="Amplitude",
       title=f"Noise-free matched-filter output: decisions at t = {Dtot // SPS} T + kT hit exactly +/-1")
ax.grid(alpha=0.4); ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax = axs[2]
ax.plot(t_tx, y_ref[:len(t_tx)], "b-", label="MF output")
ax.plot(t_dec, y_ref[Dtot + np.arange(n_show) * SPS], "ro", label="decision samples")
ax.axhline(0, color="k", ls=":", label="threshold")
ax.set(xlabel="Time (T)", ylabel="Amplitude",
       title=f"Matched-filter output with AWGN, Es/N0 = {ESN0_REF} dB (threshold = 0)")
ax.grid(alpha=0.4); ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
finish(fig, "fig02_tx_and_mf_output")

# %% Figure 3: received signal before and after matched filtering
y_b, _, r_b, _ = run_link(sym, h, SPS, None, 5, NOISE_SEED)
fig, axs = plt.subplots(2, 1, figsize=(11, 5.5), constrained_layout=True)
ax = axs[0]
ax.plot(t_tx, r_b[:len(t_tx)], color="0.5", label="noisy received signal")
ax.plot(t_tx, tx[:len(t_tx)], "b-", lw=1.2, label="noise-free s(t)")
ax.set(xlabel="Time (T)", ylabel="Amplitude", title="Before matched filter: r(t) = s(t) + n(t), Es/N0 = 5 dB")
ax.grid(alpha=0.4); ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax = axs[1]
ax.plot(t_tx, y_b[:len(t_tx)], "b-", label="MF output y(t)")
ax.plot(t_dec, y_b[Dtot + np.arange(n_show) * SPS], "ro", label="decision samples")
ax.axhline(0, color="k", ls=":", label="threshold")
ax.set(xlabel="Time (T)", ylabel="Amplitude", title="After matched filter: noise averaged out, decisions at D_tot + kT")
ax.grid(alpha=0.4); ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
finish(fig, "fig03_before_after_mf")

# %% Mandatory delay validation
yV, _, _, s2V = run_link(sym, h, SPS, None, ESN0_VAL, NOISE_SEED)
# (1) sampling BEFORE delay compensation: first decision at sample index 0
r0 = yV[np.arange(N) * SPS];            bits0 = (r0 > 0).astype(int)
ber0 = np.mean(bits0 != bits)
# (2) sampling AFTER delay compensation: first decision at index Dtot
r1 = yV[Dtot + np.arange(N) * SPS];     bits1 = (r1 > 0).astype(int)
ber1 = np.mean(bits1 != bits)
# (3) diagnosis: uncompensated decisions are the data delayed by SPAN symbols
ber0_shift = np.mean(bits0[SPAN:] != bits[:-SPAN])
print(f"\n---------------- DELAY VALIDATION (Es/N0 = {ESN0_VAL} dB) ----------------")
print(f"Sampling at index 0+k*sps              : errors = {np.sum(bits0 != bits):6d} / {N}, BER = {ber0:.4f}")
print(f"Sampling at index D_tot+k*sps (D={Dtot}) : errors = {np.sum(bits1 != bits):6d} / {N}, BER = {ber1:.3e}")
print(f"Uncompensated decisions vs bits delayed by {SPAN} symbols: BER = {ber0_shift:.3e}")
print(f"Theoretical AWGN BER at {ESN0_VAL} dB = {q_ber(ESN0_VAL)[0]:.3e}")

lags = np.arange(0, 2 * Dtot + 1)          # brute-force search of the first-decision index
ber_lag = np.zeros(len(lags))
for i, lag in enumerate(lags):
    idx = lag + np.arange(N) * SPS
    ok = idx < len(yV)
    ber_lag[i] = np.mean((yV[idx[ok]] > 0).astype(int) != bits[ok])
i_best = int(np.argmin(ber_lag))
print(f"Lag search: lowest BER {ber_lag[i_best]:.3e} at lag = {lags[i_best]} samples (calculated D_tot = {Dtot})")

n_seq = 30
k = np.arange(1, n_seq + 1)
fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), constrained_layout=True)
ax = axs[0]
ax.plot(lags, ber_lag, "b.-", label="BER at each lag")
ax.axvline(Dtot, color="r", ls="--", lw=1.3, label=f"D_tot = {Dtot} samples")
ax.set(xlabel="Sampling start index (lag, samples)", ylabel="BER",
       title=f"BER vs first-decision lag (Es/N0 = {ESN0_VAL} dB): only lag = D_tot works")
ax.grid(alpha=0.4); ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
for ax, bb, tt, col in [(axs[1], bits0, f"BEFORE delay compensation: BER = {ber0:.3f} "
                         f"(decisions are the data delayed by {SPAN} symbols)", "r"),
                        (axs[2], bits1, f"AFTER delay compensation (start at sample D_tot): BER = {ber1:.2e}", "g")]:
    s_hat = 2 * bb - 1
    ax.stem(k, sym[:n_seq], basefmt=" ", linefmt="0.3", markerfmt="ko", label="transmitted a_k")
    ax.plot(k, s_hat[:n_seq], marker="o", ls="none", mfc="none", mec=col, ms=9, label="detected")
    err = np.where(s_hat[:n_seq] != sym[:n_seq])[0]
    if len(err):
        ax.plot(k[err], s_hat[:n_seq][err], "rx", ms=10, mew=1.5, label="errors")
    ax.set(ylim=(-1.6, 1.6), xlabel="Symbol index k", ylabel="Symbol value", title=tt)
    ax.grid(alpha=0.4); ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
finish(fig, "fig04_delay_validation")

# %% Eye diagrams: effect of SNR
fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for ax, snr in zip(axs.ravel(), SNR_EYES):
    y, *_ = run_link(sym, h, SPS, None, snr, NOISE_SEED)
    rk, v = sample_symbols(y, Dtot, SPS, N)
    eh = eye_height_3sigma(rk, sym[v])
    plot_eye(ax, y, Dtot, SPS, NUM_TRACES, SPAN + 1, f"Es/N0 = {snr} dB,  EH_3sigma = {eh:.2f}")
finish(fig, "fig05_eye_vs_snr")

# %% SNR sweep: eye height and BER vs SNR
eh_sweep = np.zeros(len(SNR_SWEEP)); eh_theory = np.zeros(len(SNR_SWEEP))
for i, snr in enumerate(SNR_SWEEP):
    y, *_ = run_link(sym, h, SPS, None, snr, NOISE_SEED)
    rk, v = sample_symbols(y, Dtot, SPS, N)
    eh_sweep[i] = eye_height_3sigma(rk, sym[v])
    eh_theory[i] = 2 - 6 * np.sqrt(1 / (2 * 10 ** (snr / 10)))     # 2 - 6 sigma, Es = 1
ber_snr = np.zeros(len(SNR_BER))
for i, snr in enumerate(SNR_BER):
    y, *_ = run_link(sym, h, SPS, None, snr, NOISE_SEED)
    rk, v = sample_symbols(y, Dtot, SPS, N)
    ber_snr[i] = np.mean((rk > 0).astype(int) != bits[v])
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
ax = axs[0]
ax.plot(SNR_SWEEP, eh_sweep, "bo-", label="simulation")
ax.plot(SNR_SWEEP, eh_theory, "r--", label=r"theory $2 - 6\sigma$")
ax.axhline(2, color="k", ls=":", label="ideal noise-free value 2")
ax.set(xlabel="Es/N0 (dB)", ylabel="Eye height (amplitude, unit-less)",
       title=r"Eye height vs SNR  ($[\mu_+ - 3\sigma_+] - [\mu_- + 3\sigma_-]$)")
ax.grid(alpha=0.4); ax.legend(loc="lower right")
ax = axs[1]
ax.semilogy(SNR_BER, q_ber(SNR_BER), "r--", label="theory 0.5 erfc(sqrt(Es/N0))")
ax.semilogy(SNR_BER, np.maximum(ber_snr, 0.5 / N), "bo", label="simulation (0 errors plotted at 0.5/N)")
ax.set(xlabel="Es/N0 (dB)", ylabel="Bit error rate", title="BER vs SNR: simulation vs theory")
ax.grid(alpha=0.4, which="both"); ax.legend(loc="lower left")
finish(fig, "fig06_eyeheight_ber_vs_snr")

# %% Roll-off sweep
n_off = np.arange(-(SPS // 2), SPS // 2 + 1)      # timing offsets in samples
off_t = n_off / SPS
eh_an_beta = np.zeros((len(BETA_LIST), len(n_off)))
eh_meas_b = np.zeros(len(BETA_LIST))
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
for ib, b in enumerate(BETA_LIST):
    hb = rrc_filter(b, SPAN, SPS)
    y, *_ = run_link(sym, hb, SPS, None, ESN0_ROLLSPAN, NOISE_SEED)
    rk, v = sample_symbols(y, Dtot, SPS, N)
    eh_meas_b[ib] = eye_height_3sigma(rk, sym[v])
    gb = np.convolve(hb, hb[::-1])
    for io, o in enumerate(n_off):
        eh_an_beta[ib, io] = eye_height_analytic(gb, Dtot, SPS, o)
    plot_eye(axs[ib], y, Dtot, SPS, NUM_TRACES, SPAN + 1,
             f"beta = {b:.1f}, Es/N0 = {ESN0_ROLLSPAN} dB, EH_3sigma = {eh_meas_b[ib]:.2f}")
finish(fig, "fig07_eye_vs_rolloff")

n_fft = 8192
f_ax = np.arange(n_fft // 2) / n_fft * SPS            # frequency in units of 1/T
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
for ib, b in enumerate(BETA_LIST):
    hb = rrc_filter(b, SPAN, SPS)
    axs[0].plot(np.arange(-SPAN * SPS // 2, SPAN * SPS // 2 + 1) / SPS, hb, lw=1.2, label=f"beta = {b:.1f}")
    Hf = np.abs(np.fft.fft(hb, n_fft))[: n_fft // 2]
    axs[1].plot(f_ax, 20 * np.log10(Hf / Hf[0] + 1e-12), lw=1.2, label=f"beta = {b:.1f}")
    axs[2].plot(off_t, eh_an_beta[ib], "o-", lw=1.2, label=f"beta = {b:.1f}")
axs[0].set(xlabel="Time (T)", ylabel="Amplitude", title="RRC impulse responses")
axs[1].set(xlabel="Frequency (1/T)", ylabel="|H_RRC(f)| (dB, rel. to DC)",
           title="RRC spectra: edge at (1+beta)/2T", xlim=(0, 1.5), ylim=(-80, 5))
axs[2].set(xlabel="Sampling offset (T)", ylabel="Worst-case ISI-limited eye height",
           title="Noise-free worst-case eye height vs offset")
for a in axs:
    a.grid(alpha=0.4); a.legend()
finish(fig, "fig08_rolloff_pulse_spectrum_timing")

# %% Span sweep
eh_an_span = np.zeros(len(SPAN_LIST))
fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for ax, (i_s, sp) in zip(axs.ravel(), enumerate(SPAN_LIST)):
    hs = rrc_filter(BETA, sp, SPS); ds = sp * SPS          # delay recomputed for each span
    y, *_ = run_link(sym, hs, SPS, None, ESN0_ROLLSPAN, NOISE_SEED)
    gs = np.convolve(hs, hs[::-1])
    eh_an_span[i_s] = eye_height_analytic(gs, ds, SPS, 0)
    plot_eye(ax, y, ds, SPS, NUM_TRACES, max(sp + 1, 2),
             f"span = {sp} symbols (D_tot = {ds} samples), worst-case EH = {eh_an_span[i_s]:.2f}")
finish(fig, "fig09_eye_vs_span")

# %% Timing offset / BER
yT, *_ = run_link(sym, h, SPS, None, ESN0_TIMING, NOISE_SEED)     # noisy link for BER
y20, *_ = run_link(sym, h, SPS, None, ESN0_EHOFF, NOISE_SEED)     # high-SNR link for eye height
ber_off = np.zeros(len(n_off)); eh_off3 = np.zeros(len(n_off)); eh_off_an = np.zeros(len(n_off))
for io, o in enumerate(n_off):
    rk, v = sample_symbols(yT, Dtot, SPS, N, o)
    ber_off[io] = np.mean((rk > 0).astype(int) != bits[v])      # equal-length sequences by construction
    rk, v = sample_symbols(y20, Dtot, SPS, N, o)
    eh_off3[io] = eye_height_3sigma(rk, sym[v])
    eh_off_an[io] = eye_height_analytic(g, Dtot, SPS, o)
i_min = int(np.argmin(ber_off))
print(f"\n---------------- TIMING OFFSET (Es/N0 = {ESN0_TIMING} dB) ----------------")
print(f"offset(T)   BER        EH_3sigma({ESN0_EHOFF}dB)  EH_worst-case(noise-free)")
for io in range(len(n_off)):
    print(f"{off_t[io]:+7.3f}   {ber_off[io]:.3e}   {eh_off3[io]:8.3f}        {eh_off_an[io]:8.3f}")
print(f"Minimum BER = {ber_off[i_min]:.3e} at offset = {off_t[i_min]:+.3f} T;  AWGN theory = {q_ber(ESN0_TIMING)[0]:.3e}")

show_off = [-SPS // 2, -SPS // 4, 0, SPS // 4, SPS // 2]
fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
for ax, o in zip(axs.ravel(), show_off):
    io = int(np.where(n_off == o)[0][0])
    plot_eye(ax, yT, Dtot, SPS, NUM_TRACES, SPAN + 1, f"offset = {off_t[io]:+.2f} T: BER = {ber_off[io]:.2e}", mark_t=o / SPS)
axs.ravel()[-1].axis("off")
finish(fig, "fig10_eye_vs_sampling_offset")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
ax = axs[0]
ax.semilogy(off_t, np.maximum(ber_off, 0.5 / N), "bo-", label="simulation")
ax.axhline(q_ber(ESN0_TIMING)[0], color="r", ls="--", label="AWGN-only theory at offset 0")
ax.set(xlabel="Sampling offset (fraction of T)", ylabel="Bit error rate", title=f"BER vs timing offset, Es/N0 = {ESN0_TIMING} dB")
ax.grid(alpha=0.4, which="both"); ax.legend(loc="lower center")
ax = axs[1]
ax.plot(off_t, eh_off3, "bo-", label=f"EH_3sigma at {ESN0_EHOFF} dB")
ax.plot(off_t, eh_off_an, "rs--", label="noise-free worst case (analytic)")
ax.axhline(0, color="k", ls=":")
ax.set(xlabel="Sampling offset (fraction of T)", ylabel="Eye height (amplitude)", title="Eye height vs timing offset")
ax.grid(alpha=0.4); ax.legend(loc="lower center")
finish(fig, "fig11_ber_eyeheight_vs_offset")

# %% Multipath ISI
hch = np.zeros(CH_DELAY.max() + 1)
hch[CH_DELAY] = CH_TAPS                    # taps at their delays (first path has zero delay)
hch = hch / np.sqrt(np.sum(hch ** 2))      # unit energy: channel does not change Es, only creates ISI
yNo, *_ = run_link(sym, h, SPS, None, ESN0_MP, NOISE_SEED)
yMP, *_ = run_link(sym, h, SPS, hch, ESN0_MP, NOISE_SEED)
yMP0, _, rMP_clean, _ = run_link(sym, h, SPS, hch, np.inf, NOISE_SEED)    # noise-free
gMP = np.convolve(np.convolve(h, hch), h_mf)        # overall pulse TX * channel * RX
iMPpk = int(np.argmax(gMP))
rkN, vN = sample_symbols(yNo, Dtot, SPS, N);   eh_no = eye_height_3sigma(rkN, sym[vN])
rkM, vM = sample_symbols(yMP, Dtot, SPS, N);   eh_mp = eye_height_3sigma(rkM, sym[vM])
rk0, v0 = sample_symbols(yMP0, Dtot, SPS, N)
eh_mp_worst = eye_height_worst(rk0, sym[v0])
eh_mp_an = eye_height_analytic(gMP, Dtot, SPS, 0)
print("\n---------------- MULTIPATH ISI ----------------")
print(f"Channel taps (normalised) at delays {CH_DELAY.tolist()} samples: {np.round(hch[CH_DELAY], 3).tolist()}")
print(f"Cascade pulse (TX*channel*RX) peak index = {iMPpk}, sampling index = Dtot = {Dtot}")
print(f"  (first path has zero delay, so Dtot is still the reference; later paths shift the peak by "
      f"{iMPpk - Dtot} samples = {(iMPpk - Dtot) / SPS:.3f} T)")
print(f"Eye height (3 sigma, {ESN0_MP} dB): no multipath = {eh_no:.3f},  multipath = {eh_mp:.3f}")
print(f"Noise-free eye height with multipath: simulated worst case = {eh_mp_worst:.3f}, analytic = {eh_mp_an:.3f}")
print(f"Main-tap amplitude at decision instant = {gMP[Dtot]:.3f}, sum|ISI| = {(2 * gMP[Dtot] - eh_mp_an) / 2:.3f}")

ber_no = np.zeros(len(SNR_MP)); ber_mp = np.zeros(len(SNR_MP))
for i, snr in enumerate(SNR_MP):
    yA, *_ = run_link(sym, h, SPS, None, snr, NOISE_SEED)
    yB, *_ = run_link(sym, h, SPS, hch, snr, NOISE_SEED)
    rk, v = sample_symbols(yA, Dtot, SPS, N); ber_no[i] = np.mean((rk > 0).astype(int) != bits[v])
    rk, v = sample_symbols(yB, Dtot, SPS, N); ber_mp[i] = np.mean((rk > 0).astype(int) != bits[v])

fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
ax = axs[0, 0]
ax.plot(t_tx, tx[:len(t_tx)], "b-", lw=1.2, label="TX waveform s(t)")
ax.plot(t_tx, rMP_clean[:len(t_tx)], "r-", lw=1.2, label="after multipath channel")
ax.set(xlabel="Time (T)", ylabel="Amplitude", title="Received waveform: before and after multipath (noise-free)")
ax.grid(alpha=0.4); ax.legend(loc="upper right")
plot_eye(axs[0, 1], yNo, Dtot, SPS, NUM_TRACES, SPAN + 1, f"No multipath, Es/N0 = {ESN0_MP} dB, EH_3sigma = {eh_no:.2f}")
plot_eye(axs[1, 0], yMP, Dtot, SPS, NUM_TRACES, SPAN + 1, f"Multipath ISI, Es/N0 = {ESN0_MP} dB, EH_3sigma = {eh_mp:.2f}")
ax = axs[1, 1]
ax.semilogy(SNR_MP, q_ber(SNR_MP), "k--", label="AWGN theory")
ax.semilogy(SNR_MP, np.maximum(ber_no, 0.5 / N), "bo-", label="RRC + MF, no multipath")
ax.semilogy(SNR_MP, np.maximum(ber_mp, 0.5 / N), "rs-", label="RRC + MF, multipath")
ax.set(xlabel="Es/N0 (dB)", ylabel="Bit error rate", title="BER with and without multipath ISI")
ax.grid(alpha=0.4, which="both"); ax.legend(loc="lower left")
finish(fig, "fig12_multipath_isi")

# %% Summary
print("\n==================== SUMMARY ====================")
print(f"Roll-off study (Es/N0 = {ESN0_ROLLSPAN} dB): EH_3sigma at offset 0 | worst-case EH at -0.25T, 0, +0.25T")
iq, i0, ip = (int(np.where(n_off == o)[0][0]) for o in (-SPS // 4, 0, SPS // 4))
for ib, b in enumerate(BETA_LIST):
    print(f"  beta = {b:.1f} : {eh_meas_b[ib]:.3f} | {eh_an_beta[ib, iq]:.3f}  {eh_an_beta[ib, i0]:.3f}  {eh_an_beta[ib, ip]:.3f}")
print(f"Span study (beta = {BETA:.2f}): worst-case EH at offset 0")
for sp, e in zip(SPAN_LIST, eh_an_span):
    print(f"  span = {sp:2d} : {e:.3f}")
print(f'Figures saved in folder "{OUT_DIR}" (SAVE_FIGS = {SAVE_FIGS})')

if SHOW_FIGS and not IN_NOTEBOOK:
    plt.show()
