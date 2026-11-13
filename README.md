# Polymarket NVIDIA CEO Forecaster v3

Forecasts the Yes price of every trading market of Polymarket's event
"Next CEO of NVIDIA?" — minute by minute, 4 hours ahead (240 minutes) —
as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per market per minute.

## Method

Closed-form, deterministic, stdlib only. No Monte Carlo, no network.

- **Median = the checkpoint price.** Returns have no momentum (1-min
  autocorrelation ≈ −0.06), so the best median is the last price.
- **Per-market volatility from the RECENT regime only.** Volatility
  changed ~10× between Nov 10–11 (wild) and Nov 12–13 (calm), so a
  24h/72h window is stale. Two components, blended 50/50 in variance
  space:
  - **Garman–Klass range vol** — 5-min bars (O/H/L/C built from the
    1-min closes) over the trailing 360 min, recency-weighted
    (half-life 60 min). The high–low range carries far more information
    about volatility than close-to-close changes alone. Horizon shape
    σ(h) = σ₅·(h/5)^0.3.
  - **Ornstein–Uhlenbeck (mean-reverting) vol** — sub-diffusive horizon
    shape, θ = 0.015/min (half-life ~80 min). The baseline's √h is ~2×
    too wide at h = 240.
- **Mixture quantile shape** z = [−1.90, −0.55, 0, +0.55, +1.90]
  (narrow core, fat tails) — matches the kurtosis (~15.5) of 240-min
  price changes better than Gaussian.
- **Legal rolling CQR pad** on the 0.05/0.95 quantiles — computed only
  from the case's own price history (12 pseudo-checkpoints, 90th
  percentile of the conformity score, 31-min smoothing). Jump insurance:
  it raises 90% coverage without costing pinball loss. A
  pseudo-checkpoint skips only the markets with < 200 prices before it
  (a single young market must not veto the pad for everyone — the old
  all-or-nothing rule silently disabled the pad for 2 of 6 practice
  cases); if no market qualifies, the checkpoint is skipped.
- **Floors and guards:** σ₅ ≥ 0.0003, σ(h) ≥ 0.0005; markets with
  < 120 min history are shrunk 50% toward the cross-market median σ;
  the opening regime (first 60 min after a market's start) is excluded
  from both vol estimates.

Deliberately NOT used (all measured, all hurt or were artefacts):
sum-to-1 renormalisation, median mean-reversion, uniform widening,
compound-Poisson jump variance, Student-t quantiles, cross-market
lead-lag, MAD vol, tail ramp.

## Practice results (6 cases, local scorer — reproduces the house baseline exactly)

| version | pooled pinball loss | 90% coverage |
|---|---|---|
| house baseline | 0.363589 | 0.9496 |
| v1 (EWMA + OU, Gaussian) | 0.300215 | 0.8198 |
| **v3 (this entry)** | **0.288058** | **0.8643** |

Lower pinball loss is better. Runtime ~0.18 s/case (limit 8 s).

## Run

```sh
python3 main.py
```

Reads `$MAKEFUN_INPUT_DIR` (else `/input`) — every `*.in` file, in name
order — and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).
Output JSON has exactly one key, `forecasts`: one row per market
(`{"id", "q"}`), `q` = 240 rows × 5 quantiles, each finite in [0, 1],
non-decreasing, 6 decimals.

## License

MIT. See `LICENSE`.
