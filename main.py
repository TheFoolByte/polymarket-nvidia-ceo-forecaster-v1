#!/usr/bin/env python3
"""Polymarket NVIDIA CEO forecaster v3.

Forecasts the Yes price of every trading market of Polymarket's
"Next CEO of NVIDIA?" event, minute by minute, 4 hours ahead (240 min),
as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

Method (research waves 1-3, all measurements on practice p00-p05):
- Median = checkpoint price (no drift; 1-min return autocorr is ~-0.06,
  trend slopes are negative).
- Per-market sigma from RECENT vol only, two components blended in
  variance space (0.5/0.5):
  * GK: Garman-Klass range vol from 5-min bars (O/H/L/C from the 1-min
    closes) over the trailing 360 min, bars recency-weighted with a
    60-min half-life; sigma5 = x * sqrt(max(GK, 0)); the horizon shape
    is sigma_GK(h) = sigma5 * (h/5)^0.3.
  * OU: S_inf = std_60(trailing 6h) / sqrt(1-exp(-2*0.015*60));
    sigma_OU(h) = S_inf * sqrt(1-exp(-2*0.015*h)).
  * Blend: sigma(h) = sqrt(0.5*sigma_GK^2 + 0.5*sigma_OU^2).
- Quantile shape: mixture z = [-1.90, -0.55, 0, +0.55, +1.90] (narrow
  core, fat tails; the 240-min change distribution has kurtosis 15.5).
- Rolling CQR pad on the 0.05/0.95 quantiles, computed ONLY from the
  case's own price history (no answers): 12 pseudo-checkpoints every
  60 min over the trailing 12h; conformity score s = max(q05-y, y-q95)
  at each horizon h, using only pseudo-checkpoints with offset >= h;
  pad[h] = 90th percentile of s pooled over markets and checkpoints,
  smoothed with a 31-min moving average. The pseudo-forecasts use the
  v1 Gaussian forecaster (EWMA+OU), exactly as the measured reference.
  If no scores exist for a horizon, pad = 0.
- Floors: sigma5 >= 0.0003, sigma(h) >= 0.0005.
- Short-history markets (<120 min of prices): shrink sigma 50% toward
  the cross-market median sigma(h).
- Opening regime excluded: every market opened at 0.485 then collapsed;
  changes/bars starting within the first 60 min after a market's start
  are dropped from both vol estimates.
- Closed-form, no Monte Carlo. Deterministic (no randomness, no time
  dependence, markets in input order, sorted JSON keys).

Practice (6 cases, local pinball scorer reproducing the house baseline
exactly, 0.363589): pooled 0.289148, p04+p05 0.157176, coverage 0.8427
(v1: 0.300215 / 0.155441 / 0.8198). Runtime ~0.11-0.21 s/case (cap 8 s).

Tunable via env vars (defaults are v3): MF_GK_WINDOW (360), MF_GK_HL (60).
The 540-min safer variant (pooled 0.293519, p04+p05 0.155334) is
MF_GK_WINDOW=540.
"""
import glob
import json
import math
import os

Z_MIX = [-1.90, -0.55, 0.0, 0.55, 1.90]
Z_GAUSS = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]

GK_WINDOW = int(os.environ.get("MF_GK_WINDOW", "360"))
GK_HL = float(os.environ.get("MF_GK_HL", "60"))
HL = 60.0
ALPHA = 0.3
THETA = 0.015
BLEND = 0.5
OPEN_EXCL = 60
W_EWMA_MIN = 1440   # EWMA window (pseudo-forecaster only)
W_OU_MIN = 360      # OU window in minutes
SIGMA5_FLOOR = 0.0003
SIGMA_FLOOR = 0.0005
SHORT_N = 120       # shrink markets with fewer than this many prices
SHRINK = 0.5        # weight kept on own sigma (rest -> median)
N_PSEUDO = 12
PSEUDO_STEP = 60
CQR_Q = 0.90
CQR_SMOOTH = 31     # moving-average window for the pad (minutes)


def clamp6(v):
    r = round(v, 6)
    if r <= 0.0:
        return 0.0
    if r >= 1.0:
        return 1.0
    return r


def pstdev(vals):
    n = len(vals)
    if n == 0:
        return 0.0
    mu = sum(vals) / n
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / n)


def sigma_ewma(p, h_list):
    """v1 EWMA: weighted std of 5-min changes over the last 24h.
    Used only for the CQR pseudo-forecasts (the v1 Gaussian forecaster)."""
    n = len(p)
    start = max(0, n - (W_EWMA_MIN + 1))
    q = p[start:]
    lam = 0.5 ** (1.0 / HL)
    vals = []
    wts = []
    for i in range(0, len(q) - 5):
        if start + i < OPEN_EXCL:
            continue
        vals.append(q[i + 5] - q[i])
        age = (n - 1) - (start + i + 5)
        wts.append(lam ** age)
    if not vals:
        s5 = 0.0
    else:
        sw = sum(wts)
        mu = sum(v * w for v, w in zip(vals, wts)) / sw
        var = sum(w * (v - mu) ** 2 for v, w in zip(vals, wts)) / sw
        s5 = math.sqrt(var)
    s5 = max(s5, SIGMA5_FLOOR)
    return [s5 * ((h / 5.0) ** ALPHA) for h in h_list]


def sigma_ou(p, h_list):
    """OU horizon shape from std of 60-min changes over trailing 6h."""
    n = len(p)
    start = max(0, n - (W_OU_MIN + 1))
    q = p[start:]
    ch = []
    for j in range(0, len(q) - 60, 5):
        if start + j < OPEN_EXCL:
            continue
        ch.append(q[j + 60] - q[j])
    s60 = pstdev(ch) if len(ch) > 5 else 0.0
    den = math.sqrt(1.0 - math.exp(-2.0 * THETA * 60.0))
    s_inf = s60 / den if den > 0 else s60
    return [s_inf * math.sqrt(1.0 - math.exp(-2.0 * THETA * h)) for h in h_list]


def sigma_gk(p, h_list, window_min=GK_WINDOW, hl=GK_HL):
    """Garman-Klass range vol from 5-min bars over the trailing window.
    Bars are recency-weighted with half-life hl (None = unweighted).
    Returns sigma_GK(h) = sigma5 * (h/5)^0.3."""
    n = len(p)
    start = max(0, n - window_min - 1)
    q = p[start:]
    x = p[-1] if p else 0.0
    vals = []
    wts = []
    for j in range(0, len(q) - 5, 5):
        if start + j < OPEN_EXCL:
            continue
        bar = q[j:j + 6]
        o, hi, lo, c = bar[0], max(bar), min(bar), bar[-1]
        if o <= 0 or lo <= 0 or hi <= 0 or c <= 0:
            continue
        vals.append(0.5 * (math.log(hi / lo)) ** 2
                    - (2.0 * math.log(2.0) - 1.0) * (math.log(c / o)) ** 2)
        age = (n - 1) - (start + j + 5)
        wts.append(0.5 ** (age / hl) if hl else 1.0)
    if not vals:
        s5 = 0.0
    else:
        sw = sum(wts)
        v = sum(a * b for a, b in zip(vals, wts)) / sw
        s5 = x * math.sqrt(max(v, 0.0))
    s5 = max(s5, SIGMA5_FLOOR)
    return [s5 * ((h / 5.0) ** ALPHA) for h in h_list]


def _median_sigma(sig, horizon):
    med = []
    for i in range(horizon):
        vals = sorted(sig[mid][i] for mid in sig)
        k = len(vals)
        med.append(vals[k // 2] if k % 2 == 1
                   else 0.5 * (vals[k // 2 - 1] + vals[k // 2]))
    return med


def _forecast_rows(markets, sig, med, z, horizon):
    out = {}
    for m in markets:
        p = m["prices"]
        x = p[-1]
        s = sig[m["id"]]
        if len(p) < SHORT_N:
            s = [SHRINK * v + (1.0 - SHRINK) * med[i]
                 for i, v in enumerate(s)]
        rows = []
        for i in range(horizon):
            row = [clamp6(x + zz * s[i]) for zz in z]
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
            rows.append(row)
        out[m["id"]] = rows
    return out


def forecast_case(case, window_min=GK_WINDOW, hl=GK_HL, z=Z_MIX):
    """The v3 forecast: GK vol + OU blend, mixture z."""
    markets = case["markets"]
    horizon = int(case.get("horizon_minutes", 240))
    h_list = list(range(1, horizon + 1))
    sig = {}
    for m in markets:
        p = m["prices"]
        se = sigma_gk(p, h_list, window_min, hl)
        so = sigma_ou(p, h_list)
        sig[m["id"]] = [
            math.sqrt(BLEND * se[i] ** 2 + (1.0 - BLEND) * so[i] ** 2)
            for i in range(horizon)
        ]
    for mid in sig:
        sig[mid] = [max(v, SIGMA_FLOOR) for v in sig[mid]]
    med = _median_sigma(sig, horizon)
    return _forecast_rows(markets, sig, med, z, horizon)


def forecast_case_gauss(case):
    """The v1 Gaussian forecaster (EWMA+OU). Used only for the CQR
    pseudo-forecasts, matching the measured reference exactly."""
    markets = case["markets"]
    horizon = int(case.get("horizon_minutes", 240))
    h_list = list(range(1, horizon + 1))
    sig = {}
    for m in markets:
        p = m["prices"]
        se = sigma_ewma(p, h_list)
        so = sigma_ou(p, h_list)
        sig[m["id"]] = [
            math.sqrt(BLEND * se[i] ** 2 + (1.0 - BLEND) * so[i] ** 2)
            for i in range(horizon)
        ]
    for mid in sig:
        sig[mid] = [max(v, SIGMA_FLOOR) for v in sig[mid]]
    med = _median_sigma(sig, horizon)
    return _forecast_rows(markets, sig, med, Z_GAUSS, horizon)


def cqr_pad_legal(case):
    """Rolling conformal pad, computed ONLY from the case's own prices.
    12 pseudo-checkpoints every 60 min over the trailing 12h. A
    pseudo-checkpoint is skipped entirely if any market has fewer than
    200 prices before it. For horizon h only pseudo-checkpoints with
    offset >= h contribute (their realisation lies inside the case's
    prices). pad[h] = 90th percentile of the conformity scores
    s = max(q05 - y, y - q95), smoothed with a 31-min moving average.
    No cap (matches the measured reference)."""
    markets = case["markets"]
    horizon = int(case.get("horizon_minutes", 240))
    prices_by_id = {m["id"]: m["prices"] for m in markets}
    scores = {h: [] for h in range(1, horizon + 1)}
    for k in range(1, N_PSEUDO + 1):
        off = k * PSEUDO_STEP
        pseudo_markets = []
        ok = True
        for mid, p in prices_by_id.items():
            if len(p) - 1 - off < 200:
                ok = False
                break
            pseudo_markets.append({"id": mid, "prices": p[:len(p) - off]})
        if not ok:
            continue
        pseudo = {"markets": pseudo_markets,
                  "horizon_minutes": horizon}
        try:
            fc = forecast_case_gauss(pseudo)
        except Exception:
            continue
        for mid, p in prices_by_id.items():
            n_m = len(p)
            for h in range(1, min(off, horizon) + 1):
                y = p[n_m - 1 - off + h]
                row = fc[mid][h - 1]
                scores[h].append(max(row[0] - y, y - row[4]))
    pad = []
    for h in range(1, horizon + 1):
        s = sorted(scores[h])
        if not s:
            pad.append(0.0)
            continue
        qi = min(len(s) - 1, int(math.ceil(CQR_Q * len(s))) - 1)
        pad.append(max(0.0, s[qi]))
    half = CQR_SMOOTH // 2
    sm = []
    for h in range(horizon):
        lo = max(0, h - half)
        hi = min(horizon, h + half + 1)
        sm.append(sum(pad[lo:hi]) / (hi - lo))
    return sm


def apply_pad(rows_by_id, pad):
    out = {}
    for mid, rows in rows_by_id.items():
        nr = []
        for i, row in enumerate(rows):
            r = list(row)
            r[0] = clamp6(r[0] - pad[i])
            r[4] = clamp6(r[4] + pad[i])
            for j in range(1, 5):
                if r[j] < r[j - 1]:
                    r[j] = r[j - 1]
            nr.append(r)
        out[mid] = nr
    return out


def forecast_full(case):
    """v3: GK + mixture z, then the legal CQR pad on the outer quantiles."""
    fc = forecast_case(case)
    pad = cqr_pad_legal(case)
    fc = apply_pad(fc, pad)
    markets = case["markets"]
    return {"forecasts": [{"id": m["id"], "q": fc[m["id"]]}
                          for m in markets]}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        cid = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_full(case)
        with open(os.path.join(out_dir, cid + ".out"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps(result, sort_keys=True,
                               separators=(",", ":")))


if __name__ == "__main__":
    main()
