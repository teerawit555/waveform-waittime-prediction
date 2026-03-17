from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Config (minimal, mostly for windowing / robustness only)
# ============================================================
CFG = {
    "WINDOW_MS": 10.0,
    "MIN_PTS": 50,
    "START_MS": 0.4,
    "TAIL_MS": 2.0,
    "TAIL_WIN_MS": 0.6,
    "CLIP_LIMIT": 12.0,
    "FAST_MS": 0.1,
    "MIN_OUT_MS": 0.1,
}

CLIP_FEATS = [
    "shape_overshoot_norm",
    "shape_rebound_norm",
    "shape_min_to_end_norm",
    "shape_post_min_slope_norm",
    "tail_creep_norm",
]

EPS = 1e-12


# ============================================================
# Helpers
# ============================================================
def _infer_dt_ms(times: np.ndarray, default: float = 0.01) -> float:
    t = np.asarray(times, float)
    if t.size < 3:
        return float(default)
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else float(default)


def _safe_percentile(x: np.ndarray, q: float) -> float:
    xf = np.asarray(x, float)
    xf = xf[np.isfinite(xf)]
    if xf.size == 0:
        return 0.0
    return float(np.percentile(xf, q))


def _mad(x: np.ndarray, med: float | None = None) -> float:
    xf = np.asarray(x, float)
    xf = xf[np.isfinite(xf)]
    if xf.size == 0:
        return 0.0
    m = float(np.median(xf)) if med is None else float(med)
    return float(np.median(np.abs(xf - m)))


def _robust_std_from_mad(x: np.ndarray) -> float:
    return 1.4826 * _mad(x)


def _clip_feats(feats: dict[str, float]) -> dict[str, float]:
    lim = float(CFG["CLIP_LIMIT"])
    for k in CLIP_FEATS:
        if k in feats and np.isfinite(feats[k]):
            feats[k] = float(np.clip(feats[k], -lim, lim))
    return feats


def _tail_slice(x: np.ndarray, t: np.ndarray, dt_ms: float, tail_ms: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    tail_len = int(max(30, min(n, round(float(tail_ms) / max(dt_ms, 1e-6)))))
    return x[-tail_len:], t[-tail_len:]


# ============================================================
# Normalize wave
# ============================================================
def _best_stable_tail_level(
    values: np.ndarray,
    times: np.ndarray,
    *,
    tail_ms: float,
    win_ms: float,
) -> tuple[float, float]:
    x = np.asarray(values, float)
    t = np.asarray(times, float)
    n = x.size
    if n < 20:
        med = float(np.median(x)) if n else 0.0
        return med, float(_robust_std_from_mad(x))

    t_end = float(t[-1])
    idx_tail = np.where(t >= t_end - float(tail_ms))[0]
    if idx_tail.size < 20:
        idx_tail = np.arange(max(0, n - 50), n)

    xt = x[idx_tail]
    tt = t[idx_tail]
    dt = _infer_dt_ms(tt, default=0.01)
    win_n = int(max(10, round(win_ms / max(dt, 1e-9))))

    if xt.size <= win_n:
        med = float(np.median(xt))
        rstd = float(_robust_std_from_mad(xt))
        return med, rstd

    best_std = float("inf")
    best_med = float(np.median(xt[-win_n:]))
    for i in range(0, xt.size - win_n + 1):
        w = xt[i:i + win_n]
        med = float(np.median(w))
        rstd = float(_robust_std_from_mad(w))
        if rstd < best_std:
            best_std = rstd
            best_med = med

    return best_med, float(best_std)


def normalize_wave(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    x = np.asarray(values, float)
    t = np.asarray(times, float)
    n = x.size

    idx_start = np.where(t <= float(t[0]) + float(CFG["START_MS"]))[0]
    if idx_start.size < 10:
        idx_start = np.arange(min(30, n))
    start_ref = float(np.median(x[idx_start])) if idx_start.size else float(np.median(x))

    end_ref, end_noise = _best_stable_tail_level(
        x,
        t,
        tail_ms=float(CFG["TAIL_MS"]),
        win_ms=float(CFG["TAIL_WIN_MS"]),
    )

    p95 = _safe_percentile(x, 95)
    p05 = _safe_percentile(x, 5)
    span_raw = float(max(p95 - p05, 0.0))

    step_raw = float(end_ref - start_ref)
    direction = 1.0 if step_raw >= 0 else -1.0
    step_amp = abs(step_raw)

    sig_scale = max(abs(end_ref), abs(start_ref), span_raw, 1e-15)
    noise_floor = max(sig_scale * 1e-6, 1e-12)
    span_floor = max(sig_scale * 1e-5, 1e-12)

    global_span = float(max(span_raw, span_floor))
    end_noise = float(max(end_noise, noise_floor))
    denom_used = max(step_amp, 0.12 * global_span, 6.0 * end_noise, noise_floor)

    x_norm = (x - end_ref) / denom_used
    x_norm = np.clip(x_norm, -float(CFG["CLIP_LIMIT"]), float(CFG["CLIP_LIMIT"]))

    meta = {
        "meta_start_ref": float(start_ref),
        "meta_end_ref": float(end_ref),
        "meta_direction": float(direction),
        "meta_end_noise": float(end_noise),
        "meta_global_span": float(global_span),
        "meta_denom_used": float(denom_used),
        "meta_step_to_span": float(step_amp / (global_span + EPS)),
        "meta_noise_to_span": float(end_noise / (global_span + EPS)),
    }
    return x_norm, meta


# ============================================================
# Lean feature groups (no threshold-derived edge/glitch features)
# ============================================================
def base_features(x_norm: np.ndarray, dt_ms: float) -> dict[str, float]:
    dx = np.diff(x_norm)
    dt_s = max(float(dt_ms) * 1e-3, EPS)
    dxdt = dx / dt_s if dx.size else np.array([], dtype=float)
    return {
        "base_std": float(np.std(x_norm)),
        "base_min": float(np.min(x_norm)),
        "base_max": float(np.max(x_norm)),
        "base_p2p": float(np.ptp(x_norm)),
        "base_energy": float(np.sum(x_norm * x_norm)),
        "base_max_slope": float(np.max(np.abs(dxdt))) if dxdt.size else 0.0,
        "base_mean_abs_slope": float(np.mean(np.abs(dxdt))) if dxdt.size else 0.0,
    }


def tail_features(x_norm: np.ndarray, t: np.ndarray, dt_ms: float) -> dict[str, float]:
    tail, tail_t = _tail_slice(x_norm, t, dt_ms, float(CFG["TAIL_MS"]))
    if tail.size < 20:
        return {
            "tail_std": 0.0,
            "tail_p2p": 0.0,
            "tail_mean_abs_slope": 0.0,
            "tail_creep_norm": 0.0,
            "tail_decay_slope": 0.0,
        }

    dt_s = max(float(dt_ms) * 1e-3, EPS)
    dx = np.diff(tail)
    tail_mean_abs_slope = float(np.mean(np.abs(dx / dt_s))) if dx.size else 0.0

    half = max(1, tail.size // 2)
    a = tail[:half]
    b = tail[half:]
    tail_creep_norm = float(np.median(b) - np.median(a)) if b.size else 0.0

    err = np.clip(np.abs(tail), 1e-6, None)
    if err.size >= 10:
        try:
            tail_decay_slope = float(np.polyfit(tail_t.astype(float), np.log(err), 1)[0])
        except Exception:
            tail_decay_slope = 0.0
    else:
        tail_decay_slope = 0.0

    return {
        "tail_std": float(np.std(tail)),
        "tail_p2p": float(np.ptp(tail)),
        "tail_mean_abs_slope": tail_mean_abs_slope,
        "tail_creep_norm": tail_creep_norm,
        "tail_decay_slope": tail_decay_slope,
    }


def shape_features(x_norm: np.ndarray, t: np.ndarray, dt_ms: float, meta: dict[str, float]) -> dict[str, float]:
    n = x_norm.size
    if n < 30:
        return {
            "shape_overshoot_norm": 0.0,
            "shape_min_pos_ratio": 0.0,
            "shape_min_to_end_norm": 0.0,
            "shape_rebound_norm": 0.0,
            "shape_post_min_slope_norm": 0.0,
            "tail_monotonicity": 0.0,
            "late_activity": 0.0,
        }

    end_med = float(np.median(x_norm[-min(50, n):]))
    min_i = int(np.argmin(x_norm))
    post = x_norm[min_i:] if min_i < n - 2 else x_norm[-2:]

    k = min(200, post.size)
    if k >= 5:
        try:
            post_slope = float(np.polyfit(t[min_i:min_i + k].astype(float), post[:k], 1)[0])
        except Exception:
            post_slope = 0.0
    else:
        post_slope = 0.0

    direction = float(meta.get("meta_direction", 1.0))
    tail, _ = _tail_slice(x_norm, t, dt_ms, float(CFG["TAIL_MS"]))
    tail_dx = np.diff(tail)
    tail_monotonicity = float((np.sign(tail_dx) == np.sign(direction)).mean()) if tail_dx.size else 0.0

    k_late = int(max(10, min(n - 1, round(1.0 / max(dt_ms, 1e-9)))))
    seg = x_norm[-(k_late + 1):] if n > k_late + 1 else x_norm
    late_activity = float(np.mean(np.abs(np.diff(seg)))) if seg.size > 1 else 0.0

    return {
        "shape_overshoot_norm": float(np.max(x_norm - end_med)),
        "shape_min_pos_ratio": float(min_i / max(n - 1, 1)),
        "shape_min_to_end_norm": float(end_med - x_norm[min_i]),
        "shape_rebound_norm": float(np.max(post) - x_norm[min_i]) if post.size else 0.0,
        "shape_post_min_slope_norm": post_slope,
        "tail_monotonicity": tail_monotonicity,
        "late_activity": late_activity,
    }


def quiet_features(x_norm: np.ndarray, t: np.ndarray, dt_ms: float) -> dict[str, float]:
    n = x_norm.size
    if n < 50:
        return {
            "quiet_after_head_ratio": 0.0,
            "first_quiet_pos_ratio": 1.0,
            "post_head_std": 0.0,
        }

    head_ms = 0.8
    idx = np.where(t >= float(t[0]) + head_ms)[0]
    if idx.size < 10:
        idx = np.arange(n // 2, n)

    seg = x_norm[idx]
    dx = np.abs(np.diff(seg))
    if dx.size < 5:
        return {
            "quiet_after_head_ratio": 0.0,
            "first_quiet_pos_ratio": 1.0,
            "post_head_std": float(np.std(seg)),
        }

    # ใช้ distribution ของ dx เอง ไม่ตั้ง threshold ตายตัวจากภายนอก
    quiet_thr = float(np.quantile(dx, 0.30))
    quiet_mask = dx <= quiet_thr
    quiet_after_head_ratio = float(np.mean(quiet_mask))

    win_n = int(max(5, round(0.3 / max(dt_ms, 1e-6))))
    first_quiet = 1.0
    if quiet_mask.size >= win_n:
        for k in range(0, quiet_mask.size - win_n + 1):
            if np.all(quiet_mask[k:k + win_n]):
                global_i = int(idx[0] + k)
                first_quiet = float(global_i / max(n - 1, 1))
                break

    return {
        "quiet_after_head_ratio": quiet_after_head_ratio,
        "first_quiet_pos_ratio": first_quiet,
        "post_head_std": float(np.std(seg)),
    }


def ring_features(x_norm: np.ndarray, dt_ms: float) -> dict[str, float]:
    n = x_norm.size
    if n < 80:
        return {
            "ring_mean_peak_spacing_ms": 0.0,
            "ring_std_peak_spacing_ms": 0.0,
            "ring_damping_slope": 0.0,
        }

    tail = x_norm[-min(n, 400):]
    tail = tail - float(np.median(tail))
    d = np.diff(tail)
    s = np.sign(d)
    peak_idx = np.where((s[:-1] > 0) & (s[1:] < 0))[0] + 1

    if peak_idx.size >= 2:
        spacings = np.diff(peak_idx) * dt_ms
        mean_sp = float(np.mean(spacings))
        std_sp = float(np.std(spacings))
    else:
        mean_sp = 0.0
        std_sp = 0.0

    if peak_idx.size >= 3:
        peak_vals = np.abs(tail[peak_idx])
        y = np.log(peak_vals + EPS)
        try:
            damping = float(np.polyfit(np.arange(y.size, dtype=float), y, 1)[0])
        except Exception:
            damping = 0.0
    else:
        damping = 0.0

    return {
        "ring_mean_peak_spacing_ms": mean_sp,
        "ring_std_peak_spacing_ms": std_sp,
        "ring_damping_slope": damping,
    }


def activity_features(x_norm: np.ndarray, dt_ms: float) -> dict[str, float]:
    dx = np.abs(np.diff(x_norm))
    if dx.size < 5:
        return {"max_contiguous_activity_ms": 0.0}

    # ใช้ quantile ภายในคลื่นเอง แทน threshold คงที่
    thr = float(np.quantile(dx, 0.75))
    active = dx > thr

    max_run = 0
    cur = 0
    for a in active:
        if a:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    return {"max_contiguous_activity_ms": float(max_run * dt_ms)}


# ============================================================
# Labels (train only)
# ============================================================
def get_label_from_group(g: pd.DataFrame) -> tuple[float, str]:
    fast_ms = float(CFG["FAST_MS"])
    min_out_ms = float(CFG["MIN_OUT_MS"])
    tmax = float(np.nanmax(g["time_ms"].to_numpy(float))) if len(g) else fast_ms

    if "true_is_zero" in g.columns and pd.notna(g["true_is_zero"].iloc[0]) and int(g["true_is_zero"].iloc[0]) == 1:
        return fast_ms, "gt_true_is_zero"
    if "true_wait_time_ms" in g.columns and pd.notna(g["true_wait_time_ms"].iloc[0]):
        w = float(g["true_wait_time_ms"].iloc[0])
        return float(np.clip(w, min_out_ms, tmax)), "gt_true_wait_time_ms"
    if "wait_time_ms" in g.columns and pd.notna(g["wait_time_ms"].iloc[0]):
        w = float(g["wait_time_ms"].iloc[0])
        return float(np.clip(w, min_out_ms, tmax)), "gt_wait_time_ms"

    raise RuntimeError("Missing label columns (need true_wait_time_ms/true_is_zero/wait_time_ms)")


# ============================================================
# Per-wave extraction
# ============================================================
def extract_one_wave(g: pd.DataFrame, mode: str) -> dict[str, float | int | str]:
    t = g["time_ms"].to_numpy(float)
    x = g["value"].to_numpy(float)
    dt_ms = _infer_dt_ms(t, default=0.01)

    x_norm, meta = normalize_wave(x, t)

    feats: dict[str, float] = {}
    feats.update(meta)
    feats.update(base_features(x_norm, dt_ms))
    feats.update(tail_features(x_norm, t, dt_ms))
    feats.update(shape_features(x_norm, t, dt_ms, meta))
    feats.update(quiet_features(x_norm, t, dt_ms))
    feats.update(ring_features(x_norm, dt_ms))
    feats.update(activity_features(x_norm, dt_ms))

    feats.update(late_transition_features(x_norm, t, dt_ms))
    feats.update(plateau_features(x_norm, t, dt_ms))
    feats.update(late_window_features(x_norm, t))
    feats.update(regime_change_features(x_norm, t, dt_ms))
    feats = _clip_feats(feats)

    out: dict[str, float | int | str] = {
        "wave_id": int(g["wave_id"].iloc[0]),
        **feats,
    }

    if mode == "train":
        w, reason = get_label_from_group(g)
        out["wait_time_ms"] = float(w)
        out["dbg_label_reason"] = str(reason)

    if "type" in g.columns:
        out["type" ] = str(g["type"].iloc[0]) if pd.notna(g["type"].iloc[0]) else "unknown"
    if "sd" in g.columns:
        try:
            out["sd"] = float(np.nanmedian(g["sd"].to_numpy(float)))
        except Exception:
            out["sd"] = np.nan

    return out

# ========================
# ADD
# ========================

EPS = 1e-12


def late_transition_features(xN: np.ndarray, t: np.ndarray, dt_ms: float) -> dict:
    """
    จับ event ใหญ่ช่วงกลาง-ท้ายของ waveform
    xN = normalized waveform
    t  = time in ms
    """
    N = len(xN)
    if N < 20:
        return {
            "last_big_slope_time_ms": 0.0,
            "last_big_slope_pos_ratio": 0.0,
            "num_big_slopes_after_half": 0.0,
            "max_abs_slope_after_half": 0.0,
        }

    dx = np.diff(xN)
    abs_dx = np.abs(dx)

    med = float(np.median(abs_dx))
    mad = float(np.median(np.abs(abs_dx - med))) + EPS
    thr = med + 6.0 * 1.4826 * mad

    big_idx = np.where(abs_dx > thr)[0]

    if len(big_idx) == 0:
        last_big_time = 0.0
        last_big_ratio = 0.0
    else:
        last_i = int(big_idx[-1])
        last_big_time = float(t[last_i + 1])
        last_big_ratio = float((last_i + 1) / max(N - 1, 1))

    half_idx = int(0.5 * (N - 1))
    after_half = big_idx[big_idx >= half_idx]

    max_abs_slope_after_half = float(np.max(abs_dx[half_idx:])) if half_idx < len(abs_dx) else 0.0

    return {
        "last_big_slope_time_ms": last_big_time,
        "last_big_slope_pos_ratio": last_big_ratio,
        "num_big_slopes_after_half": float(len(after_half)),
        "max_abs_slope_after_half": max_abs_slope_after_half,
    }


def plateau_features(xN: np.ndarray, t: np.ndarray, dt_ms: float) -> dict:
    """
    หาเวลาที่ waveform เริ่มนิ่งแบบต่อเนื่อง
    """
    N = len(xN)
    if N < 40:
        return {
            "plateau_enter_time_ms": float(t[-1]) if N else 0.0,
            "plateau_enter_pos_ratio": 1.0,
            "stable_run_len_ms": 0.0,
            "tail_range_last_20pct": 0.0,
            "tail_std_last_20pct": 0.0,
            "tail_mean_abs_slope_last_20pct": 0.0,
        }

    dx = np.abs(np.diff(xN))
    med = float(np.median(dx))
    mad = float(np.median(np.abs(dx - med))) + EPS
    quiet_thr = med + 2.5 * 1.4826 * mad

    # ต้องนิ่งต่อเนื่องยาวอย่างน้อย ~0.5 ms
    W = int(max(5, round(0.5 / max(dt_ms, 1e-9))))

    plateau_idx = None
    if len(dx) >= W:
        quiet_mask = dx <= quiet_thr
        for i in range(0, len(quiet_mask) - W + 1):
            if np.all(quiet_mask[i:i + W]):
                plateau_idx = i
                break

    if plateau_idx is None:
        plateau_enter_time_ms = float(t[-1])
        plateau_enter_pos_ratio = 1.0
        stable_run_len_ms = 0.0
    else:
        plateau_enter_time_ms = float(t[plateau_idx])
        plateau_enter_pos_ratio = float(plateau_idx / max(N - 1, 1))
        stable_run_len_ms = float(t[-1] - t[plateau_idx])

    tail_start = int(0.8 * N)
    tail = xN[tail_start:]
    tail_dx = np.diff(tail)

    return {
        "plateau_enter_time_ms": plateau_enter_time_ms,
        "plateau_enter_pos_ratio": plateau_enter_pos_ratio,
        "stable_run_len_ms": stable_run_len_ms,
        "tail_range_last_20pct": float(np.ptp(tail)) if len(tail) > 1 else 0.0,
        "tail_std_last_20pct": float(np.std(tail)) if len(tail) > 1 else 0.0,
        "tail_mean_abs_slope_last_20pct": float(np.mean(np.abs(tail_dx))) if len(tail_dx) > 0 else 0.0,
    }


def late_window_features(xN: np.ndarray, t: np.ndarray) -> dict:
    """
    ดู activity ในช่วงท้ายหลายหน้าต่าง
    """
    N = len(xN)
    if N < 20:
        return {
            "std_last_10pct": 0.0,
            "std_last_20pct": 0.0,
            "std_last_30pct": 0.0,
            "mean_abs_slope_last_10pct": 0.0,
            "mean_abs_slope_last_20pct": 0.0,
            "mean_abs_slope_last_30pct": 0.0,
        }

    def _window_stats(frac: float) -> tuple[float, float]:
        start = int((1.0 - frac) * N)
        seg = xN[start:]
        if len(seg) < 2:
            return 0.0, 0.0
        dx = np.diff(seg)
        return float(np.std(seg)), float(np.mean(np.abs(dx)))

    std10, slope10 = _window_stats(0.10)
    std20, slope20 = _window_stats(0.20)
    std30, slope30 = _window_stats(0.30)

    return {
        "std_last_10pct": std10,
        "std_last_20pct": std20,
        "std_last_30pct": std30,
        "mean_abs_slope_last_10pct": slope10,
        "mean_abs_slope_last_20pct": slope20,
        "mean_abs_slope_last_30pct": slope30,
    }


def regime_change_features(xN: np.ndarray, t: np.ndarray, dt_ms: float) -> dict:
    """
    จับ multi-stage settling แบบหยาบ ๆ ด้วย mean shift
    """
    N = len(xN)
    if N < 60:
        return {
            "num_mean_shifts": 0.0,
            "largest_late_mean_shift": 0.0,
            "largest_late_shift_time_ms": 0.0,
        }

    # แบ่งเป็น window ย่อย
    W = int(max(10, round(0.4 / max(dt_ms, 1e-9))))
    means = []
    centers = []

    for i in range(0, N - W + 1, W):
        seg = xN[i:i + W]
        means.append(float(np.mean(seg)))
        centers.append(float(t[min(i + W // 2, N - 1)]))

    means = np.asarray(means, dtype=float)
    centers = np.asarray(centers, dtype=float)

    if len(means) < 2:
        return {
            "num_mean_shifts": 0.0,
            "largest_late_mean_shift": 0.0,
            "largest_late_shift_time_ms": 0.0,
        }

    dmean = np.abs(np.diff(means))
    med = float(np.median(dmean))
    mad = float(np.median(np.abs(dmean - med))) + EPS
    thr = med + 3.5 * 1.4826 * mad

    shift_idx = np.where(dmean > thr)[0]
    num_mean_shifts = float(len(shift_idx))

    # สนใจเฉพาะครึ่งหลัง
    late_mask = centers[:-1] >= (0.5 * float(t[-1]))
    late_dmean = dmean[late_mask]
    late_centers = centers[:-1][late_mask]

    if len(late_dmean) == 0:
        largest_late_mean_shift = 0.0
        largest_late_shift_time_ms = 0.0
    else:
        j = int(np.argmax(late_dmean))
        largest_late_mean_shift = float(late_dmean[j])
        largest_late_shift_time_ms = float(late_centers[j])

    return {
        "num_mean_shifts": num_mean_shifts,
        "largest_late_mean_shift": largest_late_mean_shift,
        "largest_late_shift_time_ms": largest_late_shift_time_ms,
    }

# ============================================================
# IO helpers
# ============================================================
def normalize_columns(
    df: pd.DataFrame,
    id_col: str,
    sample_col: str,
    time_col: str,
    value_col: str,
) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        id_col: "wave_id",
        sample_col: "sample",
        time_col: "time_ms",
        value_col: "value",
    }
    for c in rename_map:
        if c not in out.columns:
            raise KeyError(f"Missing required column: {c}")
    out = out.rename(columns=rename_map)

    for opt in ["sd", "type", "low_limit", "high_limit", "true_wait_time_ms", "true_is_zero", "wait_time_ms"]:
        if opt not in out.columns:
            out[opt] = np.nan
    return out


# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "pred"], required=True)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--id-col", default="wave_id")
    ap.add_argument("--sample-col", default="sample")
    ap.add_argument("--time-col", default="time_ms")
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--window-ms", type=float, default=float(CFG["WINDOW_MS"]))
    ap.add_argument("--min-pts", type=int, default=int(CFG["MIN_PTS"]))
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(in_path)
    raw = normalize_columns(raw, args.id_col, args.sample_col, args.time_col, args.value_col)

    raw["sample"] = raw["sample"].astype(int)
    raw = raw.sort_values(["wave_id", "sample"])
    raw = raw[raw["time_ms"].astype(float) <= float(args.window_ms)].copy()

    cnt = raw.groupby("wave_id")["sample"].count()
    keep = cnt[cnt >= int(args.min_pts)].index
    raw = raw[raw["wave_id"].isin(keep)].copy()

    rows = []
    for wid, g in raw.groupby("wave_id"):
        try:
            rows.append(extract_one_wave(g, mode=args.mode))
        except Exception as e:
            raise RuntimeError(f"wave_id={wid} feature extraction failed: {e}")

    out = pd.DataFrame(rows)
    if args.mode == "pred" and "wait_time_ms" in out.columns:
        out = out.drop(columns=["wait_time_ms"], errors="ignore")

    out.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path} | rows={len(out)} cols={len(out.columns)}")


if __name__ == "__main__":
    main()
