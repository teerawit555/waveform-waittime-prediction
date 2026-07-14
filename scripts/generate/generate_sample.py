# scripts/generate/generate_sample.py
"""
Generate synthetic training waveform data.

Columns in output CSV:
  wave_id, type, sample, time_ms, value, sd,
  low_limit, high_limit, wait_time_ms
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


HIGH_NOISE_RATIO = 0.8


# =============================================================================
# 1) Utility Functions
# =============================================================================

def apply_cosine_taper_settling(
    signal_array: np.ndarray,
    time_vector: np.ndarray,
    settling_time_s: float,
    target_value: float,
    strength: float = 1.0,
    gamma: float = 0.8,
) -> np.ndarray:
    """
    Smoothly forces the signal toward target_value by settling_time_s
    using a cosine taper mask.

    gamma > 1.0 → settle ไวขึ้น, gamma < 1.0 → ช้าลง
    """
    fade_mask = np.zeros_like(time_vector, dtype=float)
    active = time_vector < settling_time_s

    if np.any(active):
        t_ratio = time_vector[active] / max(settling_time_s, 1e-12)
        base = 0.5 * (1.0 + np.cos(np.pi * t_ratio))
        fade_mask[active] = np.power(base, gamma)

    deviation = signal_array - target_value
    tapered = target_value + deviation * fade_mask
    return strength * tapered + (1.0 - strength) * signal_array


def add_post_settle_noise(
    signal_array: np.ndarray,
    time_vector: np.ndarray,
    settling_time_s: float,
    target_value: float,
    rng: np.random.Generator,
    probability: float = 0.65,
    post_sd_scale=(0.0010, 0.0045),
    smoothness_range=(18, 35),
    add_wobble_prob: float = 0.00,
    wobble_scale=(0.00001, 0.00005),
    wobble_win_range=(60, 130),
) -> tuple[np.ndarray, float]:
    """
    เติม noise หลัง settle โดย scale ตาม magnitude ของ target_value
    ประกอบด้วย 4 components:
      1) base floor noise  (white noise ระดับต่ำ)
      2) correlated wiggle (smoothed → low-freq)
      3) alternating noise (สร้าง AC(1) < 0 เหมือน ADC quantization)
      4) slow drift        (สร้าง AC(5) > 0 เหมือน thermal drift)
    """
    mag = max(abs(float(target_value)), 1e-12)
    floor_abs = max(mag * 1e-4, 1e-15)

    # --- 1) Base floor noise (ทั้ง signal) ---
    base_floor_sd = max(mag * rng.uniform(5e-4, 1.5e-3), floor_abs)
    signal_array = signal_array + rng.normal(0.0, base_floor_sd, size=len(time_vector))
    final_sd = base_floor_sd

    settle_idx = int(np.searchsorted(time_vector, settling_time_s))
    remaining_len = len(time_vector) - settle_idx
    if remaining_len <= 5:
        return signal_array, final_sd

    # --- 2) Correlated wiggle (post-settle) ---
    if rng.random() < probability:
        post_sd = max(mag * rng.uniform(*post_sd_scale), floor_abs)
        final_sd = max(final_sd, post_sd)

        raw = rng.normal(0.0, post_sd, size=remaining_len)
        smoothness = int(rng.integers(smoothness_range[0], smoothness_range[1] + 1))
        smoothness = min(smoothness, remaining_len - 1)

        if smoothness > 2:
            k = np.ones(smoothness) / smoothness
            wig = np.convolve(raw, k, mode="same") * (math.sqrt(smoothness) * 0.35)
            signal_array[settle_idx:] += wig
        else:
            signal_array[settle_idx:] += raw

    # --- 3) Optional wobble (post-settle) ---
    if rng.random() < add_wobble_prob:
        wob_sd = max(mag * rng.uniform(*wobble_scale), floor_abs)
        final_sd = max(final_sd, wob_sd)

        wob = rng.normal(0.0, wob_sd, size=remaining_len)
        win = int(rng.integers(wobble_win_range[0], wobble_win_range[1] + 1))
        win = min(win, remaining_len - 1)

        if win > 3:
            k2 = np.ones(win) / win
            wob = np.convolve(wob, k2, mode="same") * math.sqrt(win)
        wob = (wob - np.mean(wob)) * 0.25
        signal_array[settle_idx:] += wob

    # --- 4) Alternating noise → AC(1) ≈ -0.4 (เหมือน ADC quantization) ---
    quant_sd = max(mag * rng.uniform(2e-4, 8e-4), floor_abs)
    quant_noise = rng.normal(0.0, quant_sd, size=remaining_len)
    alt_signs = np.empty(remaining_len, dtype=float)
    alt_signs[0::2] = 1.0
    alt_signs[1::2] = -1.0
    signal_array[settle_idx:] += quant_noise * alt_signs * 0.5

    # --- 5) Slow low-freq drift → AC(5) > 0 (เหมือน thermal drift) ---
    drift_sd = max(mag * rng.uniform(1e-4, 4e-4), floor_abs)
    drift_raw = rng.normal(0.0, drift_sd, size=remaining_len)
    win_drift = int(rng.integers(40, 80))
    k_drift = np.ones(win_drift) / win_drift
    drift = np.convolve(drift_raw, k_drift, mode="same") * math.sqrt(win_drift)
    signal_array[settle_idx:] += drift

    return signal_array, final_sd


def add_time_delay(
    t: np.ndarray,
    rng: np.random.Generator,
    max_delay_s: float = 0.0004,
) -> tuple[np.ndarray, float]:
    """Random time shift. Returns (shifted_time, delay_value_s)."""
    d = float(rng.uniform(0.0, max_delay_s))
    return np.clip(t - d, 0.0, None), d


def soft_flatten_after_settle(
    y: np.ndarray,
    t: np.ndarray,
    settling_time_s: float,
    target: float,
    blend_window_s: float = 0.0003,
) -> np.ndarray:
    """Blend เข้า target ช่วงก่อน settle และ hold หลัง settle."""
    if settling_time_s <= 0:
        return y

    N = len(y)
    si = int(np.searchsorted(t, settling_time_s))
    if si <= 1 or si >= N:
        return y

    t0 = max(settling_time_s - blend_window_s, 0.0)
    i0 = max(0, min(int(np.searchsorted(t, t0)), si))

    w = np.linspace(0.0, 1.0, max(si - i0, 1))
    y2 = y.copy()
    y2[i0:si] = (1 - w) * y2[i0:si] + w * target
    y2[si:] = target
    return y2


def flatten_after_settle(
    y: np.ndarray,
    t: np.ndarray,
    settling_time_s: float,
    blend_s: float = 0.00025,
    win_s: float = 0.00015,
) -> np.ndarray:
    """
    Lock signal หลัง settle_idx ให้แบนจริง (ก่อนใส่ noise)
    โดย hold ค่าจากค่าเฉลี่ยช่วงท้ายก่อน settle
    """
    si = int(np.searchsorted(t, settling_time_s))
    if si <= 2 or si >= len(y):
        return y

    dt = float(t[1] - t[0]) if len(t) > 1 else 1e-6
    win_n = max(3, int(win_s / max(dt, 1e-12)))
    hold_val = float(np.mean(y[max(0, si - win_n):si]))

    blend_n = max(1, int(blend_s / max(dt, 1e-12)))
    b0 = max(0, si - blend_n)
    if si > b0:
        w = np.linspace(0.0, 1.0, si - b0, endpoint=False)
        y[b0:si] = (1.0 - w) * y[b0:si] + w * hold_val

    y[si:] = hold_val
    return y


def sample_target_mixed_units(rng: np.random.Generator) -> float:
    """
    สุ่ม target_value ที่ครอบคลุมหลาย scale:
      15% → ns  (1e-9 .. 1e-8)
      20% → us  (1e-6 .. 1e-5)
      50% → ms  (1e-3 .. 1)
      15% → normal (1 .. 50)
    """
    p = rng.random()
    if p < 0.15:
        value = float(10 ** rng.uniform(-9, -8))
    elif p < 0.35:
        value = float(10 ** rng.uniform(-6, -5))
    elif p < 0.85:
        value = float(10 ** rng.uniform(-3, 0))
    else:
        value = float(rng.uniform(1.0, 50.0))

    # SignalSample.xlsx contains both polarities in roughly equal proportion.
    return value if rng.random() < 0.5 else -value


def add_signal_sample_noise(
    signal_array: np.ndarray,
    target_value: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Apply either visible or reference-level noise to any waveform.

    Robust first-difference estimates from the 11 reference signals give a
    relative high-frequency noise range of 0.12%-0.33%, with a 0.19% median.
    Most generated waves additionally use noise scaled to their waveform span,
    making noise equally visible on large and small signal ranges. A smaller
    share retains the original reference-level profile.
    """
    clean_signal = np.asarray(signal_array, dtype=float)
    mag = max(abs(float(target_value)), 1e-12)
    rel_sd = float(rng.triangular(0.0012, 0.0019, 0.0033))
    reference_sd = max(mag * rel_sd, 1e-15)

    use_high_noise = bool(rng.random() < HIGH_NOISE_RATIO)
    if use_high_noise:
        q_low, q_high = np.quantile(clean_signal, [0.01, 0.99])
        waveform_span = max(float(q_high - q_low), 0.0)
        span_noise_ratio = float(rng.triangular(0.006, 0.010, 0.016))
        white_sd = max(reference_sd, waveform_span * span_noise_ratio)
    else:
        white_sd = reference_sd

    n = len(clean_signal)

    white = rng.normal(0.0, white_sd, size=n)

    # The reference signals are mostly white noise with a small correlated
    # component. Normalize the smoothed components before scaling so the
    # requested relative noise remains stable for every waveform type.
    corr_raw = rng.normal(0.0, 1.0, size=n)
    corr_win = min(int(rng.integers(5, 13)), max(n - 1, 1))
    corr = np.convolve(corr_raw, np.ones(corr_win) / corr_win, mode="same")
    corr_std = float(np.std(corr))
    if corr_std > 0:
        corr = corr / corr_std * (white_sd * 0.18)

    drift_raw = rng.normal(0.0, 1.0, size=n)
    drift_win = min(int(rng.integers(45, 90)), max(n - 1, 1))
    drift = np.convolve(drift_raw, np.ones(drift_win) / drift_win, mode="same")
    drift_std = float(np.std(drift))
    if drift_std > 0:
        drift = drift / drift_std * (white_sd * 0.08)

    return clean_signal + white + corr + drift, white_sd


def estimate_slow_tail_label_ms(
    time_ms: np.ndarray,
    signal_array: np.ndarray,
    noise_sd: float,
) -> float:
    """Estimate when a slow monotonic tail is visually stable.

    Slow-tail waveforms intentionally keep their long exponential tail instead
    of being flattened at a preselected time. Their label therefore needs to be
    derived from the generated waveform: after light median smoothing, the
    remaining tail must stay within one generated-noise standard deviation of
    the final level.
    """
    t = np.asarray(time_ms, dtype=float)
    y = np.asarray(signal_array, dtype=float)
    if len(t) < 2 or len(y) != len(t):
        return float(t[-1]) if len(t) else 0.0

    dt_ms = max(float(np.median(np.diff(t))), 1e-12)
    smooth_samples = max(5, int(round(0.25 / dt_ms)))
    if smooth_samples % 2 == 0:
        smooth_samples += 1

    smooth = (
        pd.Series(y)
        .rolling(smooth_samples, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    tail_samples = max(smooth_samples, int(round(0.05 * len(smooth))))
    final_level = float(np.median(smooth[-tail_samples:]))
    numeric_floor = np.finfo(float).eps * max(float(np.max(np.abs(y))), 1.0)
    tolerance = max(float(noise_sd), numeric_floor)

    within_final_noise = np.abs(smooth - final_level) <= tolerance
    stable_through_end = np.logical_and.accumulate(within_final_noise[::-1])[::-1]
    stable_indices = np.flatnonzero(stable_through_end)
    stable_index = int(stable_indices[0]) if len(stable_indices) else len(t) - 1
    return float(t[stable_index])


def apply_high_noise_boost(
    y: np.ndarray,
    t_s: np.ndarray,
    settle_s: float,
    final_value: float,
    band: float,
    wave_rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """
    เพิ่ม colored noise สำหรับ wave ที่ signal_swing น้อยกว่า 5% ของ band
    (เช่น wave 9/10 ใน real data ที่ signal แบนแต่ noise สูง)

    คืนค่า (y_modified, boosted_sd)
    """
    signal_swing = abs(float(y[0]) - float(np.mean(y[int(0.75 * len(y)):])))
    if signal_swing >= 0.05 * band:
        # swing ปกติ ไม่ต้องบูสต์
        return y, 0.0

    mag = max(abs(final_value), 1e-12)
    noise_boost = float(wave_rng.uniform(2.0, 4.0))
    boosted_sd = max(mag * wave_rng.uniform(0.0010, 0.0045), 1e-12) * noise_boost

    si = int(np.searchsorted(t_s, settle_s))
    n_post = len(y) - si
    if n_post <= 0:
        return y, boosted_sd

    # Colored noise: alternating (AC<0) + slow drift (AC>0)
    raw = wave_rng.normal(0.0, boosted_sd, size=n_post)
    alt_signs = np.empty(n_post, dtype=float)
    alt_signs[0::2] = 1.0
    alt_signs[1::2] = -1.0
    alternating = raw * alt_signs

    smooth_win = int(wave_rng.integers(40, 80))
    k = np.ones(smooth_win) / smooth_win
    drift = (
        np.convolve(wave_rng.normal(0.0, boosted_sd, size=n_post), k, mode="same")
        * math.sqrt(smooth_win)
    )

    y[si:] += alternating * 0.4 + drift * 0.6
    return y, boosted_sd


# =============================================================================
# 2) Waveform Generators
# =============================================================================

def generate_step_response(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 0: Step response + damped ringing."""
    freq_hz = float(rng.uniform(100, 1200))
    w1 = 2.0 * np.pi * freq_hz
    band_half = (limit_high - limit_low) / 2.0

    overshoot_scale = float(rng.uniform(0.25, 2.25))
    direction = float(rng.choice([1.0, -1.0]))
    amp0 = band_half * overshoot_scale * direction

    tau = max(settling_time_s / float(rng.uniform(3.5, 6.5)), 1e-6)
    tc_rise = max(settling_time_s / float(rng.uniform(3.0, 4.5)), 1e-6)
    t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.00004)

    if rng.random() < 0.75:
        base = target_value * (1.0 - np.exp(-t_eff / tc_rise))
    else:
        tau2 = max(tc_rise * float(rng.uniform(1.5, 4.0)), 1e-6)
        base = target_value * (
            1.0 - 0.6 * np.exp(-t_eff / tc_rise) - 0.4 * np.exp(-t_eff / tau2)
        )

    if rng.random() < 0.6:
        ring = amp0 * np.exp(-t_eff / tau) * np.sin(w1 * t_eff)
    else:
        w2 = 2.0 * np.pi * freq_hz * float(rng.uniform(0.75, 1.25))
        mix = float(rng.uniform(0.2, 0.6))
        ring = amp0 * np.exp(-t_eff / tau) * (
            (1.0 - mix) * np.sin(w1 * t_eff) + mix * np.sin(w2 * t_eff)
        )

    y = base + ring

    if rng.random() < 0.25:
        kick_amp = (limit_high - limit_low) * float(rng.uniform(0.15, 0.8)) * float(
            rng.choice([1.0, -1.0])
        )
        kick_tau = max(settling_time_s / float(rng.uniform(6.0, 14.0)), 1e-6)
        y += kick_amp * np.exp(-t_eff / kick_tau)

    y = apply_cosine_taper_settling(
        y, time_vector, settling_time_s, target_value,
        strength=float(rng.uniform(0.88, 0.98))
    )
    y = flatten_after_settle(y, time_vector, settling_time_s)

    return y, 0.0, "type0_Step_Response"


def generate_high_start_oscillation(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 1: Damped oscillation (2–4 cycles) with startup ramp."""
    t = time_vector.astype(float)
    band = float(limit_high - limit_low)
    t_eff, _ = add_time_delay(t, rng, max_delay_s=0.00004)

    num_cycles = float(rng.uniform(2.0, 4.0))
    w = 2.0 * np.pi * num_cycles / max(settling_time_s, 1e-6)

    eps = float(rng.uniform(0.010, 0.025))
    tau_env = max(-settling_time_s / np.log(max(eps, 1e-9)), 1e-6)
    env = np.exp(-t_eff / tau_env)

    tau_ramp = max(settling_time_s / float(rng.uniform(10.0, 20.0)), 1e-6)
    ramp = 1.0 - np.exp(-t_eff / tau_ramp)

    A = band * float(rng.uniform(1.4, 2.6))
    bias_amp = band * float(rng.uniform(-0.25, 0.25))
    bias_tau = max(settling_time_s / float(rng.uniform(1.6, 3.0)), 1e-6)
    bias = bias_amp * np.exp(-t_eff / bias_tau)

    if rng.random() < 0.15:
        w2 = w * float(rng.uniform(0.90, 1.10))
        mix = float(rng.uniform(0.25, 0.50))
        osc = A * ramp * env * (
            (1.0 - mix) * np.cos(w * t_eff) + mix * np.cos(w2 * t_eff)
        )
    else:
        osc = A * ramp * env * np.cos(w * t_eff)

    y = target_value + bias + osc
    y = apply_cosine_taper_settling(
        y, t, settling_time_s, target_value,
        strength=float(rng.uniform(0.92, 0.99))
    )

    return y, 0.0, "type1_Damped_Osc"


def generate_continuous_triangular_pulses(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 2: Triangular pulse train (period อิงจาก t_end ไม่ใช่ settle)."""
    y = np.full_like(time_vector, target_value, dtype=float)
    band = float(limit_high - limit_low)
    avg_height = band * float(rng.uniform(0.5, 1.5))
    t_end = float(time_vector[-1])

    avg_period = float(rng.uniform(t_end / 14.0, t_end / 8.0))
    pulse_width = avg_period * float(rng.uniform(0.15, 0.30))
    is_height_const = bool(rng.random() < 0.70)
    is_period_const = bool(rng.random() < 0.70)

    start_after_s = 0.0005
    current_time = (
        max(float(settling_time_s), start_after_s)
        + float(rng.uniform(0.0, avg_period * 0.3))
    )

    while current_time < t_end:
        height = avg_height if is_height_const else avg_height * float(rng.uniform(0.85, 1.15))
        t_start = current_time
        t_peak = current_time + pulse_width / 2.0
        t_end_pulse = current_time + pulse_width

        rise = (time_vector >= t_start) & (time_vector < t_peak)
        if np.any(rise):
            y[rise] += (height / (pulse_width / 2.0)) * (time_vector[rise] - t_start)

        fall = (time_vector >= t_peak) & (time_vector < t_end_pulse)
        if np.any(fall):
            y[fall] += height - (height / (pulse_width / 2.0)) * (time_vector[fall] - t_peak)

        period = avg_period if is_period_const else avg_period * float(rng.uniform(0.90, 1.10))
        current_time += period

    early = time_vector < (0.2 * time_vector[-1])
    y[early] = np.maximum(y[early], target_value)
    pre = time_vector < max(float(settling_time_s), start_after_s)
    y[pre] = target_value

    return y, 0.0, "type2_Triangle_Wave"


def generate_overdamped_decay(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 4a: Overdamped exponential decay (no ringing)."""
    direction = 1.0 if target_value >= 0 else -1.0
    start_amp = direction * (limit_high - limit_low) * float(rng.uniform(1.5, 3.0))
    tau = settling_time_s / float(rng.uniform(3.0, 5.0))

    y = target_value + start_amp * np.exp(-time_vector / max(tau, 1e-12))
    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=1.0)
    return y, 0.0, "type4_overdamped_no_overshoot"


def generate_overdamped_decay1(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 4b: Bi-exponential overdamped with single undershoot."""
    band = float(limit_high - limit_low)
    tau_fast = max(settling_time_s / float(rng.uniform(6.0, 12.0)), 1e-6)
    tau_slow = max(settling_time_s / float(rng.uniform(1.8, 3.5)), 1e-6)
    A_pos = band * float(rng.uniform(1.5, 3.0))
    A_neg = A_pos * float(rng.uniform(0.35, 0.75))

    t_eff = time_vector
    if rng.random() < 0.35:
        t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.00004)

    direction = 1.0 if target_value >= 0 else -1.0
    y = target_value + direction * (
        A_pos * np.exp(-t_eff / tau_fast) - A_neg * np.exp(-t_eff / tau_slow)
    )
    y = apply_cosine_taper_settling(
        y, time_vector, settling_time_s, target_value,
        strength=float(rng.uniform(0.55, 0.8))
    )
    y = flatten_after_settle(y, time_vector, settling_time_s)
    return y, 0.0, "type4_overdamped_decay_overshoot"


def generate_slow_tail_biexponential(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 4c: fast knee followed by a long monotonic tail.

    This shape mirrors Signal1/3/4/6/7 in SignalSample.xlsx, which are not
    fully described by a single exponential within the 10 ms window.
    """
    direction = 1.0 if target_value >= 0 else -1.0
    band = float(limit_high - limit_low)
    amplitude = band * float(rng.uniform(1.8, 3.2))
    fast_mix = float(rng.uniform(0.45, 0.72))
    tau_fast = max(settling_time_s / float(rng.uniform(4.5, 8.0)), 1e-6)
    tau_slow = max(settling_time_s / float(rng.uniform(0.75, 1.35)), 1e-6)
    transient = amplitude * (
        fast_mix * np.exp(-time_vector / tau_fast)
        + (1.0 - fast_mix) * np.exp(-time_vector / tau_slow)
    )
    y = target_value + direction * transient
    return y, 0.0, "type4_slow_tail_biexponential"


def generate_noisy_flat_settling(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 3: noisy near-flat waveform with a short startup transient."""
    t = time_vector.astype(float)
    mag = max(abs(float(target_value)), 1e-12)
    band = max(float(limit_high - limit_low), mag * 0.02)

    tau = max(settling_time_s / float(rng.uniform(3.0, 5.0)), 1e-6)
    offset = band * float(rng.uniform(-1.2, 1.2))
    transient = offset * np.exp(-t / tau)

    if rng.random() < 0.55:
        cycles = float(rng.uniform(0.75, 2.0))
        transient += (
            band
            * float(rng.uniform(0.2, 0.7))
            * np.exp(-t / max(tau * 0.75, 1e-6))
            * np.sin(2.0 * np.pi * cycles * t / max(settling_time_s, 1e-6))
        )

    y = target_value + transient
    return y, 0.0, "type3_Noisy_Flat_Settling"


def generate_stationary_noisy_flat(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 3b: stationary flat signal that is stable from the first sample.

    This mirrors reference signals such as Signal10: the clean waveform stays
    at its final value for the full capture, while the shared noise policy adds
    the visible white-noise floor, small correlation, and slow drift.
    """
    del settling_time_s, limit_low, limit_high, rng
    y = np.full_like(time_vector, target_value, dtype=float)
    return y, 0.0, "type3_Stationary_Noisy_Flat"


def generate_pulse_train(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 5a: Square pulse train (positive only, variable amplitude)."""
    y = np.full_like(time_vector, target_value, dtype=float)
    band = float(limit_high - limit_low)
    t_end = float(time_vector[-1])

    amp_scale = float(rng.uniform(1.2, 2.5) if rng.random() < 0.35 else rng.uniform(0.25, 1.1))
    base_amp = band * amp_scale
    period = float(rng.uniform(t_end / 5.0, t_end / 2.0))
    duty = float(rng.uniform(0.08, 0.20))
    jitter_frac = float(rng.uniform(0.00, 0.08))

    start_after_s = 0.0005
    current_time = (
        max(float(settling_time_s), start_after_s)
        + float(rng.uniform(0.0, period * 0.3))
    )

    prev_amp = None
    has_pulse = False

    while current_time < t_end:
        this_period = period * float(rng.uniform(1.0 - jitter_frac, 1.0 + jitter_frac))
        width = max(this_period * duty * float(rng.uniform(0.85, 1.15)), 1e-6)
        mask = (time_vector >= current_time) & (time_vector < min(current_time + width, t_end))

        if np.any(mask):
            has_pulse = True
            if prev_amp is not None and rng.random() < 0.35:
                this_amp = prev_amp
            else:
                this_amp = base_amp * float(rng.uniform(0.4, 1.6))
                prev_amp = this_amp
            y[mask] += this_amp

        current_time += this_period

    if not has_pulse:
        t_mid = 0.5 * t_end
        mask = (time_vector >= t_mid) & (time_vector < t_mid + 0.05 * t_end)
        y[mask] += base_amp

    pre = time_vector < max(float(settling_time_s), start_after_s)
    y[pre] = target_value
    return y, 0.0, "type5_Square_Pulse_Wave"


def generate_pulse_train_HARD(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 5b: Pulse train with period wander + polarity flip."""
    y = np.full_like(time_vector, target_value)
    band = limit_high - limit_low
    t_end = time_vector[-1]

    period = rng.uniform(t_end / 6, t_end / 2.2)
    cur = rng.uniform(0, period * 0.5)

    while cur < t_end:
        p = period * rng.uniform(0.7, 1.4)
        w = p * rng.uniform(0.08, 0.25)
        amp = band * rng.uniform(0.3, 2.0) * (1 if rng.random() > 0.15 else -1)
        mask = (time_vector >= cur) & (time_vector < cur + w)
        y[mask] += amp
        cur += p

    return y, 0.0, "type5_Square_Pulse_HARD"


# =============================================================================
# 3) Main
# =============================================================================

def iter_generation_plan(n_waves: int, ratios, rng: np.random.Generator):
    """สร้าง shuffled list ของ generator functions ตาม ratio ที่กำหนด."""
    counts = []
    allocated = 0
    for func, r in ratios:
        cnt = int(n_waves * float(r))
        counts.append([func, cnt])
        allocated += cnt

    if n_waves - allocated > 0:
        counts[0][1] += n_waves - allocated

    order = [func for func, cnt in counts for _ in range(cnt)]
    rng.shuffle(order)
    yield from order


def main():
    ap = argparse.ArgumentParser("Generate synthetic waveform training data")
    ap.add_argument("--out",             default="data/raw/data_for_train.csv")
    ap.add_argument("--n_waves", "--n-waves", type=int, default=1000)
    ap.add_argument("--dt_ms",           type=float, default=0.01)
    ap.add_argument("--t_end_ms",        type=float, default=9.9)
    ap.add_argument("--waves_per_flush", type=int,   default=10)
    ap.add_argument("--seed",            type=int,   default=None)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    t_ms = np.arange(0.0, args.t_end_ms + 1e-12, args.dt_ms)
    t_s = t_ms / 1000.0
    n_samples = len(t_ms)

    ratios = [
        (generate_step_response,                0.14),
        (generate_high_start_oscillation,       0.10),
        (generate_overdamped_decay,             0.16),
        (generate_overdamped_decay1,            0.10),
        (generate_slow_tail_biexponential,      0.14),
        (generate_noisy_flat_settling,          0.06),
        (generate_stationary_noisy_flat,        0.08),
        (generate_pulse_train,                  0.08),
        (generate_continuous_triangular_pulses, 0.08),
        (generate_pulse_train_HARD,             0.06),
    ]

    # type ที่ไม่ควรมี settling → บังคับ settle = 0.1ms
    no_settle_funcs = {
        generate_continuous_triangular_pulses,
        generate_stationary_noisy_flat,
        generate_pulse_train,
        generate_pulse_train_HARD,
    }

    master_rng = np.random.default_rng(args.seed)
    gen_sequence = iter_generation_plan(args.n_waves, ratios, master_rng)

    print(f"Generating {args.n_waves} waves × {n_samples} samples = {args.n_waves*n_samples:,} rows")
    print(f"Output: {out_path}")

    wrote_header = False
    batch_frames = []

    for wave_id, gen_func in enumerate(gen_sequence, start=1):

        # --- Sample parameters ---
        final_value = sample_target_mixed_units(master_rng)
        mag = max(abs(final_value), 1e-12)
        band = max(mag * float(master_rng.uniform(0.05, 0.15)), mag * 0.02)
        low  = final_value - band / 2.0
        high = final_value + band / 2.0

        # --- Sample settling time ---
        t_end_ms = float(args.t_end_ms)
        max_settle_ms = 0.92 * t_end_ms
        p = master_rng.random()
        if p < 0.78:
            settle_time_ms = float(master_rng.uniform(1.5, max(0.55 * max_settle_ms, 1.7)))
        elif p < 0.96:
            settle_time_ms = float(master_rng.uniform(0.55 * max_settle_ms, 0.85 * max_settle_ms))
        else:
            settle_time_ms = float(master_rng.uniform(0.85 * max_settle_ms, max_settle_ms))

        if gen_func in no_settle_funcs:
            settle_time_ms = 0.1

        settle_s = settle_time_ms / 1000.0

        # --- Generate waveform ---
        wave_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        y, used_sd, type_name = gen_func(t_s, final_value, settle_s, low, high, wave_rng)

        # --- Shared noise policy for every generator ---
        y, used_sd = add_signal_sample_noise(
            y,
            final_value,
            wave_rng,
        )

        # Slow tails are intentionally not flattened at the sampled shape time.
        # Label the point where the generated tail is actually stable instead.
        if gen_func is generate_slow_tail_biexponential:
            measured_label_ms = estimate_slow_tail_label_ms(t_ms, y, used_sd)
            settle_time_ms = max(settle_time_ms, measured_label_ms)
            settle_s = settle_time_ms / 1000.0

        # --- Compute post-settle limits ---
        si = int(np.searchsorted(t_s, settle_s))
        post = y[si:] if si < len(y) else y
        low_settle  = float(np.min(post))
        high_settle = float(np.max(post))

        if len(y) != n_samples:
            raise RuntimeError(
                f"len(y)={len(y)} expected={n_samples} (wave_id={wave_id}, type={type_name})"
            )

        # --- Build DataFrame ---
        dfw = pd.DataFrame({
            "wave_id":    np.full(n_samples, wave_id,          dtype=np.int32),
            "type":       np.full(n_samples, type_name,        dtype=object),
            "sample":     np.arange(n_samples,                 dtype=np.int32),
            "time_ms":    t_ms.astype(np.float32),
            "value":      np.asarray(y,                        dtype=np.float64),
            "sd":         np.full(n_samples, float(used_sd),   dtype=np.float64),
            "low_limit":  np.full(n_samples, low_settle,       dtype=np.float64),
            "high_limit": np.full(n_samples, high_settle,      dtype=np.float64),
            "wait_time_ms": np.full(n_samples, float(settle_time_ms), dtype=np.float32),
        })

        batch_frames.append(dfw)

        if (wave_id % args.waves_per_flush) == 0:
            pd.concat(batch_frames, ignore_index=True).to_csv(
                out_path, mode="a", index=False, header=(not wrote_header)
            )
            wrote_header = True
            batch_frames.clear()

    # flush ที่เหลือ
    if batch_frames:
        pd.concat(batch_frames, ignore_index=True).to_csv(
            out_path, mode="a", index=False, header=(not wrote_header)
        )

    print(f"Done. Saved: {out_path}")


if __name__ == "__main__":
    main()
