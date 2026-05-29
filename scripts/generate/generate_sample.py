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


def unit_noise_boost(target_value: float, *, extreme: bool = False) -> float:
    """
    Smaller-unit signals get more relative noise.
    Normal-scale values stay near 1x; micro/nano-scale values are boosted.
    """
    mag = max(abs(float(target_value)), 1e-12)
    if mag >= 1e-3:
        return 1.0
    if mag >= 1e-6:
        exponent = 0.10 if extreme else 0.07
        cap = 3.0 if extreme else 2.0
        return float(np.clip((1e-3 / mag) ** exponent, 1.0, cap))

    if extreme:
        return float(np.clip(2.4 * (1e-6 / mag) ** 0.12, 2.4, 9.0))
    return float(np.clip(1.45 * (1e-6 / mag) ** 0.08, 1.45, 3.2))


def range_noise_boost(signal_swing: float, band: float) -> float:
    """
    If waveform swing is tiny compared with the value band, make noise more visible.
    This mimics the flat-but-noisy SignalSample waves.
    """
    band = max(abs(float(band)), 1e-15)
    swing_ratio = max(abs(float(signal_swing)) / band, 1e-6)
    return float(np.clip((0.08 / swing_ratio) ** 0.45, 1.0, 8.0))


def value_range_noise_boost(
    target_value: float,
    signal_swing: float,
    band: float,
    *,
    extreme: bool = False,
) -> float:
    return unit_noise_boost(target_value, extreme=extreme) * range_noise_boost(signal_swing, band)


def sample_noise_profile_scale(rng: np.random.Generator) -> float:
    """Mix low, normal, and reference-heavy noise without changing waveform type."""
    p = rng.random()
    if p < 0.01:
        return float(rng.uniform(0.45, 0.85))
    if p < 0.52:
        return float(rng.uniform(1.15, 2.05))
    return float(rng.uniform(2.20, 6.25))


def smooth_noise(rng: np.random.Generator, n: int, sd: float, window: int) -> np.ndarray:
    window = max(3, min(int(window), max(n - 1, 3)))
    raw = rng.normal(0.0, sd, size=n)
    kernel = np.ones(window) / window
    return np.convolve(raw, kernel, mode="same") * math.sqrt(window)


def smoothstep_mask(time_vector: np.ndarray, center_s: float, width_s: float, floor: float) -> np.ndarray:
    if center_s <= 0.0 or width_s <= 0.0:
        return np.ones_like(time_vector, dtype=float)

    x = (time_vector - (center_s - width_s)) / max(2.0 * width_s, 1e-12)
    x = np.clip(x, 0.0, 1.0)
    eased = x * x * (3.0 - 2.0 * x)
    return floor + (1.0 - floor) * eased


def estimate_signal_sample_wait_time_ms(
    y: np.ndarray,
    t_ms: np.ndarray,
    min_fraction: float = 0.60,
) -> float:
    """
    Estimate a conservative label for slow SignalSample-like relaxations.
    The signal is considered settled only after the smoothed curve stays near
    its tail baseline; otherwise the label moves close to the end of the trace.
    """
    if len(y) < 10:
        return float(t_ms[-1])

    n = len(y)
    win = max(9, min(61, (n // 25) | 1))
    kernel = np.ones(win) / win
    smooth = np.convolve(np.asarray(y, dtype=float), kernel, mode="same")

    edge = win // 2
    if edge > 0:
        smooth[:edge] = smooth[edge]
        smooth[-edge:] = smooth[-edge - 1]

    tail_start = max(int(0.85 * n), n - max(20, win))
    tail = smooth[tail_start:]
    tail_baseline = float(np.median(tail))
    tail_noise = 1.4826 * float(np.median(np.abs(tail - tail_baseline)))
    value_range = float(np.ptp(smooth))
    tolerance = max(3.0 * tail_noise, 0.012 * value_range, 1e-15)

    abs_delta = np.abs(smooth - tail_baseline)
    suffix_max = np.maximum.accumulate(abs_delta[::-1])[::-1]
    first_allowed = min(max(int(min_fraction * n), 0), n - 1)
    candidates = np.flatnonzero(suffix_max[first_allowed:] <= tolerance)
    if len(candidates) == 0:
        return float(t_ms[-1])

    label_idx = first_allowed + int(candidates[0])
    return float(t_ms[label_idx])


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
    noise_strength: float = 1.0,
    micro_noise_strength: float = 1.0,
    normal_noise_strength: float = 1.0,
    jitter_probability_boost: float = 1.0,
    steady_jitter_strength: float = 0.0,
    rare_heavy_jitter_probability: float = 0.0,
    rare_heavy_jitter_strength: float = 1.0,
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
    signal_span = max(float(np.ptp(signal_array)), mag * 0.02, 1e-15)
    floor_abs = max(mag * 1e-4, signal_span * 2e-5, 1e-15)
    noise_strength = max(float(noise_strength), 0.0)
    micro_noise_strength = max(float(micro_noise_strength), 0.0)
    normal_noise_strength = max(float(normal_noise_strength), 0.0)
    steady_jitter_strength = max(float(steady_jitter_strength), 0.0)
    rare_heavy_jitter_probability = float(np.clip(rare_heavy_jitter_probability, 0.0, 1.0))
    rare_heavy_jitter_strength = max(float(rare_heavy_jitter_strength), 0.0)
    is_micro_value = 1e-6 <= mag < 1e-3
    is_normal_value = mag >= 1e-3
    effective_strength = noise_strength
    if is_micro_value:
        effective_strength *= micro_noise_strength
    elif is_normal_value:
        effective_strength *= normal_noise_strength
    unit_scale = unit_noise_boost(target_value, extreme=False)
    noise_scale = unit_scale * sample_noise_profile_scale(rng) * effective_strength
    tiny_cap_boost = float(np.clip(unit_scale / 1.6, 1.0, 6.0))
    micro_tail_boost = 1.0 + (0.45 * max(micro_noise_strength - 1.0, 0.0) if is_micro_value else 0.0)
    micro_cap_boost = 1.0 + (0.65 * max(micro_noise_strength - 1.0, 0.0) if is_micro_value else 0.0)
    if mag >= 1.0:
        normal_tail_boost = 1.65
        normal_cap_boost = 1.85
    elif mag >= 1e-3:
        normal_tail_boost = 1.35
        normal_cap_boost = 1.45
    else:
        normal_tail_boost = 1.0
        normal_cap_boost = 1.0
    n = len(time_vector)

    # --- 1) Base floor noise (ทั้ง signal) ---
    base_floor_sd = max(
        mag * rng.uniform(5e-4, 1.5e-3) * noise_scale,
        signal_span * rng.uniform(4e-4, 1.2e-3) * min(noise_scale, 3.2),
        floor_abs,
    )
    signal_array = signal_array + rng.normal(0.0, base_floor_sd * 0.85, size=n)
    final_sd = base_floor_sd

    settle_idx = int(np.searchsorted(time_vector, settling_time_s))
    remaining_len = len(time_vector) - settle_idx
    if remaining_len <= 5:
        return signal_array, final_sd

    dt_s = float(time_vector[1] - time_vector[0]) if len(time_vector) > 1 else 1e-5
    fade_width_s = max(0.00035, 35.0 * dt_s)
    floor_level = 0.80 if noise_scale >= 1.75 else 0.35
    envelope = smoothstep_mask(time_vector, settling_time_s, fade_width_s, floor_level)

    # --- 2) Correlated wiggle (post-settle) ---
    if rng.random() < probability:
        post_sd = max(mag * rng.uniform(*post_sd_scale) * noise_scale, floor_abs)
        final_sd = max(final_sd, post_sd)

        smoothness = int(rng.integers(smoothness_range[0], smoothness_range[1] + 1))
        wig = smooth_noise(rng, n, post_sd, smoothness) * 0.55
        signal_array += wig * envelope

    # --- 3) Optional wobble (post-settle) ---
    if rng.random() < add_wobble_prob:
        wob_sd = max(mag * rng.uniform(*wobble_scale) * noise_scale, floor_abs)
        final_sd = max(final_sd, wob_sd)

        win = int(rng.integers(wobble_win_range[0], wobble_win_range[1] + 1))
        wob = smooth_noise(rng, n, wob_sd, win)
        wob = (wob - np.mean(wob)) * 0.25
        signal_array += wob * envelope

    # --- 4) Alternating noise → AC(1) ≈ -0.4 (เหมือน ADC quantization) ---
    quant_sd = max(mag * rng.uniform(2e-4, 8e-4) * noise_scale, floor_abs)
    quant_noise = smooth_noise(rng, n, quant_sd, int(rng.integers(3, 8)))
    signal_array += quant_noise * envelope * 0.18

    # Always add a small tail texture so settled traces do not become ruler-flat.
    texture_sd = max(
        mag * rng.uniform(1.8e-3, 4.8e-3) * max(noise_scale, 1.25 * effective_strength) * normal_tail_boost * micro_tail_boost,
        signal_span * rng.uniform(1.0e-3, 2.8e-3) * max(noise_scale, 1.15 * effective_strength) * normal_tail_boost * micro_tail_boost,
        floor_abs,
    )
    texture_sd = min(
        texture_sd,
        max(
            mag * 0.0150 * tiny_cap_boost * normal_cap_boost * micro_cap_boost,
            signal_span * 0.0100 * tiny_cap_boost * normal_cap_boost * micro_cap_boost,
            floor_abs,
        ),
    )
    tail_texture = rng.normal(0.0, texture_sd, size=n)
    tail_texture = 0.78 * tail_texture + 0.22 * smooth_noise(
        rng, n, texture_sd, int(rng.integers(3, 6))
    )
    signal_array += tail_texture * envelope
    final_sd = max(final_sd, texture_sd)

    if steady_jitter_strength > 0.0:
        steady_sd = max(
            mag * 1.2e-3 * effective_strength * steady_jitter_strength,
            signal_span * 9e-4 * effective_strength * steady_jitter_strength,
            floor_abs,
        )
        steady_sd = min(
            steady_sd,
            max(
                mag * 0.0100 * tiny_cap_boost * normal_cap_boost * micro_cap_boost,
                signal_span * 0.0070 * tiny_cap_boost * normal_cap_boost * micro_cap_boost,
                floor_abs,
            ),
        )
        steady_jitter = rng.normal(0.0, steady_sd, size=n)
        steady_jitter = 0.82 * steady_jitter + 0.18 * smooth_noise(
            rng, n, steady_sd, int(rng.integers(3, 5))
        )
        signal_array += steady_jitter * envelope
        final_sd = max(final_sd, steady_sd)

    if rare_heavy_jitter_probability > 0.0 and rng.random() < rare_heavy_jitter_probability:
        heavy_sd = max(
            mag * rng.uniform(0.010, 0.026) * rare_heavy_jitter_strength,
            signal_span * rng.uniform(0.007, 0.020) * rare_heavy_jitter_strength,
            floor_abs,
        )
        heavy_sd = min(
            heavy_sd,
            max(
                mag * 0.060 * tiny_cap_boost * normal_cap_boost * micro_cap_boost,
                signal_span * 0.045 * tiny_cap_boost * normal_cap_boost * micro_cap_boost,
                floor_abs,
            ),
        )
        heavy_jitter = rng.normal(0.0, heavy_sd, size=n)
        heavy_jitter = 0.70 * heavy_jitter + 0.30 * smooth_noise(
            rng, n, heavy_sd, int(rng.integers(4, 9))
        )
        slow_jitter = smooth_noise(rng, n, heavy_sd * 0.80, int(rng.integers(18, 38)))
        signal_array += (heavy_jitter + slow_jitter * 0.45) * envelope
        final_sd = max(final_sd, heavy_sd)

    # Reference-like fine jitter: dense vibration similar to SignalSample tails.
    jitter_probability = min(0.86, (0.50 + 0.06 * noise_scale) * float(jitter_probability_boost))
    if rng.random() < jitter_probability:
        jitter_scale = min(noise_scale, 3.4)
        jitter_sd = max(
            mag * rng.uniform(1.1e-3, 3.8e-3) * jitter_scale * normal_tail_boost * micro_tail_boost,
            signal_span * rng.uniform(6e-4, 2.0e-3) * min(jitter_scale, 2.5) * normal_tail_boost * micro_tail_boost,
            floor_abs,
        )
        jitter_sd = min(
            jitter_sd,
            max(
                mag * 0.0120 * tiny_cap_boost * normal_cap_boost * micro_cap_boost,
                signal_span * 0.0080 * tiny_cap_boost * normal_cap_boost * micro_cap_boost,
                floor_abs,
            ),
        )

        fine_jitter = rng.normal(0.0, jitter_sd, size=n)
        connected_jitter = smooth_noise(rng, n, jitter_sd, int(rng.integers(3, 7)))
        signal_array += (fine_jitter * 0.68 + connected_jitter * 0.32) * envelope
        final_sd = max(final_sd, jitter_sd)

    # --- 5) Slow low-freq drift → AC(5) > 0 (เหมือน thermal drift) ---
    drift_sd = max(mag * rng.uniform(1e-4, 4e-4) * noise_scale, floor_abs)
    win_drift = int(rng.integers(40, 80))
    drift = smooth_noise(rng, n, drift_sd, win_drift)
    signal_array += drift * envelope

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
        return float(10 ** rng.uniform(-9, -8))
    elif p < 0.35:
        return float(10 ** rng.uniform(-6, -5))
    elif p < 0.85:
        return float(10 ** rng.uniform(-3, 0))
    else:
        return float(rng.uniform(1.0, 50.0))


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
    noise_boost = float(wave_rng.uniform(2.0, 4.0)) * value_range_noise_boost(
        final_value, signal_swing, band
    )
    boosted_sd = max(mag * wave_rng.uniform(0.0012, 0.0050), 1e-12) * noise_boost

    n = len(y)
    if n <= 5:
        return y, boosted_sd

    # Whole-trace reference-like jitter: visible but without an abrupt noisy block.
    fine_jitter = wave_rng.normal(0.0, boosted_sd, size=n)
    connected_jitter = smooth_noise(wave_rng, n, boosted_sd, int(wave_rng.integers(3, 8)))
    smooth_win = int(wave_rng.integers(40, 80))
    drift = smooth_noise(wave_rng, n, boosted_sd, smooth_win)

    strong_jitter = wave_rng.random() < 0.70
    fine_weight = 0.52 if strong_jitter else 0.28
    connected_weight = 0.34 if strong_jitter else 0.28
    drift_weight = 0.14 if strong_jitter else 0.44
    y += fine_jitter * fine_weight + connected_jitter * connected_weight + drift * drift_weight
    return y, boosted_sd


def add_signal_sample_noise(
    y: np.ndarray,
    target_value: float,
    signal_swing: float,
    band: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Add mixed white/alternating/drift noise like the reference SignalSample set."""
    n = len(y)
    mag = max(abs(float(target_value)), 1e-12)
    boost = value_range_noise_boost(target_value, signal_swing, band, extreme=True)
    base_sd = max(
        mag * rng.uniform(0.00035, 0.0014),
        abs(float(band)) * rng.uniform(0.0025, 0.0100),
        1e-15,
    )
    sd = min(base_sd * boost, max(abs(float(band)) * 0.18, mag * 0.018, 1e-15))

    white = rng.normal(0.0, sd, size=n)

    alt_signs = np.empty(n, dtype=float)
    alt_signs[0::2] = 1.0
    alt_signs[1::2] = -1.0
    alternating = rng.normal(0.0, sd * rng.uniform(0.20, 0.45), size=n) * alt_signs

    drift_win = int(rng.integers(35, 95))
    drift_win = max(3, min(drift_win, n - 1))
    k = np.ones(drift_win) / drift_win
    drift = np.convolve(rng.normal(0.0, sd * 0.8, size=n), k, mode="same") * math.sqrt(drift_win)

    return y + white + alternating + drift * 0.35, sd


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

    overshoot_scale = float(rng.uniform(1.5, 8.0 if target_value < 1.0 else 15.0))
    direction = float(rng.choice([1.0, -1.0]))
    amp0 = band_half * overshoot_scale * direction

    tau = max(settling_time_s / float(rng.uniform(2.5, 6.0)), 1e-6)
    tc_rise = max(settling_time_s / float(rng.uniform(3.0, 10.0)), 1e-6)
    t_eff, _ = add_time_delay(time_vector, rng, max_delay_s=0.00004)

    if rng.random() < 0.5:
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
        strength=float(rng.uniform(0.65, 0.70))
    )
    y = flatten_after_settle(y, time_vector, settling_time_s)

    y, sd = add_post_settle_noise(
        y, time_vector, settling_time_s, target_value, rng,
        probability=0.95,
        smoothness_range=(10, 28),
        post_sd_scale=(0.0009, 0.0016),
        add_wobble_prob=0.35,
        wobble_scale=(0.00012, 0.00030),
        noise_strength=1.25,
        micro_noise_strength=1.8,
        normal_noise_strength=1.35,
        jitter_probability_boost=1.35,
        steady_jitter_strength=1.0,
        rare_heavy_jitter_probability=0.20,
        rare_heavy_jitter_strength=1.0,
    )
    return y, sd, "type0_Step_Response"


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

    y, sd = add_post_settle_noise(
        y, t, settling_time_s, target_value, rng,
        probability=0.70,
        post_sd_scale=(0.00035, 0.00075),
        smoothness_range=(20, 38),
        add_wobble_prob=0.04,
        wobble_scale=(0.00005, 0.00012),
        wobble_win_range=(55, 110),
    )
    return y, sd, "type1_Damped_Osc"


def generate_continuous_triangular_pulses(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 2: Triangular pulse train (period อิงจาก t_end ไม่ใช่ settle)."""
    y = np.full_like(time_vector, target_value, dtype=float)
    band = float(limit_high - limit_low)
    avg_height = band * float(rng.uniform(0.5, 1.5))
    t_end = float(time_vector[-1])

    avg_period = float(rng.uniform(t_end / 20.0, t_end / 6.0))
    pulse_width = avg_period * float(rng.uniform(0.15, 0.30))
    is_height_const = bool(rng.choice([True, False]))
    is_period_const = bool(rng.choice([True, False]))

    start_after_s = 0.0005
    current_time = (
        max(float(settling_time_s), start_after_s)
        + float(rng.uniform(0.0, avg_period * 0.3))
    )

    while current_time < t_end:
        height = avg_height if is_height_const else avg_height * float(rng.uniform(0.5, 1.5))
        t_start = current_time
        t_peak = current_time + pulse_width / 2.0
        t_end_pulse = current_time + pulse_width

        rise = (time_vector >= t_start) & (time_vector < t_peak)
        if np.any(rise):
            y[rise] += (height / (pulse_width / 2.0)) * (time_vector[rise] - t_start)

        fall = (time_vector >= t_peak) & (time_vector < t_end_pulse)
        if np.any(fall):
            y[fall] += height - (height / (pulse_width / 2.0)) * (time_vector[fall] - t_peak)

        period = avg_period if is_period_const else avg_period * float(rng.uniform(0.7, 1.3))
        current_time += period

    mag = max(abs(float(target_value)), 1e-12)
    floor_abs = max(mag * 1e-4, 1e-15)
    sd = max(mag * float(rng.uniform(0.00005, 0.0002)), floor_abs)
    y += rng.normal(0.0, sd, size=len(time_vector))

    early = time_vector < (0.2 * time_vector[-1])
    y[early] = np.maximum(y[early], target_value)
    pre = time_vector < max(float(settling_time_s), start_after_s)
    y[pre] = target_value + rng.normal(0.0, sd, size=int(np.sum(pre)))

    return y, sd, "type2_Triangle_Wave"


def generate_overdamped_decay(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 4a: Overdamped exponential decay (no ringing)."""
    start_amp = (limit_high - limit_low) * float(rng.uniform(1.5, 3.0))
    tau = settling_time_s / float(rng.uniform(3.0, 5.0))

    y = target_value + start_amp * np.exp(-time_vector / max(tau, 1e-12))
    y = apply_cosine_taper_settling(y, time_vector, settling_time_s, target_value, strength=1.0)
    y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng)
    return y, sd, "type4_overdamped_no_overshoot"


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

    y = target_value + A_pos * np.exp(-t_eff / tau_fast) - A_neg * np.exp(-t_eff / tau_slow)
    y = apply_cosine_taper_settling(
        y, time_vector, settling_time_s, target_value,
        strength=float(rng.uniform(0.55, 0.8))
    )
    y = flatten_after_settle(y, time_vector, settling_time_s)
    y, sd = add_post_settle_noise(y, time_vector, settling_time_s, target_value, rng)
    return y, sd, "type4_overdamped_decay_overshoot"


def generate_signal_sample_relaxation(
    time_vector, target_value, settling_time_s, limit_low, limit_high, rng
):
    """Type 6: SignalSample-like monotonic relaxation with range-aware noise."""
    t = time_vector.astype(float)
    band = max(float(limit_high - limit_low), max(abs(float(target_value)) * 0.02, 1e-15))
    t_end = max(float(t[-1]), 1e-12)

    flat_like = bool(rng.random() < 0.18)
    if flat_like:
        signal_swing = band * float(rng.uniform(0.003, 0.035))
    else:
        signal_swing = band * float(rng.uniform(0.45, 1.25))

    # The reference set decays for positive targets and rises for negative ones.
    # Flat-like waves can go either way because noise dominates their tiny trend.
    if flat_like:
        direction = float(rng.choice([1.0, -1.0]))
    else:
        direction = 1.0 if target_value >= 0 else -1.0

    if flat_like:
        # Already stable: only a tiny continuous drift is allowed, no late burst.
        normalized_t = t / t_end
        trend = direction * signal_swing * float(rng.uniform(0.05, 0.25)) * (1.0 - normalized_t)
        slow_drift = signal_swing * float(rng.uniform(0.02, 0.08)) * np.sin(
            2.0 * np.pi * normalized_t * float(rng.uniform(0.7, 1.6)) + float(rng.uniform(0.0, 2.0 * np.pi))
        )
        y = target_value + trend + slow_drift
        type_name = "type6_SignalSample_FlatNoise"
    else:
        t_eff, _ = add_time_delay(t, rng, max_delay_s=min(0.00004, 0.01 * t_end))
        tau_fast = max(settling_time_s / float(rng.uniform(2.5, 7.0)), 0.015 * t_end, 1e-8)
        tau_slow = max(t_end * float(rng.uniform(0.25, 0.95)), tau_fast * 1.5, 1e-8)
        mix = float(rng.uniform(0.55, 0.88))

        envelope = mix * np.exp(-t_eff / tau_fast) + (1.0 - mix) * np.exp(-t_eff / tau_slow)
        y = target_value + direction * signal_swing * envelope

        if rng.random() < 0.35:
            bump_center = float(rng.uniform(0.35, 0.75)) * t_end
            bump_width = float(rng.uniform(0.035, 0.10)) * t_end
            bump_amp = signal_swing * float(rng.uniform(-0.035, 0.035))
            y += bump_amp * np.exp(-0.5 * ((t - bump_center) / max(bump_width, 1e-12)) ** 2)
        type_name = "type6_SignalSample_Relaxation"

    y, sd = add_signal_sample_noise(
        y, target_value=target_value, signal_swing=signal_swing, band=band, rng=rng
    )
    return y, sd, type_name


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

    y, sd = add_post_settle_noise(
        y, time_vector, settling_time_s=0.0, target_value=target_value,
        rng=rng, probability=0.0, add_wobble_prob=0.0,
        noise_strength=0.28,
    )
    pre = time_vector < max(float(settling_time_s), start_after_s)
    pre_count = int(np.sum(pre))
    if pre_count:
        pre_sd = max(sd * 0.65, max(abs(float(target_value)), 1e-12) * 8e-4)
        y[pre] = target_value + rng.normal(0.0, pre_sd, size=pre_count)
    return y, sd, "type5_Square_Pulse_Wave"


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

    y, sd = add_post_settle_noise(
        y, time_vector, 0.0, target_value, rng, probability=0.0, add_wobble_prob=0.0,
        noise_strength=0.28,
    )
    return y, sd, "type5_Square_Pulse_HARD"


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
    ap.add_argument("--reference_like_ratio", "--reference-like-ratio", type=float, default=0.08)
    ap.add_argument("--seed",            type=int,   default=None)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    t_ms = np.arange(0.0, args.t_end_ms + 1e-12, args.dt_ms)
    t_s = t_ms / 1000.0
    n_samples = len(t_ms)

    base_ratios = [
        (generate_step_response,                0.22),
        (generate_high_start_oscillation,       0.20),
        (generate_overdamped_decay,             0.20),
        (generate_overdamped_decay1,            0.12),
        (generate_pulse_train,                  0.12),
        (generate_continuous_triangular_pulses, 0.07),
        (generate_pulse_train_HARD,             0.07),
    ]
    reference_like_ratio = float(np.clip(args.reference_like_ratio, 0.0, 1.0))
    ratios = [(func, ratio * (1.0 - reference_like_ratio)) for func, ratio in base_ratios]
    if reference_like_ratio > 0.0:
        ratios.append((generate_signal_sample_relaxation, reference_like_ratio))

    # type ที่ไม่ควรมี settling → บังคับ settle = 0.1ms
    no_settle_funcs = {
        generate_continuous_triangular_pulses,
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
        if gen_func is generate_signal_sample_relaxation and master_rng.random() < 0.55:
            final_value *= -1.0
        mag = max(abs(final_value), 1e-12)
        band = max(mag * float(master_rng.uniform(0.05, 0.15)), mag * 0.02)
        low  = final_value - band / 2.0
        high = final_value + band / 2.0

        # --- Sample settling time ---
        t_end_ms = float(args.t_end_ms)
        max_settle_ms = 0.75 * t_end_ms
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
        if type_name == "type6_SignalSample_FlatNoise":
            settle_time_ms = 0.1
            settle_s = settle_time_ms / 1000.0
        elif type_name == "type6_SignalSample_Relaxation":
            settle_time_ms = estimate_signal_sample_wait_time_ms(y, t_ms)
            settle_s = settle_time_ms / 1000.0

        # --- High-noise boost สำหรับ flat signal (เหมือน wave 9/10 ใน real data) ---
        if not type_name.startswith("type6_SignalSample"):
            y, boost_sd = apply_high_noise_boost(
                y, t_s, settle_s, final_value, band, wave_rng
            )
            if boost_sd > 0.0:
                used_sd = max(used_sd, boost_sd)

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
