"""
Groundwater time-series fingerprinting (wavelet + meta + seasonality)

Main entry:
  fingerprint = compute_fingerprint(times_ms, values, params=None)
  -> returns dict with keys:
     - 'vector' : 1D numpy array (fingerprint)
     - 'features': dict of named features
     - 'flags': dict of boolean flags
     - 'params': used parameters
"""

import numpy as np
import pywt
from scipy import stats
from scipy.signal import lombscargle

EPS = 1e-12

def _prepare(times_ms, values):
    times_ms = np.asarray(times_ms).astype(float)
    vals = np.asarray(values).astype(float)
    order = np.argsort(times_ms)
    times_ms = times_ms[order]
    vals = vals[order]
    # drop nan pairs
    mask = ~np.isnan(times_ms) & ~np.isnan(vals)
    return times_ms[mask], vals[mask]

def _times_ms_to_days(times_ms):
    return times_ms / 1000.0 / 86400.0

def _gap_threshold_days(median_dt_days):
    # small gaps tolerated; large gaps split segments
    return max(10.0 * max(median_dt_days, 1e-6), 30.0)  # at least 30 days

def _gap_aware_resample(times_days, vals, N, gap_thresh_days):
    """
    Create N uniformly spaced target times between t0 and t1 (in days).
    Interpolate linearly only when the bracket interval <= gap_thresh_days.
    Else put NaN.
    """
    t0, t1 = times_days[0], times_days[-1]
    target = np.linspace(t0, t1, N)
    idx = np.searchsorted(times_days, target)  # right index
    left = idx - 1
    right = idx.copy()
    res = np.full_like(target, np.nan, dtype=float)

    # exact matches -> copy value
    exact_mask = (idx < len(times_days)) & (times_days[idx] == target)
    res[exact_mask] = vals[idx[exact_mask]]

    # valid interpolation positions: left >=0, right < len, not exact match
    interp_mask = (~exact_mask) & (left >= 0) & (right < len(times_days))
    if np.any(interp_mask):
        l = left[interp_mask]
        r = right[interp_mask]
        gap = times_days[r] - times_days[l]
        good = gap <= gap_thresh_days
        if np.any(good):
            sel = np.where(interp_mask)[0][good]
            l = left[sel]; r = right[sel]
            t_l = times_days[l]; t_r = times_days[r]
            v_l = vals[l]; v_r = vals[r]
            frac = (target[sel] - t_l) / (t_r - t_l + EPS)
            res[sel] = v_l + frac * (v_r - v_l)
    return target, res

def _fill_small_nans(arr):
    # arr: 1D with some NaNs, assume endpoints are inside range
    x = np.arange(len(arr))
    good = ~np.isnan(arr)
    if good.sum() == 0:
        return arr.copy()
    filled = np.interp(x, x[good], arr[good])
    return filled

def _longest_contiguous_segment(times_days, vals, gap_thresh_days):
    # Returns times, vals of the longest contiguous segment where dt <= gap_thresh_days
    dt = np.diff(times_days)
    breaks = np.where(dt > gap_thresh_days)[0]
    segments = []
    start = 0
    for b in breaks:
        segments.append((start, b))
        start = b + 1
    segments.append((start, len(times_days) - 1))
    # compute lengths
    best = max(segments, key=lambda s: s[1] - s[0])
    i0, i1 = best
    return times_days[i0:i1+1], vals[i0:i1+1]

def _wavelet_summary(signal, wavelet_name='db4', max_level=6):
    # returns lists of per-band (relative_energy, meanabs, maxpos)
    N = len(signal)
    # determine allowed level
    allowed = pywt.dwt_max_level(N, pywt.Wavelet(wavelet_name).dec_len)
    L = min(max_level, allowed)
    coeffs = pywt.wavedec(signal, wavelet_name, level=L)
    energies = [np.sum(c.astype(float)**2) for c in coeffs]
    total = sum(energies) + EPS
    rel_energies = [e/total for e in energies]
    meanabs = [np.mean(np.abs(c)) if len(c)>0 else 0.0 for c in coeffs]
    maxpos = [ (np.argmax(np.abs(c))/len(c)) if len(c)>0 else 0.0 for c in coeffs]
    # order: coeffs[0]=approx_highest, then detail bands
    return np.array(rel_energies), np.array(meanabs), np.array(maxpos), L

def _seasonal_lombscargle(times_days, vals, require_points=8):
    """
    Returns (annual_power_abs, annual_power_ratio).
    times_days: absolute days -> convert to relative (start=0)
    """
    if len(times_days) < require_points:
        return 0.0, 0.0
    t = times_days - times_days[0]
    y = np.asarray(vals) - np.nanmean(vals)
    # frequency in cycles/day for 1/year
    f_annual = 1.0 / 365.25
    # narrow band around annual
    freqs_ann = f_annual * np.linspace(0.9, 1.1, 60)
    w_ann = 2.0 * np.pi * freqs_ann
    try:
        p_ann = lombscargle(t, y, w_ann, precenter=False)
    except Exception:
        # fallback small arrays or numerical issues
        return 0.0, 0.0
    seasonal_power = np.max(p_ann) if np.any(np.isfinite(p_ann)) else 0.0

    # broad band for normalization: 0.1/year .. 2/year
    freqs_broad = np.linspace(0.1/365.25, 2.0/365.25, 300)
    w_broad = 2.0 * np.pi * freqs_broad
    p_broad = lombscargle(t, y, w_broad, precenter=False)
    total_power = np.sum(p_broad) + EPS
    ratio = seasonal_power / total_power
    return float(seasonal_power), float(ratio)

def compute_fingerprint(times_ms, values, params=None):
    """
    Returns a dict with vector, features, flags, params.
    params is optional dict controlling:
      - N_resample (default 512)
      - wavelet ('db4')
      - max_wavelet_level (default 6)
      - nan_frac_threshold (default 0.30)
      - min_points_for_wavelet (default 10)

    Example:
        t = sample[:,0]; v = sample[:,1]
        res = compute_fingerprint(t, v)
        print("fingerprint length:", res['vector'].size)
        print("vector:", res['vector'])
        print("flags:", res['flags'])
        print("features keys:", res['features'].keys())
    """
    if params is None:
        params = {}
    N = int(params.get('N_resample', 512))
    wavelet = params.get('wavelet', 'db4')
    max_level = int(params.get('max_wavelet_level', 6))
    nan_frac_threshold = float(params.get('nan_frac_threshold', 0.30))
    min_points_for_wavelet = int(params.get('min_points_for_wavelet', 10))

    times_ms, vals = _prepare(times_ms, values)
    features = {}
    flags = {'too_short': False, 'lots_of_gaps': False, 'is_roughly_constant': False}
    if len(times_ms) == 0:
        raise ValueError("Empty series after removing NaNs")

    # convert to days
    times_days = _times_ms_to_days(times_ms)
    n_points = len(times_days)
    dt_days = np.diff(times_days) if n_points > 1 else np.array([0.0])
    median_dt = float(np.median(dt_days)) if len(dt_days) > 0 else 0.0
    min_dt = float(np.min(dt_days)) if len(dt_days) > 0 else 0.0
    max_dt = float(np.max(dt_days)) if len(dt_days) > 0 else 0.0
    pct_gaps = float(np.mean(dt_days > 2.0 * max(median_dt, 1e-9))) if len(dt_days)>0 else 0.0

    # meta features
    start_year = int(np.floor((times_days[0] / 365.25) + 1970))  # days since epoch -> crude year
    end_year = int(np.floor((times_days[-1] / 365.25) + 1970))
    span_days = float(times_days[-1] - times_days[0])

    features.update({
        'n_points': int(n_points),
        'start_year': start_year,
        'end_year': end_year,
        'span_days': span_days,
        'median_dt_days': median_dt,
        'min_dt_days': min_dt,
        'max_dt_days': max_dt,
        'pct_large_gaps': pct_gaps
    })

    if n_points < min_points_for_wavelet:
        flags['too_short'] = True

    # detect near-constant series
    if np.nanstd(vals) < 1e-6:
        flags['is_roughly_constant'] = True

    gap_thresh = _gap_threshold_days(median_dt)

    # resample gap-aware
    target_times, resampled = _gap_aware_resample(times_days, vals, N, gap_thresh)
    nan_frac = float(np.mean(np.isnan(resampled)))
    features['nan_frac_after_resample'] = nan_frac

    wavelet_rel_energy = np.zeros(max_level + 1)
    wavelet_meanabs = np.zeros(max_level + 1)
    wavelet_maxpos = np.zeros(max_level + 1)
    used_level = 0

    if nan_frac <= nan_frac_threshold and not flags['too_short']:
        # safe to fill small NaNs and run wavelet on full resampled signal
        filled = _fill_small_nans(resampled)
        # detrend small linear trend before wavelet (helps)
        detrended = filled - np.polyval(np.polyfit(np.arange(len(filled)), filled, 1), np.arange(len(filled)))
        re, ma, mp, used_level = _wavelet_summary(detrended, wavelet_name=wavelet, max_level=max_level)
        wavelet_rel_energy[:len(re)] = re
        wavelet_meanabs[:len(ma)] = ma
        wavelet_maxpos[:len(mp)] = mp
    else:
        # lots of gaps: fallback to longest contiguous original segment
        flags['lots_of_gaps'] = True
        seg_t, seg_v = _longest_contiguous_segment(times_days, vals, gap_thresh)
        if len(seg_t) >= min_points_for_wavelet:
            # resample this segment uniformly
            seg_target = np.linspace(seg_t[0], seg_t[-1], N)
            seg_resampled = np.interp(seg_target, seg_t, seg_v)
            detrended = seg_resampled - np.polyval(np.polyfit(np.arange(len(seg_resampled)), seg_resampled, 1), np.arange(len(seg_resampled)))
            re, ma, mp, used_level = _wavelet_summary(detrended, wavelet_name=wavelet, max_level=max_level)
            wavelet_rel_energy[:len(re)] = re
            wavelet_meanabs[:len(ma)] = ma
            wavelet_maxpos[:len(mp)] = mp
        else:
            # cannot compute wavelet reliably
            flags['too_short'] = True
            # leave wavelet arrays as zeros

    # global statistics (on original values)
    mean_v = float(np.nanmean(vals))
    std_v = float(np.nanstd(vals))
    skew_v = float(stats.skew(vals)) if len(vals) > 2 else 0.0
    kurt_v = float(stats.kurtosis(vals)) if len(vals) > 3 else 0.0
    qs = np.nanpercentile(vals, [10,25,50,75,90]).tolist()

    features.update({
        'mean': mean_v, 'std': std_v, 'skewness': skew_v, 'kurtosis': kurt_v,
        'q10': qs[0], 'q25': qs[1], 'q50': qs[2], 'q75': qs[3], 'q90': qs[4],
        'used_wavelet_levels': int(used_level)
    })

    # seasonality from original time series (Lomb-Scargle)
    seasonal_abs, seasonal_ratio = _seasonal_lombscargle(times_days, vals, require_points=8)
    features['seasonal_power_abs'] = seasonal_abs
    features['seasonal_power_ratio'] = seasonal_ratio

    # assemble fingerprint vector (concise, deterministic order)
    vec_parts = []
    # wavelet: for levels 0..used_level -> rel_energy, meanabs, maxpos
    for i in range(used_level + 1):
        vec_parts.append(wavelet_rel_energy[i])
        vec_parts.append(wavelet_meanabs[i])
        vec_parts.append(wavelet_maxpos[i])
    # pad if used_level < max_level
    # then add global stats
    vec_parts.extend([mean_v, std_v, skew_v, kurt_v])
    vec_parts.extend(qs)
    # time/meta numeric (normalize start/end years to small numbers)
    vec_parts.extend([features['start_year'], features['end_year'], span_days, n_points, median_dt, pct_gaps])
    # seasonality
    vec_parts.extend([seasonal_abs, seasonal_ratio])
    # flags as ints
    vec_parts.extend([int(flags['too_short']), int(flags['lots_of_gaps']), int(flags['is_roughly_constant'])])

    vector = np.asarray(vec_parts, dtype=float)

    # final size check (must be less than 100 by requirement)
    if vector.size >= 100:
        # compress by summarizing wavelet bands further (fallback)
        # keep first 6 bands only
        keep_bands = min(6, used_level+1)
        wp = []
        for i in range(keep_bands):
            wp += [wavelet_rel_energy[i], wavelet_meanabs[i], wavelet_maxpos[i]]
        # rebuild vector with truncated wavelet info
        vec = wp + [mean_v, std_v, skew_v, kurt_v] + qs + [features['start_year'], features['end_year'], span_days, n_points, median_dt, pct_gaps] + [seasonal_abs, seasonal_ratio] + [int(flags['too_short']), int(flags['lots_of_gaps']), int(flags['is_roughly_constant'])]
        vector = np.asarray(vec, dtype=float)

    out = {
        'vector': vector,
        'features': features,
        'flags': flags,
        'params': {
            'N_resample': N,
            'wavelet': wavelet,
            'max_wavelet_level': max_level,
            'nan_frac_threshold': nan_frac_threshold,
            'min_points_for_wavelet': min_points_for_wavelet
        }
    }
    return out
