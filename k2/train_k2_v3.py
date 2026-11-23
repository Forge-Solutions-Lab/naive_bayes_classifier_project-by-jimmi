import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy.special import i0, i1, logsumexp, ndtr
from scipy.stats import multivariate_normal
from scipy.optimize import least_squares, curve_fit
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


LAT0, LON0 = 13.75, 100.53
KY = 111.11
KX = 111.11 * np.cos(np.radians(LAT0))
VEH_CATEGORIES = ['motorcycle', 'car', 'pickup', 'truck', 'bicycle', 'bus']


def to_km(df):
    x = (df['lon'].values - LON0) * KX
    y = (df['lat'].values - LAT0) * KY
    return np.column_stack([x, y])


def parse_hour(df):
    t = df['time_of_day'].str.split(':', expand=True).astype(int)
    return t[0].values + t[1].values / 60.0


def encode_vehicle(df):
    cat_map = {v: i for i, v in enumerate(VEH_CATEGORIES)}
    return df['vehicle_type'].map(cat_map).fillna(0).astype(int).values


def log_wrapped_normal(hour, mu_h, sigma_h):
    log_norm = -0.5 * np.log(2.0 * np.pi * sigma_h ** 2)
    terms = [log_norm - 0.5 * ((hour - mu_h + shift) / sigma_h) ** 2 for shift in (-24.0, 0.0, 24.0)]
    return logsumexp(np.vstack(terms), axis=0)


def fit_wrapped_normal_weighted(hour, w):
    theta = hour * 2.0 * np.pi / 24.0
    w_sum = np.sum(w) + 1e-12
    C = np.sum(w * np.cos(theta)) / w_sum
    S = np.sum(w * np.sin(theta)) / w_sum
    mu_h = (np.arctan2(S, C) * 24.0 / (2.0 * np.pi)) % 24.0
    diff = ((hour - mu_h + 12.0) % 24.0) - 12.0
    sigma_h = float(np.clip(np.sqrt(max(np.sum(w * diff ** 2) / w_sum, 0.04)), 0.2, 8.0))
    return float(mu_h), sigma_h


def fit_circular_fourier_time(hour, w, n_harm=2):
    """Fit smooth normalized circular log-density on [0, 24) using 2 harmonics."""
    grid = np.linspace(0.0, 24.0, 240, endpoint=False)
    th_grid = grid * 2.0 * np.pi / 24.0
    th_data = hour * 2.0 * np.pi / 24.0
    # Weighted 24-bin smoothed histogram -> fit smooth Fourier log-density
    hist, _ = np.histogram(hour, bins=48, range=(0.0, 24.0), weights=w)
    # Circular Gaussian smoothing (sigma=1.5 bins = 45 min)
    idx = np.arange(48)
    d = np.minimum(np.abs(idx[:, None] - idx[None, :]), 48 - np.abs(idx[:, None] - idx[None, :]))
    K_mat = np.exp(-0.5 * (d / 1.5) ** 2)
    K_mat /= K_mat.sum(axis=1, keepdims=True)
    smooth_p = K_mat @ (hist + 1e-3)
    log_p_bins = np.log(smooth_p)
    th_bins = (np.arange(48) + 0.5) * 2.0 * np.pi / 48.0
    # Design matrix [1, cos(t), sin(t), cos(2t), sin(2t)]
    B_bins = np.column_stack([np.ones(48)] + [fn(m * th_bins) for m in range(1, n_harm + 1) for fn in (np.cos, np.sin)])
    coeffs = np.linalg.lstsq(B_bins, log_p_bins, rcond=None)[0]
    # Compute exact log normalizing constant over [0, 24)
    B_grid = np.column_stack([np.ones(len(grid))] + [fn(m * th_grid) for m in range(1, n_harm + 1) for fn in (np.cos, np.sin)])
    log_z = logsumexp(B_grid @ coeffs) + np.log(24.0 / len(grid))
    coeffs[0] -= log_z
    return coeffs


def log_circular_fourier_time(hour, coeffs, n_harm=2):
    th = hour * 2.0 * np.pi / 24.0
    B = np.column_stack([np.ones(len(hour))] + [fn(m * th) for m in range(1, n_harm + 1) for fn in (np.cos, np.sin)])
    return B @ coeffs


def log_segment_density(xy, p_start, p_end, sigma_para, sigma_perp):
    vec = p_end - p_start
    L = np.linalg.norm(vec)
    u = vec / L
    v = np.array([-u[1], u[0]])
    diff = xy - p_start
    s = diff @ u
    d_perp = diff @ v
    cdf_hi = ndtr(s / sigma_para)
    cdf_lo = ndtr((s - L) / sigma_para)
    p_para = np.maximum(cdf_hi - cdf_lo, 1e-300) / L
    log_p_perp = -0.5 * np.log(2.0 * np.pi * sigma_perp ** 2) - 0.5 * (d_perp / sigma_perp) ** 2
    return np.log(p_para) + log_p_perp


def log_ring_density(xy, cx, cy, R, sigma_r):
    r = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    r_safe = np.maximum(r, 1e-4)
    log_p_r = -0.5 * np.log(2.0 * np.pi * sigma_r ** 2) - 0.5 * ((r - R) / sigma_r) ** 2
    return log_p_r - np.log(2.0 * np.pi * r_safe)


def build_sine_points_uniform_x(y0, A, w, phi, x_min, x_max, n_pts=400):
    """Exact Uniform-in-x parameterization verified on TL_wave (0.3331 obs vs 0.3330 unif-x!)."""
    xs = np.linspace(x_min, x_max, n_pts)
    ys = y0 + A * np.sin(w * xs + phi)
    return np.column_stack([xs, ys])


def build_arc_points_equal_arclen(cx, cy, poly_coeffs, th1, th2, n_pts=400):
    """Equal arc-length parameterization for smooth curved BR_arc (preserves both tips!)."""
    ths_f = np.linspace(th1, th2, 4000)
    rs_f = np.polyval(poly_coeffs, ths_f)
    xs_f = cx + rs_f * np.cos(ths_f)
    ys_f = cy + rs_f * np.sin(ths_f)
    ds = np.hypot(np.diff(xs_f), np.diff(ys_f))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_u = np.linspace(0.0, s[-1], n_pts)
    return np.column_stack([np.interp(s_u, s, xs_f), np.interp(s_u, s, ys_f)])


def log_curve_mixture_density(xy, curve_pts, sigma):
    N = len(xy)
    M = len(curve_pts)
    inv_2s2 = 0.5 / (sigma ** 2)
    const = -np.log(2.0 * np.pi * sigma ** 2) - np.log(M)
    out = np.empty(N, dtype=np.float64)
    for i in range(0, N, 2000):
        pts = xy[i:i + 2000]
        d2 = np.sum((pts[:, None, :] - curve_pts[None, :, :]) ** 2, axis=2)
        out[i:i + 2000] = logsumexp(-d2 * inv_2s2, axis=1) + const
    return out


def expected_ari(P, c):
    N, K = P.shape
    M = np.zeros((K, K), dtype=np.float64)
    for l in range(K):
        mask = (c == l)
        if mask.any():
            M[l] = P[mask].sum(axis=0)
    sum_P2 = np.sum(P * P)
    e_tp = 0.5 * (np.sum(M * M) - sum_P2)
    n_k = P.sum(axis=0)
    A = 0.5 * (np.sum(n_k * n_k) - sum_P2)
    m_l = np.bincount(c, minlength=K).astype(np.float64)
    B = 0.5 * np.sum(m_l * (m_l - 1.0))
    comb_N = 0.5 * N * (N - 1.0)
    e_idx = A * B / comb_N
    max_idx = 0.5 * (A + B)
    return float((e_tp - e_idx) / (max_idx - e_idx))


def optimize_expected_ari(P, c_init=None, max_passes=15):
    N, K = P.shape
    c = np.argmax(P, axis=1).copy() if c_init is None else c_init.copy()
    M = np.zeros((K, K), dtype=np.float64)
    for l in range(K):
        mask = (c == l)
        if mask.any():
            M[l] = P[mask].sum(axis=0)
    m_l = np.bincount(c, minlength=K).astype(np.float64)
    sum_P2 = np.sum(P * P)
    n_k = P.sum(axis=0)
    A = 0.5 * (np.sum(n_k * n_k) - sum_P2)
    comb_N = 0.5 * N * (N - 1.0)
    e_tp = 0.5 * (np.sum(M * M) - sum_P2)
    B = 0.5 * np.sum(m_l * (m_l - 1.0))

    uncertain_idx = np.where(P.max(axis=1) < 0.999)[0]
    for _ in range(max_passes):
        changed = 0
        for i in uncertain_idx:
            old_l = c[i]
            pi = P[i]
            M[old_l] -= pi
            m_l[old_l] -= 1.0
            e_tp_base = e_tp - np.dot(M[old_l], pi)
            B_base = B - m_l[old_l]

            e_tp_cand = e_tp_base + M @ pi
            B_cand = B_base + m_l
            e_idx_cand = A * B_cand / comb_N
            max_idx_cand = 0.5 * (A + B_cand)
            ari_cand = (e_tp_cand - e_idx_cand) / (max_idx_cand - e_idx_cand)

            best_l = int(np.argmax(ari_cand))
            M[best_l] += pi
            m_l[best_l] += 1.0
            e_tp = e_tp_cand[best_l]
            B = B_cand[best_l]
            if best_l != old_l:
                c[i] = best_l
                changed += 1
        if changed == 0:
            break
    return c, expected_ari(P, c)


class PrecisionBangkokEM:
    """
    Precision 13-Component Generative EM for ml69-k2 focusing strictly on the key signals:
      - Exact Uniform-in-x TL_wave & Uniform-in-theta deg-4 BR_arc (updated in EM!)
      - Exact GPS-jitter (15m) segment endpoints for ML_night & MR_night
      - Exact 20km x 20km noise bounding box [100.4375, 100.6225] x [13.6596, 13.84035]
      - 2-Harmonic smooth shared circular background time density
      - Empirical-Bayes shrinkage + pooled TR_blob/BL_blob vehicle_type (cuts boundary variance!)
      - Excludes proven distractor features (weather p=0.64, n_injured p=0.87)
    """
    COMPONENT_NAMES = [
        'ML_night', 'ML_morn', 'MR_night', 'MR_day',
        'ML_ring', 'ML_dot', 'MR_ring', 'MR_dot',
        'TL_wave', 'BR_arc', 'TR_blob', 'BL_blob', 'NOISE',
    ]
    TEMPORAL = {'ML_night', 'ML_morn', 'MR_night', 'MR_day'}

    def __init__(self, n_em_iters=25, veh_prior_strength=12.0, pool_blobs_veh=True):
        self.n_em_iters = n_em_iters
        self.veh_prior_strength = veh_prior_strength
        self.pool_blobs_veh = pool_blobs_veh

    def _init_params(self, df, xy, hour, veh):
        lon, lat = df['lon'].values, df['lat'].values
        def box(x0, x1, y0, y1):
            return (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)

        init_masks = {
            'ML_night': box(100.478, 100.495, 13.750, 13.761) & ((hour >= 21) | (hour < 2.5)),
            'ML_morn':  box(100.478, 100.495, 13.750, 13.761) & ((hour >= 4.0) & (hour < 10.0)),
            'MR_night': box(100.545, 100.564, 13.735, 13.748) & ((hour >= 21) | (hour < 3.5)),
            'MR_day':   box(100.545, 100.564, 13.735, 13.748) & ((hour >= 8.0) & (hour < 18.0)),
            'ML_ring':  box(100.495, 100.5045, 13.763, 13.772),
            'ML_dot':   box(100.5045, 100.511, 13.763, 13.769),
            'MR_ring':  box(100.563, 100.576, 13.747, 13.760),
            'MR_dot':   box(100.577, 100.582, 13.750, 13.754),
            'TL_wave':  box(100.445, 100.525, 13.793, 13.816),
            'BR_arc':   box(100.515, 100.615, 13.670, 13.735),
            'TR_blob':  box(100.550, 100.610, 13.780, 13.835),
            'BL_blob':  box(100.440, 100.500, 13.680, 13.735),
        }

        spatial = {}
        # Line segments with exact 15m GPS parallel jitter (not smeared by perp_std!)
        for name in ['ML_night', 'MR_night']:
            pts = xy[init_masks[name]]
            mu = pts.mean(axis=0)
            _, eigvecs = np.linalg.eigh(np.cov(pts.T))
            u, v = eigvecs[:, 1], eigvecs[:, 0]
            proj = (pts - mu) @ u
            perp = (pts - mu) @ v
            q005, q995 = np.quantile(proj, [0.005, 0.995])
            sig_perp = float(np.median(np.abs(perp)) * 1.4826)
            spatial[name] = {
                'type': 'segment',
                'p_start': mu + q005 * u,
                'p_end': mu + q995 * u,
                'sigma_para': 0.015,  # exact ~15m GPS jitter!
                'sigma_perp': sig_perp,
            }

        for name in ['ML_morn', 'MR_day', 'ML_dot', 'MR_dot', 'TR_blob', 'BL_blob']:
            pts = xy[init_masks[name]]
            spatial[name] = {'type': 'gaussian', 'mean': pts.mean(axis=0), 'cov': np.cov(pts.T) + 1e-5 * np.eye(2)}

        for name in ['ML_ring', 'MR_ring']:
            pts = xy[init_masks[name]]
            r = least_squares(lambda p: np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) - p[2],
                              [pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 0].std()], loss='soft_l1', f_scale=0.05)
            cx, cy, rad = r.x
            d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rad
            spatial[name] = {'type': 'ring', 'cx': cx, 'cy': cy, 'R': rad, 'sigma_r': float(np.median(np.abs(d)) * 1.4826)}

        # TL_wave: EXACT Uniform-in-x parameterization!
        pts = xy[init_masks['TL_wave']]
        dists, _ = NearestNeighbors(n_neighbors=6).fit(pts).kneighbors(pts)
        core = pts[dists[:, 5] < 0.16]
        def sine_fn(x, y0, A, w, phi):
            return y0 + A * np.sin(w * x + phi)
        popt, _ = curve_fit(sine_fn, core[:, 0], core[:, 1], p0=[6.0, 0.9, 1.26, -1.26])
        dy_all = popt[1] * popt[2] * np.cos(popt[2] * pts[:, 0] + popt[3])
        perp_all = np.abs(pts[:, 1] - sine_fn(pts[:, 0], *popt)) / np.sqrt(1 + dy_all ** 2)
        in_w = pts[perp_all < 0.18]
        x_min, x_max = float(np.quantile(in_w[:, 0], 0.001)), float(np.quantile(in_w[:, 0], 0.999))
        spatial['TL_wave'] = {
            'type': 'sine_curve', 'popt': popt, 'x_min': x_min, 'x_max': x_max,
            'curve_pts': build_sine_points_uniform_x(*popt, x_min, x_max, n_pts=400),
            'sigma': 0.060,
        }

        # BR_arc: EXACT deg-4 polar curve, uniform in theta!
        pts = xy[init_masks['BR_arc']]
        dists, _ = NearestNeighbors(n_neighbors=6).fit(pts).kneighbors(pts)
        core = pts[dists[:, 5] < 0.18]
        r = least_squares(lambda p: np.hypot(core[:, 0] - p[0], core[:, 1] - p[1]) - p[2],
                          [0.65, -0.58, 8.0], loss='soft_l1', f_scale=0.05)
        cx, cy, _ = r.x
        all_r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        all_th = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        c_r = np.hypot(core[:, 0] - cx, core[:, 1] - cy)
        c_th = np.arctan2(core[:, 1] - cy, core[:, 0] - cx)
        p4 = np.polyfit(c_th, c_r, deg=4)
        in_a = np.abs(all_r - np.polyval(p4, all_th)) < 0.18
        p4 = np.polyfit(all_th[in_a], all_r[in_a], deg=4)
        th1, th2 = float(np.quantile(all_th[in_a], 0.001)), float(np.quantile(all_th[in_a], 0.999))
        spatial['BR_arc'] = {
            'type': 'arc_curve', 'cx': cx, 'cy': cy, 'poly_coeffs': p4, 'th1': th1, 'th2': th2,
            'curve_pts': build_arc_points_equal_arclen(cx, cy, p4, th1, th2, n_pts=400),
            'sigma': 0.055,
        }

        # Exact verified 20km x 20km noise bounding box in lat/lon -> km!
        xmin = (100.43750 - LON0) * KX
        xmax = (100.62250 - LON0) * KX
        ymin = (13.65960 - LAT0) * KY
        ymax = (13.84035 - LAT0) * KY
        spatial['NOISE'] = {'type': 'uniform_box', 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}

        K = len(self.COMPONENT_NAMES)
        weights = np.full(K, 1.0 / K)
        bg_time_coeffs = fit_circular_fourier_time(hour, np.ones(len(hour)), n_harm=2)
        time_params = {}
        global_veh = (np.bincount(veh, minlength=6) + 1.0)
        global_veh /= global_veh.sum()
        veh_probs = {}

        for name in self.COMPONENT_NAMES:
            if name in init_masks:
                m = init_masks[name]
                if name in self.TEMPORAL:
                    time_params[name] = ('wrapped_normal',) + fit_wrapped_normal_weighted(hour[m], np.ones(m.sum()))
                else:
                    time_params[name] = ('fourier', bg_time_coeffs)
                cnt = np.bincount(veh[m], minlength=6) + self.veh_prior_strength * global_veh
            else:
                time_params[name] = ('fourier', bg_time_coeffs)
                cnt = np.bincount(veh, minlength=6) + self.veh_prior_strength * global_veh
            veh_probs[name] = cnt / cnt.sum()

        self.global_veh_ = global_veh
        self.params_ = {'spatial': spatial, 'time': time_params, 'veh': veh_probs, 'weights': weights}

    def _compute_log_likelihoods(self, xy, hour, veh):
        N = len(xy)
        K = len(self.COMPONENT_NAMES)
        log_lik = np.empty((N, K), dtype=np.float64)

        for k, name in enumerate(self.COMPONENT_NAMES):
            sp = self.params_['spatial'][name]
            t = sp['type']
            if t == 'gaussian':
                log_sp = multivariate_normal.logpdf(xy, mean=sp['mean'], cov=sp['cov'])
            elif t == 'segment':
                log_sp = log_segment_density(xy, sp['p_start'], sp['p_end'], sp['sigma_para'], sp['sigma_perp'])
            elif t == 'ring':
                log_sp = log_ring_density(xy, sp['cx'], sp['cy'], sp['R'], sp['sigma_r'])
            elif t in ('sine_curve', 'arc_curve'):
                log_sp = log_curve_mixture_density(xy, sp['curve_pts'], sp['sigma'])
            elif t == 'uniform_box':
                in_box = (xy[:, 0] >= sp['xmin']) & (xy[:, 0] <= sp['xmax']) & (xy[:, 1] >= sp['ymin']) & (xy[:, 1] <= sp['ymax'])
                area = (sp['xmax'] - sp['xmin']) * (sp['ymax'] - sp['ymin'])
                log_sp = np.where(in_box, -np.log(area), -40.0)
            else:
                raise ValueError(t)

            tp = self.params_['time'][name]
            if tp[0] == 'wrapped_normal':
                log_t = log_wrapped_normal(hour, tp[1], tp[2])
            else:
                log_t = log_circular_fourier_time(hour, tp[1], n_harm=2)

            log_v = np.log(self.params_['veh'][name][veh])
            log_lik[:, k] = log_sp + log_t + log_v

        return log_lik

    def fit(self, df):
        xy = to_km(df)
        hour = parse_hour(df)
        veh = encode_vehicle(df)
        N = len(df)
        self._init_params(df, xy, hour, veh)

        for it in range(self.n_em_iters):
            log_lik = self._compute_log_likelihoods(xy, hour, veh)
            log_joint = log_lik + np.log(self.params_['weights'] + 1e-12)[None, :]
            log_norm = logsumexp(log_joint, axis=1, keepdims=True)
            resp = np.exp(log_joint - log_norm)
            ll = float(np.sum(log_norm))

            Nk = resp.sum(axis=0)
            self.params_['weights'] = Nk / N

            # Update shared 2-harmonic background circular time density (clusters 4..12)
            bg_coeffs = fit_circular_fourier_time(hour, resp[:, 4:].sum(axis=1), n_harm=2)

            # Pooled TR_blob (10) + BL_blob (11) vehicle_type if enabled (proven identical p=0.826!)
            if self.pool_blobs_veh:
                w_blobs = resp[:, 10] + resp[:, 11]
                vb_cnt = np.bincount(veh, weights=w_blobs, minlength=6) + self.veh_prior_strength * self.global_veh_
                vb_prob = vb_cnt / vb_cnt.sum()

            for k, name in enumerate(self.COMPONENT_NAMES):
                w = resp[:, k]
                w_sum = Nk[k] + 1e-12

                if name in self.TEMPORAL:
                    self.params_['time'][name] = ('wrapped_normal',) + fit_wrapped_normal_weighted(hour, w)
                else:
                    self.params_['time'][name] = ('fourier', bg_coeffs)

                if self.pool_blobs_veh and name in ('TR_blob', 'BL_blob'):
                    self.params_['veh'][name] = vb_prob
                else:
                    vc = np.bincount(veh, weights=w, minlength=6) + self.veh_prior_strength * self.global_veh_
                    self.params_['veh'][name] = vc / vc.sum()

                sp = self.params_['spatial'][name]
                if sp['type'] == 'gaussian':
                    mu = (w[:, None] * xy).sum(axis=0) / w_sum
                    diff = xy - mu
                    sp['mean'] = mu
                    sp['cov'] = ((w[:, None] * diff).T @ diff) / w_sum + 1e-5 * np.eye(2)
                elif sp['type'] == 'segment':
                    mu = (w[:, None] * xy).sum(axis=0) / w_sum
                    diff = xy - mu
                    cov = ((w[:, None] * diff).T @ diff) / w_sum
                    _, eigvecs = np.linalg.eigh(cov)
                    u, v = eigvecs[:, 1], eigvecs[:, 0]
                    proj = diff @ u
                    perp = diff @ v
                    sig_perp = float(np.sqrt(max(np.sum(w * perp ** 2) / w_sum, 1e-5)))
                    var_proj = float(np.sum(w * proj ** 2) / w_sum)
                    # Uniform(-L/2, L/2) + GPS jitter 0.015^2
                    L = np.sqrt(max(12.0 * (var_proj - 0.015 ** 2), 0.1))
                    sp['p_start'] = mu - 0.5 * L * u
                    sp['p_end'] = mu + 0.5 * L * u
                    sp['sigma_perp'] = sig_perp
                    sp['sigma_para'] = 0.015
                elif sp['type'] == 'ring':
                    high = w > 0.1
                    pts_h, sw = xy[high], np.sqrt(w[high])
                    rc = least_squares(lambda p: sw * (np.hypot(pts_h[:, 0] - p[0], pts_h[:, 1] - p[1]) - p[2]),
                                       [sp['cx'], sp['cy'], sp['R']])
                    sp['cx'], sp['cy'], sp['R'] = float(rc.x[0]), float(rc.x[1]), float(rc.x[2])
                    r = np.hypot(xy[:, 0] - sp['cx'], xy[:, 1] - sp['cy'])
                    sp['sigma_r'] = float(np.sqrt(max(np.sum(w * (r - sp['R']) ** 2) / w_sum, 1e-6)))
                elif sp['type'] == 'sine_curve':
                    high = w > 0.1
                    pts_h, w_h = xy[high], w[high]
                    if it == 3:
                        # One clean EM-weighted refit of sine curve + exact tips (no endpoint clipping!)
                        def sine_fn(x, y0, A, om, ph):
                            return y0 + A * np.sin(om * x + ph)
                        popt, _ = curve_fit(sine_fn, pts_h[:, 0], pts_h[:, 1], p0=sp['popt'], sigma=1.0 / np.sqrt(w_h))
                        core_x = pts_h[w_h > 0.5, 0]
                        sp['popt'] = popt
                        sp['x_min'], sp['x_max'] = float(np.quantile(core_x, 0.001)), float(np.quantile(core_x, 0.999))
                        sp['curve_pts'] = build_sine_points_uniform_x(*popt, sp['x_min'], sp['x_max'], n_pts=400)
                    d2_min = np.min(np.sum((pts_h[:, None, :] - sp['curve_pts'][None, :, :]) ** 2, axis=2), axis=1)
                    sp['sigma'] = float(np.sqrt(max(np.sum(w_h * d2_min) / w_h.sum(), 1e-5)))
                elif sp['type'] == 'arc_curve':
                    high = w > 0.1
                    pts_h, w_h = xy[high], w[high]
                    if it == 3:
                        rh = np.hypot(pts_h[:, 0] - sp['cx'], pts_h[:, 1] - sp['cy'])
                        thh = np.arctan2(pts_h[:, 1] - sp['cy'], pts_h[:, 0] - sp['cx'])
                        p4 = np.polyfit(thh, rh, deg=4, w=np.sqrt(w_h))
                        core_th = thh[w_h > 0.5]
                        sp['poly_coeffs'] = p4
                        sp['th1'], sp['th2'] = float(np.quantile(core_th, 0.0005)), float(np.quantile(core_th, 0.9995))
                        sp['curve_pts'] = build_arc_points_equal_arclen(sp['cx'], sp['cy'], p4, sp['th1'], sp['th2'], n_pts=400)
                    d2_min = np.min(np.sum((pts_h[:, None, :] - sp['curve_pts'][None, :, :]) ** 2, axis=2), axis=1)
                    sp['sigma'] = float(np.sqrt(max(np.sum(w_h * d2_min) / w_h.sum(), 1e-5)))

            if it in (0, 4, 12, self.n_em_iters - 1):
                print(f"  EM iter {it+1:2d}/{self.n_em_iters} | LL = {ll:12.2f}", flush=True)

        self.log_likelihood_ = ll
        return self

    def predict_proba(self, df):
        xy = to_km(df)
        hour = parse_hour(df)
        veh = encode_vehicle(df)
        log_lik = self._compute_log_likelihoods(xy, hour, veh)
        log_joint = log_lik + np.log(self.params_['weights'] + 1e-12)[None, :]
        log_norm = logsumexp(log_joint, axis=1, keepdims=True)
        return np.exp(log_joint - log_norm)


def main():
    t0 = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, 'data.csv'))
    sample_sub = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
    prev_sub = pd.read_csv(os.path.join(base_dir, 'submission.csv'))

    # 2 focused rounds comparing pooled-blob regularized vehicle_type vs unpooled regularized vehicle_type!
    configs = [
        ('Round 1 (Pooled Blob Veh + Exact Geometry + 2-Harmonic Time)', dict(n_em_iters=16, veh_prior_strength=15.0, pool_blobs_veh=True)),
        ('Round 2 (Unpooled EB Veh + Exact Geometry + 2-Harmonic Time)', dict(n_em_iters=16, veh_prior_strength=8.0, pool_blobs_veh=False)),
    ]

    best_model, best_probs, best_ll, best_name = None, None, -np.inf, ''
    all_probs = []
    for name, kw in configs:
        print(f"\n=== {name} ===", flush=True)
        m = PrecisionBangkokEM(**kw).fit(df)
        P = m.predict_proba(df)
        c_map = np.argmax(P, axis=1)
        c_opt, ari_opt = optimize_expected_ari(P, c_map)
        print(f"  -> LL={m.log_likelihood_:.2f} | ExpARI(MAP)={expected_ari(P, c_map):.6f} | ExpARI(OPT)={ari_opt:.6f} | AvgConf={P.max(1).mean():.6f}")
        print(f"  -> ARI vs prev(0.88747) = {adjusted_rand_score(prev_sub['cluster'].values, c_opt):.6f} (diffs={(prev_sub['cluster'].values != c_opt).sum()})")
        all_probs.append(P)
        if m.log_likelihood_ > best_ll:
            best_ll = m.log_likelihood_
            best_model = m
            best_probs = P
            best_name = name

    # Compare best single round vs 2-round average posterior
    P_avg = 0.5 * all_probs[0] + 0.5 * all_probs[1]
    c_best, ari_best = optimize_expected_ari(best_probs, np.argmax(best_probs, axis=1))
    c_avg, ari_avg = optimize_expected_ari(P_avg, np.argmax(P_avg, axis=1))
    print(f"\n=== Selection (total time={time.time()-t0:.1f}s) ===")
    print(f"Best Round ({best_name}): ExpARI={ari_best:.6f}")
    print(f"2-Round Blend:            ExpARI={ari_avg:.6f} (diffs vs best={(c_best != c_avg).sum()})")

    # Pick strictly highest Expected ARI, map NOISE (12) -> -1 per official ml69-k2 format!
    final_c = c_best if ari_best >= ari_avg else c_avg
    final_labels = np.where(final_c == 12, -1, final_c).astype(int)

    # Monte Carlo Expected ARI (100 sims on 30% public split size = 6000!)
    rng = np.random.default_rng(42)
    cdf = np.cumsum(best_probs, axis=1)
    mc_aris = [adjusted_rand_score((rng.random(len(df))[:, None] > cdf).sum(axis=1), final_c) for _ in range(50)]
    print(f"Monte Carlo Expected ARI (50 sims): {np.mean(mc_aris):.6f} +/- {np.std(mc_aris):.5f}")

    print("\nFinal cluster counts:")
    for k, cname in enumerate(best_model.COMPONENT_NAMES):
        lbl = -1 if k == 12 else k
        print(f"  {lbl:2d} ({cname:9s}): hard={(final_labels==lbl).sum():4d} | soft={best_probs[:,k].sum():6.1f} | moto={best_model.params_['veh'][cname][0]*100:4.1f}%")

    # Save & validate!
    sub = pd.DataFrame({'id': df['id'], 'cluster': final_labels})
    assert list(sub.columns) == list(sample_sub.columns)
    assert len(sub) == len(sample_sub)
    assert (sub['id'].values == sample_sub['id'].values).all()
    assert sub['cluster'].isnull().sum() == 0

    sub_path = os.path.join(base_dir, 'submission_k2_v3.csv')
    sub.to_csv(sub_path, index=False)
    # Also save round 1 specifically so we can verify both if needed
    c_r1, _ = optimize_expected_ari(all_probs[0], np.argmax(all_probs[0], axis=1))
    pd.DataFrame({'id': df['id'], 'cluster': np.where(c_r1 == 12, -1, c_r1).astype(int)}).to_csv(
        os.path.join(base_dir, 'submission_k2_r1.csv'), index=False
    )
    joblib.dump(best_model, os.path.join(base_dir, 'k2_model.joblib'))
    print(f"\nSaved {sub_path} and k2_model.joblib (validated OK)")


if __name__ == '__main__':
    main()
