import os
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
    """Exact wrapped normal log-density on [0, 24) (sum over k in {-1, 0, 1})."""
    terms = []
    log_norm = -0.5 * np.log(2.0 * np.pi * sigma_h**2)
    for shift in (-24.0, 0.0, 24.0):
        diff = hour - mu_h + shift
        terms.append(log_norm - 0.5 * (diff / sigma_h)**2)
    return logsumexp(np.vstack(terms), axis=0)


def fit_wrapped_normal_weighted(hour, w):
    theta = hour * 2.0 * np.pi / 24.0
    w_sum = np.sum(w) + 1e-12
    C = np.sum(w * np.cos(theta)) / w_sum
    S = np.sum(w * np.sin(theta)) / w_sum
    mu_h = (np.arctan2(S, C) * 24.0 / (2.0 * np.pi)) % 24.0
    # Circular signed distance in [-12, 12)
    diff = ((hour - mu_h + 12.0) % 24.0) - 12.0
    var_h = float(np.sum(w * diff**2) / w_sum)
    sigma_h = float(np.clip(np.sqrt(max(var_h, 0.04)), 0.2, 8.0))
    return float(mu_h), sigma_h


def log_von_mises(theta, mu, kappa):
    return kappa * np.cos(theta - mu) - np.log(2.0 * np.pi) - np.log(i0(kappa))


def fit_von_mises_weighted(theta, w):
    w_sum = np.sum(w) + 1e-12
    C = np.sum(w * np.cos(theta)) / w_sum
    S = np.sum(w * np.sin(theta)) / w_sum
    mu = np.arctan2(S, C)
    R = np.clip(np.hypot(C, S), 1e-6, 0.9995)
    kappa = R * (2.0 - R**2) / (1.0 - R**2)
    for _ in range(3):
        A = i1(kappa) / i0(kappa)
        dA = 1.0 - A**2 - A / max(kappa, 1e-6)
        kappa = max(1e-4, kappa - (A - R) / max(dA, 1e-6))
    return float(mu), float(min(kappa, 100.0))


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
    log_p_perp = -0.5 * np.log(2.0 * np.pi * sigma_perp**2) - 0.5 * (d_perp / sigma_perp)**2
    return np.log(p_para) + log_p_perp


def log_ring_density(xy, cx, cy, R, sigma_r):
    r = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
    r_safe = np.maximum(r, 1e-4)
    log_p_r = -0.5 * np.log(2.0 * np.pi * sigma_r**2) - 0.5 * ((r - R) / sigma_r)**2
    return log_p_r - np.log(2.0 * np.pi * r_safe)


def build_sine_curve_points(y0, A, w, phi, x_min, x_max, n_pts=400):
    xs_fine = np.linspace(x_min, x_max, 4000)
    ys_fine = y0 + A * np.sin(w * xs_fine + phi)
    ds = np.hypot(np.diff(xs_fine), np.diff(ys_fine))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_uniform = np.linspace(0.0, s[-1], n_pts)
    xs = np.interp(s_uniform, s, xs_fine)
    ys = np.interp(s_uniform, s, ys_fine)
    return np.column_stack([xs, ys])


def build_arc_curve_points(cx, cy, poly_coeffs, th1, th2, n_pts=400):
    ths_fine = np.linspace(th1, th2, 4000)
    rs_fine = np.polyval(poly_coeffs, ths_fine)
    xs_fine = cx + rs_fine * np.cos(ths_fine)
    ys_fine = cy + rs_fine * np.sin(ths_fine)
    ds = np.hypot(np.diff(xs_fine), np.diff(ys_fine))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_uniform = np.linspace(0.0, s[-1], n_pts)
    xs = np.interp(s_uniform, s, xs_fine)
    ys = np.interp(s_uniform, s, ys_fine)
    return np.column_stack([xs, ys])


def log_curve_mixture_density(xy, curve_pts, sigma):
    N = len(xy)
    M = len(curve_pts)
    inv_2s2 = 0.5 / (sigma**2)
    const = -np.log(2.0 * np.pi * sigma**2) - np.log(M)
    out = np.empty(N, dtype=np.float64)
    chunk = 2000
    for i in range(0, N, chunk):
        pts = xy[i:i + chunk]
        d2 = np.sum((pts[:, None, :] - curve_pts[None, :, :])**2, axis=2)
        out[i:i + chunk] = logsumexp(-d2 * inv_2s2, axis=1) + const
    return out


def expected_ari(P, c):
    """Exact analytical E_{z ~ P}[ARI(z, c)] under posterior matrix P (N x K)."""
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


def optimize_expected_ari(P, c_init=None, max_passes=20):
    """
    Exact coordinate ascent maximizing E_{z ~ P}[ARI(z, c)].
    Because ARI is quadratic in cluster sizes, the Bayes-optimal decision rule
    for ARI differs from MAP (argmax) on borderline points between clusters of
    different sizes (e.g. uniform NOISE ~3000 pts vs blobs ~1500 pts).
    """
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


class BangkokAccidentMixtureModel:
    """
    13-component generative mixture model (12 structured clusters + 1 uniform noise)
    fitted via Expectation-Maximization on (lat, lon, time_of_day, vehicle_type),
    paired with Exact Expected-ARI Bayes-optimal decision assignment.
    """
    COMPONENT_NAMES = [
        'ML_night',  # 0: line segment + midnight
        'ML_morn',   # 1: 2D Gaussian + morning
        'MR_night',  # 2: line segment + midnight + high moto
        'MR_day',    # 3: 2D Gaussian + midday
        'ML_ring',   # 4: hollow circle
        'ML_dot',    # 5: 2D Gaussian
        'MR_ring',   # 6: hollow circle
        'MR_dot',    # 7: 2D Gaussian
        'TL_wave',   # 8: sine wave curve
        'BR_arc',    # 9: smooth curved arc
        'TR_blob',   # 10: large 2D Gaussian
        'BL_blob',   # 11: large 2D Gaussian
        'NOISE',     # 12: uniform 20km x 20km background
    ]
    TEMPORAL_CLUSTERS = {'ML_night', 'ML_morn', 'MR_night', 'MR_day'}

    def __init__(self, n_em_iters=30, use_ari_opt=True):
        self.n_em_iters = n_em_iters
        self.use_ari_opt = use_ari_opt
        self.params_ = {}

    def _init_params(self, df, xy, hour, theta, veh):
        N = len(df)
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

        # Line segments: ML_night, MR_night
        for name in ['ML_night', 'MR_night']:
            pts = xy[init_masks[name]]
            mu = pts.mean(axis=0)
            _, eigvecs = np.linalg.eigh(np.cov(pts.T))
            u, v = eigvecs[:, 1], eigvecs[:, 0]
            proj = (pts - mu) @ u
            perp = (pts - mu) @ v
            q02, q98 = np.quantile(proj, [0.02, 0.98])
            sig_perp = float(np.median(np.abs(perp)) * 1.4826)
            spatial[name] = {
                'type': 'segment',
                'p_start': mu + q02 * u,
                'p_end': mu + q98 * u,
                'sigma_para': sig_perp,
                'sigma_perp': sig_perp,
            }

        # 2D Gaussians
        for name in ['ML_morn', 'MR_day', 'ML_dot', 'MR_dot', 'TR_blob', 'BL_blob']:
            pts = xy[init_masks[name]]
            spatial[name] = {
                'type': 'gaussian',
                'mean': pts.mean(axis=0),
                'cov': np.cov(pts.T) + 1e-5 * np.eye(2),
            }

        # Rings
        for name in ['ML_ring', 'MR_ring']:
            pts = xy[init_masks[name]]
            r = least_squares(
                lambda p: np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) - p[2],
                [pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 0].std()],
                loss='soft_l1', f_scale=0.05
            )
            cx, cy, rad = r.x
            d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rad
            sig_r = float(np.median(np.abs(d)) * 1.4826)
            spatial[name] = {'type': 'ring', 'cx': cx, 'cy': cy, 'R': rad, 'sigma_r': sig_r}

        # Sine wave: TL_wave
        pts = xy[init_masks['TL_wave']]
        dists, _ = NearestNeighbors(n_neighbors=6).fit(pts).kneighbors(pts)
        core = pts[dists[:, 5] < 0.16]
        def sine_fn(x, y0, A, w, phi):
            return y0 + A * np.sin(w * x + phi)
        popt, _ = curve_fit(sine_fn, core[:, 0], core[:, 1], p0=[6.0, 0.9, 1.26, -1.26])
        dy_dx = popt[1] * popt[2] * np.cos(popt[2] * core[:, 0] + popt[3])
        perp = (core[:, 1] - sine_fn(core[:, 0], *popt)) / np.sqrt(1 + dy_dx**2)
        sig_wave = float(np.std(perp))
        dy_all = popt[1] * popt[2] * np.cos(popt[2] * pts[:, 0] + popt[3])
        perp_all = np.abs(pts[:, 1] - sine_fn(pts[:, 0], *popt)) / np.sqrt(1 + dy_all**2)
        in_wave = pts[perp_all < 3.0 * sig_wave]
        x_min, x_max = float(np.quantile(in_wave[:, 0], 0.001)), float(np.quantile(in_wave[:, 0], 0.999))
        curve_pts = build_sine_curve_points(*popt, x_min, x_max, n_pts=400)
        spatial['TL_wave'] = {
            'type': 'curve',
            'popt': popt, 'x_min': x_min, 'x_max': x_max,
            'curve_pts': curve_pts, 'sigma': sig_wave,
        }

        # Smooth curved arc: BR_arc
        pts = xy[init_masks['BR_arc']]
        dists, _ = NearestNeighbors(n_neighbors=6).fit(pts).kneighbors(pts)
        core = pts[dists[:, 5] < 0.18]
        r = least_squares(
            lambda p: np.hypot(core[:, 0] - p[0], core[:, 1] - p[1]) - p[2],
            [0.65, -0.58, 8.0], loss='soft_l1', f_scale=0.05
        )
        cx, cy, _ = r.x
        core_r = np.hypot(core[:, 0] - cx, core[:, 1] - cy)
        core_th = np.arctan2(core[:, 1] - cy, core[:, 0] - cx)
        poly_coeffs = np.polyfit(core_th, core_r, deg=4)
        all_r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        all_th = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        in_arc = np.abs(all_r - np.polyval(poly_coeffs, all_th)) < 0.20
        poly_coeffs = np.polyfit(all_th[in_arc], all_r[in_arc], deg=4)
        sig_arc = float(np.std(all_r[in_arc] - np.polyval(poly_coeffs, all_th[in_arc])))
        th1, th2 = float(np.quantile(all_th[in_arc], 0.001)), float(np.quantile(all_th[in_arc], 0.999))
        arc_pts = build_arc_curve_points(cx, cy, poly_coeffs, th1, th2, n_pts=400)
        spatial['BR_arc'] = {
            'type': 'curve',
            'cx': cx, 'cy': cy, 'poly_coeffs': poly_coeffs,
            'th1': th1, 'th2': th2,
            'curve_pts': arc_pts, 'sigma': sig_arc,
        }

        # Exact 20km x 20km Uniform Noise Box around (13.75, 100.53)
        spatial['NOISE'] = {
            'type': 'uniform_box',
            'xmin': -10.005, 'xmax': 10.005,
            'ymin': -10.045, 'ymax': 10.045,
        }

        K = len(self.COMPONENT_NAMES)
        weights = np.full(K, 1.0 / K)
        time_params = {}
        veh_probs = {}
        for name in self.COMPONENT_NAMES:
            if name in init_masks:
                m = init_masks[name]
                if name in self.TEMPORAL_CLUSTERS:
                    time_params[name] = ('wrapped_normal',) + fit_wrapped_normal_weighted(hour[m], np.ones(m.sum()))
                else:
                    time_params[name] = ('von_mises',) + fit_von_mises_weighted(theta[m], np.ones(m.sum()))
                counts = np.bincount(veh[m], minlength=len(VEH_CATEGORIES)) + 1.0
            else:
                time_params[name] = ('von_mises',) + fit_von_mises_weighted(theta, np.ones(N))
                counts = np.bincount(veh, minlength=len(VEH_CATEGORIES)) + 1.0
            veh_probs[name] = counts / counts.sum()

        self.params_ = {
            'spatial': spatial,
            'time': time_params,
            'veh': veh_probs,
            'weights': weights,
        }

    def _compute_log_likelihoods(self, xy, hour, theta, veh):
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
            elif t == 'curve':
                log_sp = log_curve_mixture_density(xy, sp['curve_pts'], sp['sigma'])
            elif t == 'uniform_box':
                in_box = (
                    (xy[:, 0] >= sp['xmin']) & (xy[:, 0] <= sp['xmax']) &
                    (xy[:, 1] >= sp['ymin']) & (xy[:, 1] <= sp['ymax'])
                )
                area = (sp['xmax'] - sp['xmin']) * (sp['ymax'] - sp['ymin'])
                log_sp = np.where(in_box, -np.log(area), -35.0)
            else:
                raise ValueError(t)

            tp = self.params_['time'][name]
            if tp[0] == 'wrapped_normal':
                # Convert density from per-hour to per-radian so scale matches Von Mises!
                log_t = log_wrapped_normal(hour, tp[1], tp[2]) + np.log(24.0 / (2.0 * np.pi))
            else:
                log_t = log_von_mises(theta, tp[1], tp[2])

            log_v = np.log(self.params_['veh'][name][veh])
            log_lik[:, k] = log_sp + log_t + log_v

        return log_lik

    def fit(self, df):
        xy = to_km(df)
        hour = parse_hour(df)
        theta = hour * 2.0 * np.pi / 24.0
        veh = encode_vehicle(df)
        N = len(df)

        self._init_params(df, xy, hour, theta, veh)

        for it in range(self.n_em_iters):
            log_lik = self._compute_log_likelihoods(xy, hour, theta, veh)
            log_joint = log_lik + np.log(self.params_['weights'] + 1e-12)[None, :]
            log_norm = logsumexp(log_joint, axis=1, keepdims=True)
            resp = np.exp(log_joint - log_norm)
            ll = float(np.sum(log_norm))

            Nk = resp.sum(axis=0)
            self.params_['weights'] = Nk / N

            # Shared background time distribution across non-temporal clusters (4..12)
            w_bg_time = resp[:, 4:].sum(axis=1)
            bg_mu_t, bg_kap_t = fit_von_mises_weighted(theta, w_bg_time)

            for k, name in enumerate(self.COMPONENT_NAMES):
                w = resp[:, k]
                w_sum = Nk[k] + 1e-12

                # Update time distribution
                if name in self.TEMPORAL_CLUSTERS:
                    self.params_['time'][name] = ('wrapped_normal',) + fit_wrapped_normal_weighted(hour, w)
                else:
                    self.params_['time'][name] = ('von_mises', bg_mu_t, bg_kap_t)

                # Update vehicle_type categorical
                v_counts = np.bincount(veh, weights=w, minlength=len(VEH_CATEGORIES)) + 1.0
                self.params_['veh'][name] = v_counts / v_counts.sum()

                # Update spatial parameters
                sp = self.params_['spatial'][name]
                if sp['type'] == 'gaussian':
                    mu = (w[:, None] * xy).sum(axis=0) / w_sum
                    diff = xy - mu
                    cov = ((w[:, None] * diff).T @ diff) / w_sum + 1e-5 * np.eye(2)
                    sp['mean'] = mu
                    sp['cov'] = cov
                elif sp['type'] == 'segment':
                    mu = (w[:, None] * xy).sum(axis=0) / w_sum
                    diff = xy - mu
                    cov = ((w[:, None] * diff).T @ diff) / w_sum
                    _, eigvecs = np.linalg.eigh(cov)
                    u, v = eigvecs[:, 1], eigvecs[:, 0]
                    proj = diff @ u
                    perp = diff @ v
                    var_perp = float(np.sum(w * perp**2) / w_sum)
                    sig_perp = np.sqrt(max(var_perp, 1e-5))
                    var_proj = float(np.sum(w * proj**2) / w_sum)
                    L = np.sqrt(max(12.0 * (var_proj - var_perp), 0.05))
                    sp['p_start'] = mu - 0.5 * L * u
                    sp['p_end'] = mu + 0.5 * L * u
                    sp['sigma_perp'] = sig_perp
                    sp['sigma_para'] = sig_perp
                elif sp['type'] == 'ring':
                    # Refine ring center (cx, cy) and radius R using high-weight points
                    high = w > 0.1
                    pts_h = xy[high]
                    sw = np.sqrt(w[high])
                    res_c = least_squares(
                        lambda p: sw * (np.hypot(pts_h[:, 0] - p[0], pts_h[:, 1] - p[1]) - p[2]),
                        [sp['cx'], sp['cy'], sp['R']]
                    )
                    sp['cx'], sp['cy'], sp['R'] = float(res_c.x[0]), float(res_c.x[1]), float(res_c.x[2])
                    r = np.hypot(xy[:, 0] - sp['cx'], xy[:, 1] - sp['cy'])
                    sp['sigma_r'] = float(np.sqrt(max(np.sum(w * (r - sp['R'])**2) / w_sum, 1e-6)))
                elif sp['type'] == 'curve':
                    high = w > 0.1
                    pts_h = xy[high]
                    w_h = w[high]
                    d2_min = np.min(np.sum((pts_h[:, None, :] - sp['curve_pts'][None, :, :])**2, axis=2), axis=1)
                    sp['sigma'] = float(np.sqrt(max(np.sum(w_h * d2_min) / (np.sum(w_h) + 1e-12), 1e-5)))

            if it in (0, 4, 9, 14, 19, 24, self.n_em_iters - 1):
                print(f"EM iter {it+1:2d} | log-likelihood = {ll:12.2f}")

        return self

    def predict_proba(self, df):
        xy = to_km(df)
        hour = parse_hour(df)
        theta = hour * 2.0 * np.pi / 24.0
        veh = encode_vehicle(df)
        log_lik = self._compute_log_likelihoods(xy, hour, theta, veh)
        log_joint = log_lik + np.log(self.params_['weights'] + 1e-12)[None, :]
        log_norm = logsumexp(log_joint, axis=1, keepdims=True)
        return np.exp(log_joint - log_norm)

    def predict(self, df):
        probs = self.predict_proba(df)
        c_map = np.argmax(probs, axis=1)
        if not self.use_ari_opt:
            return c_map
        c_opt, _ = optimize_expected_ari(probs, c_map)
        return c_opt


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, 'data.csv'))
    sample_sub = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))

    model = BangkokAccidentMixtureModel(n_em_iters=30, use_ari_opt=True)
    model.fit(df)

    probs = model.predict_proba(df)
    c_map = np.argmax(probs, axis=1)
    c_opt, ari_opt = optimize_expected_ari(probs, c_map)
    ari_map = expected_ari(probs, c_map)

    print("\n=== MAP vs Expected-ARI-Optimal Assignment ===")
    print(f"MAP (argmax)   Analytical Expected ARI: {ari_map:.6f}")
    print(f"ARI-Optimal    Analytical Expected ARI: {ari_opt:.6f}  (gain = +{ari_opt - ari_map:.6f})")
    print(f"Points reassigned from MAP -> ARI-Opt : {(c_opt != c_map).sum()}")

    # Monte Carlo verification (50 draws from posterior)
    rng = np.random.default_rng(42)
    cdf = np.cumsum(probs, axis=1)
    mc_map, mc_opt, mc_non_noise = [], [], []
    for _ in range(50):
        u = rng.random(len(df))[:, None]
        z_sim = (u > cdf).sum(axis=1)
        mc_map.append(adjusted_rand_score(z_sim, c_map))
        mc_opt.append(adjusted_rand_score(z_sim, c_opt))
        m = z_sim != 12
        mc_non_noise.append(adjusted_rand_score(z_sim[m], c_opt[m]))
    print(f"MC ARI (50 sims, all 20k) MAP: {np.mean(mc_map):.6f} +/- {np.std(mc_map):.5f}")
    print(f"MC ARI (50 sims, all 20k) OPT: {np.mean(mc_opt):.6f} +/- {np.std(mc_opt):.5f}")
    print(f"MC ARI (50 sims, non-noise)  : {np.mean(mc_non_noise):.6f} +/- {np.std(mc_non_noise):.5f}")

    preds = c_opt if ari_opt >= ari_map else c_map

    print("\n=== Final Cluster Summary ===")
    print(f"{'ID':>2} | {'Name':10s} | {'MAP n':>6} | {'OPT n':>6} | {'Soft n':>8} | {'Weight':>6} | {'Moto%':>5}")
    for k, name in enumerate(model.COMPONENT_NAMES):
        soft_n = probs[:, k].sum()
        w = model.params_['weights'][k]
        moto = model.params_['veh'][name][0] * 100
        print(f"{k:2d} | {name:10s} | {(c_map == k).sum():6d} | {(preds == k).sum():6d} | {soft_n:8.1f} | {w:6.4f} | {moto:5.1f}")

    # Save model & submission
    model_path = os.path.join(base_dir, 'k2_model.joblib')
    sub_path = os.path.join(base_dir, 'submission.csv')
    joblib.dump(model, model_path)

    sub = pd.DataFrame({'id': df['id'], 'cluster': preds.astype(int)})
    assert list(sub.columns) == list(sample_sub.columns)
    assert len(sub) == len(sample_sub)
    assert (sub['id'].values == sample_sub['id'].values).all()
    assert sub['cluster'].isnull().sum() == 0
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved {model_path} and {sub_path} (validated OK, shape={sub.shape})")


if __name__ == '__main__':
    main()
