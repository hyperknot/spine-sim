from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .model import (
    SimulationResult,
    SpineModel,
    initial_state_static,
    newmark_nonlinear,
    newmark_peak_element_force,
)


@dataclass
class CalibrationCase:
    """A single calibration case with input acceleration and target force."""

    name: str
    time_s: np.ndarray
    accel_g: np.ndarray
    force_time_s: np.ndarray
    force_n: np.ndarray


@dataclass
class PeakCalibrationCase:
    """
    Peak-only calibration case.

    We calibrate to the peak of a specific element force (e.g., T12-L1).
    """

    name: str
    time_s: np.ndarray
    accel_g: np.ndarray
    target_peak_force_n: float
    settle_ms: float = 0.0


@dataclass
class CalibrationResult:
    params: dict
    success: bool
    cost: float
    residual_norm: float


def calibrate_model_peaks_joint(
    base_model: SpineModel,
    cases: list[PeakCalibrationCase],
    t12_element_index: int,
    *,
    init_params: dict,
    bounds: dict[str, tuple[float, float]],
    apply_params: Callable[[SpineModel, dict], SpineModel],
    max_nfev: int = 200,
    verbose: bool = False,
    n_starts: int = 5,
    cost_tol: float = 1e-4,
    stall_iters: int = 10,
    # Exploration (always ALL cases)
    explore_samples: int = 60,
    explore_keep: int = 30,
    explore_local_frac: float = 0.6,
    explore_local_sigma: float = 0.20,
    diversity_min_dist: float = 0.35,
    # Snapping
    snap_norm_step_explore: float = 0.01,
    snap_norm_step_refine: float = 0.001,
    # Caching bins (for model-cache only; peak/settle caches are exact-to-snap)
    cache_norm_step_model: float = 0.01,
    # Cache sizes
    max_model_cache: int = 256,
    max_settle_cache: int = 4096,
    max_peak_cache: int = 20000,
    # Simulation accuracy knobs (explore can be coarser)
    explore_max_newton_iter: int = 12,
    explore_newton_tol: float = 1e-7,
    refine_max_newton_iter: int = 25,
    refine_newton_tol: float = 1e-9,
) -> CalibrationResult:
    """
    Peak calibration with:
      - exploration (global + local perturbations) ALWAYS on all cases,
      - diversity-based seed selection,
      - least_squares refinement from each seed,
      - shared caches across ALL refinement starts.

    Important behavioral rules:
      - Bounds with lo == hi are treated as disabled and NOT optimized.
      - Snapping is applied:
          * exploration: coarse snapping (discourages micro-chasing)
          * refinement: fine snapping (still discourages micro-chasing but allows movement)
      - Stall counting only advances when the snapped evaluation key changes.
        This fixes the "refinement ends after Iter 1" issue.

    Notes on caching:
      - Model cache can be shared between explore/refine (safe).
      - Peak/settle depend on simulation tolerances; we keep separate caches for explore vs refine.
      - Refinement caches are shared across all starts (what you asked for).
    """
    if not cases:
        raise ValueError('No calibration cases provided.')

    # -------------------------
    # Key ordering
    # -------------------------
    base_keys = [
        's_k_spine',
        's_c_spine',
        'buttocks_k_n_per_m',
        'buttocks_c_ns_per_m',
        'buttocks_limit_mm',
    ]

    def _key_sort(k: str) -> tuple:
        if k in base_keys:
            return (0, base_keys.index(k), k)
        if k in ('c_base_ns_per_m', 'disc_poly_k2_n_per_m2', 'disc_poly_k3_n_per_m3'):
            return (1, k, k)
        if k.startswith('maxwell_k_ratio_'):
            return (2, int(k.split('_')[-1]), k)
        if k.startswith('maxwell_tau_ms_'):
            return (3, int(k.split('_')[-1]), k)
        return (9, k, k)

    all_keys = sorted(bounds.keys(), key=_key_sort)

    for k in all_keys:
        if k not in init_params:
            raise ValueError(f"Missing init param '{k}'.")
        if k not in bounds:
            raise ValueError(f"Missing bounds for param '{k}'.")

    # -------------------------
    # Case precomputes
    # -------------------------
    targets = np.asarray([float(c.target_peak_force_n) for c in cases], dtype=float)
    scales = np.asarray([max(abs(t), 1.0) for t in targets], dtype=float)
    case_dt = np.asarray([float(np.median(np.diff(c.time_s))) for c in cases], dtype=float)

    # -------------------------
    # Internal parameterization (log for positive-bounded)
    # -------------------------
    n_all = len(all_keys)
    lo_phys = np.zeros(n_all, dtype=float)
    hi_phys = np.zeros(n_all, dtype=float)
    x0_phys = np.zeros(n_all, dtype=float)
    use_log = np.zeros(n_all, dtype=bool)

    for i, k in enumerate(all_keys):
        lo, hi = float(bounds[k][0]), float(bounds[k][1])
        if hi < lo:
            raise ValueError(f'Invalid bounds for {k}: [{lo}, {hi}]')
        lo_phys[i] = lo
        hi_phys[i] = hi
        x0_phys[i] = float(init_params[k])
        if lo > 0.0 and hi > 0.0:
            use_log[i] = True

    # Enabled vs fixed (lo == hi disabled)
    enabled_mask = (hi_phys - lo_phys) > 0.0
    enabled_idx = np.nonzero(enabled_mask)[0]
    fixed_idx = np.nonzero(~enabled_mask)[0]

    def _internal_from_phys(x_phys_full: np.ndarray) -> np.ndarray:
        x_int = x_phys_full.copy()
        x_int[use_log] = np.log(np.clip(x_int[use_log], 1e-300, None))
        return x_int

    def _phys_from_internal(x_int_full: np.ndarray) -> np.ndarray:
        x_phys = x_int_full.copy()
        x_phys[use_log] = np.exp(x_phys[use_log])
        return x_phys

    def _params_dict_from_int_full(x_int_full: np.ndarray) -> dict:
        x_phys_full = _phys_from_internal(x_int_full)
        return {all_keys[i]: float(x_phys_full[i]) for i in range(n_all)}

    def _format_param_line(p: dict, keys: list[str]) -> str:
        parts = []
        for k in keys:
            if k in p:
                parts.append(f'{k}={p[k]:.6g}')
        return ', '.join(parts)

    def _print_case_details(details: list[tuple[str, float, float, float]]) -> None:
        for name, pred, target, r in details:
            print(
                f'    {name}: pred={pred:.1f}N, target={target:.1f}N, '
                f'err={pred - target:+.1f}N, resid={r:+.4f}'
            )

    # Full internal vectors
    x0_phys_clipped = np.clip(x0_phys, lo_phys, hi_phys)
    x0_int_full = _internal_from_phys(x0_phys_clipped)

    lo_int_full = lo_phys.copy()
    hi_int_full = hi_phys.copy()
    lo_int_full[use_log] = np.log(np.clip(lo_int_full[use_log], 1e-300, None))
    hi_int_full[use_log] = np.log(np.clip(hi_int_full[use_log], 1e-300, None))

    if enabled_idx.size == 0:
        # Nothing to optimize; evaluate once and return.
        p = {all_keys[i]: float(x0_phys_clipped[i]) for i in range(n_all)}
        model = apply_params(base_model, p)
        res = np.zeros(len(cases), dtype=float)
        details = []
        for j, case in enumerate(cases):
            y0 = np.zeros(model.size(), dtype=float)
            v0 = np.zeros(model.size(), dtype=float)
            s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

            if case.settle_ms > 0.0:
                dt = case_dt[j]
                n_settle = int(round((case.settle_ms / 1000.0) / dt)) + 1
                t_settle = dt * np.arange(n_settle)
                a_settle = np.zeros_like(t_settle)
                settle_out = newmark_peak_element_force(
                    model,
                    t_settle,
                    a_settle,
                    y0,
                    v0,
                    s0,
                    peak_element_index=None,
                    max_newton_iter=refine_max_newton_iter,
                    newton_tol=refine_newton_tol,
                )
                y0, v0, s0 = settle_out.y_final, settle_out.v_final, settle_out.s_final

            out = newmark_peak_element_force(
                model,
                case.time_s,
                case.accel_g,
                y0,
                v0,
                s0,
                peak_element_index=t12_element_index,
                max_newton_iter=refine_max_newton_iter,
                newton_tol=refine_newton_tol,
            )
            pred = float(out.peak_force_n)
            r = (pred - targets[j]) / scales[j]
            res[j] = r
            details.append((case.name, pred, targets[j], r))

        if verbose:
            print('=== Peak calibration (nothing enabled) ===')
            print(f'  Params: {_format_param_line(p, base_keys)}')
            print('  Residuals per case:')
            _print_case_details(details)
            print(f'  Cost: {float(np.sum(res**2)):.6f}')

        return CalibrationResult(
            params=p,
            success=True,
            cost=float(np.sum(res**2)),
            residual_norm=float(np.linalg.norm(res)),
        )

    # Enabled-only vectors for least_squares
    x0_int = x0_int_full[enabled_idx]
    lb = lo_int_full[enabled_idx]
    ub = hi_int_full[enabled_idx]

    # -------------------------
    # Shared caches
    # -------------------------
    settle_time_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

    # Model cache shared across explore/refine
    model_cache: OrderedDict[bytes, SpineModel] = OrderedDict()

    # Peak/settle caches separated by stage (because tolerances may differ)
    explore_peak_cache: OrderedDict[tuple[bytes, str], float] = OrderedDict()
    explore_settle_cache: OrderedDict[
        tuple[bytes, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = OrderedDict()

    refine_peak_cache: OrderedDict[tuple[bytes, str], float] = OrderedDict()
    refine_settle_cache: OrderedDict[
        tuple[bytes, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = OrderedDict()

    def _lru_get(od: OrderedDict, key):
        v = od.get(key)
        if v is not None:
            od.move_to_end(key)
        return v

    def _lru_put(od: OrderedDict, key, value, max_size: int) -> None:
        if key in od:
            od[key] = value
            od.move_to_end(key)
            return
        od[key] = value
        od.move_to_end(key)
        while len(od) > max_size:
            od.popitem(last=False)

    def _norm01(x_int_full: np.ndarray) -> np.ndarray:
        denom = hi_int_full - lo_int_full
        denom = np.where(denom == 0.0, 1.0, denom)
        z = (x_int_full - lo_int_full) / denom
        return np.clip(z, 0.0, 1.0)

    def _snap_int_full(x_int_full: np.ndarray, step: float) -> np.ndarray:
        if step <= 0.0:
            return x_int_full
        z = _norm01(x_int_full)
        zq = np.rint(z / step) * step
        zq = np.clip(zq, 0.0, 1.0)
        denom = hi_int_full - lo_int_full
        denom = np.where(denom == 0.0, 1.0, denom)
        return lo_int_full + zq * denom

    def _model_key(x_int_full_snapped: np.ndarray) -> bytes:
        # Model caching uses its own binning step (independent from snap step).
        # This is an approximation, but only for model object creation cost.
        # Peak/settle caches stay tied to the actual snap evaluation key.
        z = _norm01(x_int_full_snapped)
        step = cache_norm_step_model if cache_norm_step_model > 0.0 else 0.0
        if step > 0.0:
            q = np.rint(z / step).astype(np.int16)
            return q.tobytes()
        return z.astype(np.float32).tobytes()

    def _eval_key(x_int_full_snapped: np.ndarray) -> bytes:
        # Evaluation key is exact-to-snap (no extra binning) so we don't “reuse wrong physics”.
        return x_int_full_snapped.astype(np.float64).tobytes()

    def _get_model(x_int_full_snapped: np.ndarray) -> SpineModel:
        mkey = _model_key(x_int_full_snapped)
        model = _lru_get(model_cache, mkey)
        if model is not None:
            return model
        p = _params_dict_from_int_full(x_int_full_snapped)
        model = apply_params(base_model, p)
        _lru_put(model_cache, mkey, model, max_model_cache)
        return model

    def _evaluate_all_cases(
        x_int_full_raw: np.ndarray,
        *,
        stage: str,
        snap_step: float,
        max_newton_iter: int,
        newton_tol: float,
        want_details: bool,
    ) -> tuple[np.ndarray, float, list[tuple[str, float, float, float]], dict, bytes]:
        """
        Evaluate on all cases, with stage-specific snap and stage-specific peak/settle caches.
        Returns: (residuals, cost, details, params_dict, eval_key)
        """
        x_int_full_snapped = _snap_int_full(x_int_full_raw, snap_step)
        ekey = _eval_key(x_int_full_snapped)

        model = _get_model(x_int_full_snapped)
        p = _params_dict_from_int_full(x_int_full_snapped)

        if stage == 'explore':
            peak_cache = explore_peak_cache
            settle_cache = explore_settle_cache
        elif stage == 'refine':
            peak_cache = refine_peak_cache
            settle_cache = refine_settle_cache
        else:
            raise ValueError(f"Unknown stage '{stage}'")

        res = np.zeros(len(cases), dtype=float)
        details: list[tuple[str, float, float, float]] = []

        for j, case in enumerate(cases):
            pk_key = (ekey, case.name)
            pred_peak = _lru_get(peak_cache, pk_key)

            if pred_peak is None:
                y0 = np.zeros(model.size(), dtype=float)
                v0 = np.zeros(model.size(), dtype=float)
                s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

                if case.settle_ms > 0.0:
                    dt = case_dt[j]
                    s_key = (ekey, dt, float(case.settle_ms))
                    settled = _lru_get(settle_cache, s_key)
                    if settled is None:
                        t_key = (dt, float(case.settle_ms))
                        if t_key not in settle_time_cache:
                            n_settle = int(round((case.settle_ms / 1000.0) / dt)) + 1
                            t_settle = dt * np.arange(n_settle)
                            a_settle = np.zeros_like(t_settle)
                            settle_time_cache[t_key] = (t_settle, a_settle)
                        t_settle, a_settle = settle_time_cache[t_key]

                        settle_out = newmark_peak_element_force(
                            model,
                            t_settle,
                            a_settle,
                            y0,
                            v0,
                            s0,
                            peak_element_index=None,
                            max_newton_iter=max_newton_iter,
                            newton_tol=newton_tol,
                        )
                        y0, v0, s0 = settle_out.y_final, settle_out.v_final, settle_out.s_final
                        _lru_put(settle_cache, s_key, (y0, v0, s0), max_settle_cache)
                    else:
                        y0, v0, s0 = settled

                out = newmark_peak_element_force(
                    model,
                    case.time_s,
                    case.accel_g,
                    y0,
                    v0,
                    s0,
                    peak_element_index=t12_element_index,
                    max_newton_iter=max_newton_iter,
                    newton_tol=newton_tol,
                )
                pred_peak = float(out.peak_force_n)
                _lru_put(peak_cache, pk_key, pred_peak, max_peak_cache)

            r = (pred_peak - targets[j]) / scales[j]
            res[j] = r
            if want_details:
                details.append((case.name, pred_peak, targets[j], r))

        cost = float(np.sum(res**2))
        return res, cost, details, p, ekey

    # -------------------------
    # Debug header
    # -------------------------
    if verbose:
        enabled_keys = [all_keys[i] for i in enabled_idx]
        fixed_keys = [all_keys[i] for i in fixed_idx]
        print('=== Peak calibration setup ===')
        print(f'  total params: {len(all_keys)}')
        print(f'  enabled: {len(enabled_keys)}')
        print(f'  fixed (disabled via lo==hi): {len(fixed_keys)}')
        if fixed_keys:
            print('  fixed keys:')
            for k in fixed_keys:
                lo, _hi = bounds[k]
                print(f'    {k} = {lo} (fixed)')
        print(f'  exploration cases: ALL ({len(cases)})')
        print(
            f'  explore: samples={explore_samples}, keep={explore_keep}, local_frac={explore_local_frac:.2f}, local_sigma={explore_local_sigma:.2f}'
        )
        print(f'  starts: n_starts={n_starts}, diversity_min_dist={diversity_min_dist:.2f}')
        print(f'  snapping: explore={snap_norm_step_explore}, refine={snap_norm_step_refine}')
        print(
            f'  explore solver: max_newton_iter={explore_max_newton_iter}, newton_tol={explore_newton_tol}'
        )
        print(
            f'  refine  solver: max_newton_iter={refine_max_newton_iter}, newton_tol={refine_newton_tol}'
        )

    # -------------------------
    # Exploration (ALL cases)
    # -------------------------
    rng = np.random.default_rng(42)

    def _sample_global() -> np.ndarray:
        x = x0_int_full.copy()
        r = lb + rng.random(lb.size) * (ub - lb)
        x[enabled_idx] = r
        return x

    def _sample_local() -> np.ndarray:
        # Local perturbation around baseline in normalized space.
        x = x0_int_full.copy()
        z0 = _norm01(x0_int_full)[enabled_idx]
        z = z0 + rng.normal(0.0, explore_local_sigma, size=z0.size)
        z = np.clip(z, 0.0, 1.0)
        denom = hi_int_full - lo_int_full
        denom = np.where(denom == 0.0, 1.0, denom)
        x[enabled_idx] = lo_int_full[enabled_idx] + z * denom[enabled_idx]
        return x

    explored: list[tuple[float, np.ndarray]] = []

    # Baseline evaluation
    res0, cost0, det0, p0, _ek0 = _evaluate_all_cases(
        x0_int_full,
        stage='explore',
        snap_step=snap_norm_step_explore,
        max_newton_iter=explore_max_newton_iter,
        newton_tol=explore_newton_tol,
        want_details=True,
    )
    explored.append((cost0, x0_int_full.copy()))
    best_cost = cost0
    best_x = x0_int_full.copy()

    if verbose:
        print(f'[explore] 0/{explore_samples} cost={cost0:.6f} best={best_cost:.6f}')
        print(f'  baseline params: {_format_param_line(p0, base_keys)}')
        print('  baseline residuals:')
        _print_case_details(det0)

    for i in range(1, explore_samples + 1):
        if rng.random() < explore_local_frac:
            x = _sample_local()
            mode = 'local'
        else:
            x = _sample_global()
            mode = 'global'

        res, cost, details, p, _ek = _evaluate_all_cases(
            x,
            stage='explore',
            snap_step=snap_norm_step_explore,
            max_newton_iter=explore_max_newton_iter,
            newton_tol=explore_newton_tol,
            want_details=True,
        )
        explored.append((cost, x))

        improved = False
        if cost < best_cost:
            best_cost = cost
            best_x = x.copy()
            improved = True

        print(
            f'[explore] {i}/{explore_samples} mode={mode} cost={cost:.6f} best={best_cost:.6f} '
            f'{"IMPROVED" if improved else ""}'.rstrip()
        )
        print(f'  params: {_format_param_line(p, base_keys)}')
        print('  residuals:')
        _print_case_details(details)

        if improved:
            res_b, cost_b, det_b, p_b, _ = _evaluate_all_cases(
                best_x,
                stage='explore',
                snap_step=snap_norm_step_explore,
                max_newton_iter=explore_max_newton_iter,
                newton_tol=explore_newton_tol,
                want_details=True,
            )
            print('  ----')
            print(f'  NEW BEST cost={cost_b:.6f}')
            print(f'  NEW BEST params: {_format_param_line(p_b, base_keys)}')
            print('  NEW BEST residuals:')
            _print_case_details(det_b)

    explored.sort(key=lambda t: t[0])

    # -------------------------
    # Seed selection (diverse among best)
    # -------------------------
    pool_n = max(explore_keep, n_starts) * 3
    pool = explored[: min(pool_n, len(explored))]

    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        za = _norm01(a)[enabled_idx]
        zb = _norm01(b)[enabled_idx]
        return float(np.linalg.norm(za - zb))

    seeds_full: list[np.ndarray] = [pool[0][1].copy()]
    for _cost, x in pool[1:]:
        if len(seeds_full) >= n_starts:
            break
        if all(_dist(x, s) >= diversity_min_dist for s in seeds_full):
            seeds_full.append(x.copy())

    # Fill if diversity filter too strict
    for _cost, x in pool[1:]:
        if len(seeds_full) >= n_starts:
            break
        if not any(np.allclose(x, s, rtol=0.0, atol=1e-14) for s in seeds_full):
            seeds_full.append(x.copy())

    if verbose:
        best_p = _params_dict_from_int_full(_snap_int_full(pool[0][1], snap_norm_step_explore))
        print(
            f'[seeds] pool_n={len(pool)}, selected_seeds={len(seeds_full)} (requested n_starts={n_starts})'
        )
        print(f'[seeds] best explore params: {_format_param_line(best_p, base_keys)}')
        print(f'[seeds] best explore cost: {pool[0][0]:.6f}')

    # -------------------------
    # Refinement (least_squares per seed)
    # -------------------------
    class EarlyStopException(Exception):
        pass

    best_result: CalibrationResult | None = None

    for start_i, x_seed_full in enumerate(seeds_full):
        x0 = x_seed_full[enabled_idx].copy()

        eval_count = 0
        unique_key_count = 0

        prev_key = None
        prev_cost_at_key = None
        stall_count = 0

        best_state: tuple[dict, float, np.ndarray] | None = None

        if verbose:
            p_seed = _params_dict_from_int_full(_snap_int_full(x_seed_full, snap_norm_step_refine))
            print(f'\n=== Refinement start {start_i + 1}/{len(seeds_full)} ===')
            print(f'  seed params: {_format_param_line(p_seed, base_keys)}')
            print(f'  refine snap step: {snap_norm_step_refine}')

        def residuals(x_var: np.ndarray) -> np.ndarray:
            nonlocal \
                eval_count, \
                unique_key_count, \
                prev_key, \
                prev_cost_at_key, \
                stall_count, \
                best_state

            x_full = x0_int_full.copy()
            x_full[enabled_idx] = x_var

            res, cost, details, p, ekey = _evaluate_all_cases(
                x_full,
                stage='refine',
                snap_step=snap_norm_step_refine,
                max_newton_iter=refine_max_newton_iter,
                newton_tol=refine_newton_tol,
                want_details=True,
            )

            eval_count += 1
            new_key = ekey != prev_key

            if new_key:
                unique_key_count += 1
                if prev_cost_at_key is not None:
                    delta = cost - prev_cost_at_key
                    if abs(delta) < cost_tol:
                        stall_count += 1
                    else:
                        stall_count = 0
                prev_cost_at_key = cost
                prev_key = ekey

            if best_state is None or cost < best_state[1]:
                best_state = (p.copy(), cost, res.copy())

            # You asked for lots of debug: print every evaluation.
            print(
                f'\n--- Start {start_i + 1}, Eval {eval_count} (unique_keys={unique_key_count}) ---'
            )
            print(f'  Params: {_format_param_line(p, base_keys)}')
            print('  Residuals per case:')
            _print_case_details(details)
            if new_key:
                print(f'  Cost: {cost:.6f}  stall={stall_count}/{stall_iters}  (key changed)')
            else:
                print(f'  Cost: {cost:.6f}  stall={stall_count}/{stall_iters}  (same key)')

            print(
                f'  Cache sizes: model_cache={len(model_cache)} '
                f'refine_peak_cache={len(refine_peak_cache)} refine_settle_cache={len(refine_settle_cache)}'
            )

            if stall_count >= stall_iters:
                raise EarlyStopException

            return res

        try:
            out = least_squares(
                residuals,
                x0,
                bounds=(lb, ub),
                max_nfev=max_nfev,
                verbose=0,
            )

            x_full = x0_int_full.copy()
            x_full[enabled_idx] = out.x

            # Evaluate final at snapped refine resolution for reporting consistency
            res_f, cost_f, _details_f, p_f, _ek_f = _evaluate_all_cases(
                x_full,
                stage='refine',
                snap_step=snap_norm_step_refine,
                max_newton_iter=refine_max_newton_iter,
                newton_tol=refine_newton_tol,
                want_details=False,
            )

            result = CalibrationResult(
                params=p_f,
                success=bool(out.success),
                cost=float(cost_f),
                residual_norm=float(np.linalg.norm(res_f)),
            )

        except EarlyStopException:
            if verbose:
                print(f'\n  -> Early stop: cost stalled for {stall_iters} unique-key steps')

            if best_state is None:
                # fallback: evaluate seed once
                res_f, cost_f, _details_f, p_f, _ek_f = _evaluate_all_cases(
                    x0_int_full,
                    stage='refine',
                    snap_step=snap_norm_step_refine,
                    max_newton_iter=refine_max_newton_iter,
                    newton_tol=refine_newton_tol,
                    want_details=False,
                )
                best_state = (p_f, cost_f, res_f)

            p_best, cost_best, res_best = best_state
            result = CalibrationResult(
                params=p_best,
                success=True,
                cost=float(cost_best),
                residual_norm=float(np.linalg.norm(res_best)),
            )

        if best_result is None or result.cost < best_result.cost:
            best_result = result

        if verbose:
            print(f'\n=== Refinement end {start_i + 1}/{len(seeds_full)} ===')
            print(f'  best cost so far = {best_result.cost:.6f}')
            print(f'  best params so far: {_format_param_line(best_result.params, base_keys)}')

    if best_result is None:
        raise RuntimeError('Calibration failed to produce any result.')

    return best_result


def calibrate_model_curves_joint(
    base_model: SpineModel,
    cases: list[CalibrationCase],
    t12_element_index: int,
    *,
    init_params: dict,
    bounds: dict[str, tuple[float, float]],
    apply_params: Callable[[SpineModel, dict], SpineModel],
    max_nfev: int = 200,
) -> CalibrationResult:
    """
    Joint curve-based calibration (waveform residuals).

    Disabled parameters (lo == hi) are removed from optimization automatically.
    """
    if not cases:
        raise ValueError('No calibration cases provided.')

    base_keys = [
        's_k_spine',
        's_c_spine',
        'buttocks_k_n_per_m',
        'buttocks_c_ns_per_m',
        'buttocks_limit_mm',
    ]

    def _key_sort(k: str) -> tuple:
        if k in base_keys:
            return (0, base_keys.index(k), k)
        if k in ('c_base_ns_per_m', 'disc_poly_k2_n_per_m2', 'disc_poly_k3_n_per_m3'):
            return (1, k, k)
        if k.startswith('maxwell_k_ratio_'):
            return (2, int(k.split('_')[-1]), k)
        if k.startswith('maxwell_tau_ms_'):
            return (3, int(k.split('_')[-1]), k)
        return (9, k, k)

    all_keys = sorted(bounds.keys(), key=_key_sort)

    for k in all_keys:
        if k not in init_params:
            raise ValueError(f"Missing init param '{k}'.")
        if k not in bounds:
            raise ValueError(f"Missing bounds for param '{k}'.")

    n_all = len(all_keys)
    lo = np.zeros(n_all, dtype=float)
    hi = np.zeros(n_all, dtype=float)
    x0 = np.zeros(n_all, dtype=float)
    use_log = np.zeros(n_all, dtype=bool)

    for i, k in enumerate(all_keys):
        lo[i] = float(bounds[k][0])
        hi[i] = float(bounds[k][1])
        if hi[i] < lo[i]:
            raise ValueError(f'Invalid bounds for {k}: [{lo[i]}, {hi[i]}]')
        x0[i] = float(init_params[k])
        if lo[i] > 0.0 and hi[i] > 0.0:
            use_log[i] = True

    enabled = (hi - lo) > 0.0
    enabled_idx = np.nonzero(enabled)[0]

    lo_int = lo.copy()
    hi_int = hi.copy()
    x0_phys = np.clip(x0, lo, hi)
    x0_int = x0_phys.copy()

    if np.any(use_log):
        lo_int[use_log] = np.log(np.clip(lo_int[use_log], 1e-300, None))
        hi_int[use_log] = np.log(np.clip(hi_int[use_log], 1e-300, None))
        x0_int[use_log] = np.log(np.clip(x0_int[use_log], 1e-300, None))

    def _phys_from_int(x_int_full: np.ndarray) -> np.ndarray:
        x = x_int_full.copy()
        x[use_log] = np.exp(x[use_log])
        return x

    if enabled_idx.size == 0:
        params = {all_keys[i]: float(x0_phys[i]) for i in range(n_all)}
        return CalibrationResult(params=params, success=True, cost=0.0, residual_norm=0.0)

    x0_var = x0_int[enabled_idx]
    lb = lo_int[enabled_idx]
    ub = hi_int[enabled_idx]

    def residuals(x_var: np.ndarray) -> np.ndarray:
        x_full = x0_int.copy()
        x_full[enabled_idx] = x_var
        x_phys = _phys_from_int(x_full)
        p = {all_keys[i]: float(x_phys[i]) for i in range(n_all)}
        model = apply_params(base_model, p)

        res_all = []
        for case in cases:
            y0s, v0s, s0s = initial_state_static(model, base_accel_g0=0.0)
            sim = newmark_nonlinear(model, case.time_s, case.accel_g, y0s, v0s, s0s)

            pred_force = sim.element_forces_n[:, t12_element_index]
            pred_force_interp = np.interp(case.force_time_s, sim.time_s, pred_force)

            scale = max(float(np.max(np.abs(case.force_n))), 1.0)
            res_all.append((pred_force_interp - case.force_n) / scale)

        return np.concatenate(res_all)

    out = least_squares(residuals, x0_var, bounds=(lb, ub), max_nfev=max_nfev)

    x_full = x0_int.copy()
    x_full[enabled_idx] = out.x
    x_phys = _phys_from_int(x_full)
    params = {all_keys[i]: float(x_phys[i]) for i in range(n_all)}

    return CalibrationResult(
        params=params,
        success=bool(out.success),
        cost=float(out.cost),
        residual_norm=float(np.linalg.norm(out.fun)),
    )
