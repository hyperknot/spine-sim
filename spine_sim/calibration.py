from __future__ import annotations

import time
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
    n_starts: int = 1,
    cost_tol: float = 1e-4,
    stall_iters: int = 10,
    # Exploration / diversity
    explore_samples: int = 200,
    explore_keep: int = 40,
    diversity_min_dist: float = 0.25,
    explore_case_count: int = 3,
    # Snap + cache quantization in normalized (0..1) internal coordinate space
    snap_norm_step: float = 0.02,
    cache_norm_step: float = 0.02,
    # Cache sizes (shared across ALL starts)
    max_model_cache: int = 128,
    max_settle_cache: int = 4096,
    max_peak_cache: int = 20000,
) -> CalibrationResult:
    """
    Joint peak-based calibration.

    Disabled rule:
      Any bound [lo, hi] with lo == hi is treated as FIXED and removed from optimization.

    Performance features:
      - Uses peak-only integrator for calibration (no full time-history allocations).
      - Caches across ALL starts:
          (param_bin -> model),
          (param_bin, dt, settle_ms -> settled y/v/s),
          (param_bin, case_name -> predicted peak).
      - Exploration stage evaluates only a subset of cases (fast), then rescoring best points
        with all cases.

    Debug / visibility:
      - Exploration prints a line for every sample with current cost + best cost.
      - Prints per-case residual table for the current-best candidate frequently.
      - Prints cache hit/miss stats.

    "Less micro-optimization":
      - snap_norm_step quantizes movement in normalized-space so tiny float differences collapse.
      - cache_norm_step bins parameters so caching hits even with small solver perturbations.
    """
    if not cases:
        raise ValueError('No calibration cases provided.')

    # Stable key order (base keys first, then extras, then indexed maxwell keys)
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

    # Validate init + bounds presence
    for k in all_keys:
        if k not in init_params:
            raise ValueError(f"Missing init param '{k}'.")
        if k not in bounds:
            raise ValueError(f"Missing bounds for param '{k}'.")

    # Case precomputes
    targets = np.asarray([float(c.target_peak_force_n) for c in cases], dtype=float)
    scales = np.asarray([max(abs(t), 1.0) for t in targets], dtype=float)
    case_dt = np.asarray([float(np.median(np.diff(c.time_s))) for c in cases], dtype=float)

    # Build internal parameterization:
    # - if lo>0 and hi>0: internal is log(phys), phys = exp(internal)
    # - else: linear internal = phys
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

    # Identify enabled vs fixed variables (lo == hi is disabled)
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

    # Initial internal full vector
    x0_phys_clipped = np.clip(x0_phys, lo_phys, hi_phys)
    x0_int_full = _internal_from_phys(x0_phys_clipped)

    # Bounds in internal space (full)
    lo_int_full = lo_phys.copy()
    hi_int_full = hi_phys.copy()
    lo_int_full[use_log] = np.log(np.clip(lo_int_full[use_log], 1e-300, None))
    hi_int_full[use_log] = np.log(np.clip(hi_int_full[use_log], 1e-300, None))

    def _params_dict_from_int_full(x_int_full: np.ndarray) -> dict:
        x_phys_full = _phys_from_internal(x_int_full)
        return {all_keys[i]: float(x_phys_full[i]) for i in range(n_all)}

    def _format_param_line(p: dict, *, keys: list[str]) -> str:
        parts = []
        for k in keys:
            if k in p:
                v = p[k]
                if isinstance(v, float):
                    parts.append(f'{k}={v:.6g}')
                else:
                    parts.append(f'{k}={v}')
        return ', '.join(parts)

    def _print_case_details(details: list[tuple[str, float, float, float]]) -> None:
        # details: (name, pred, target, resid)
        for name, pred, target, r in details:
            print(
                f'    {name}: pred={pred:.1f}N, target={target:.1f}N, '
                f'err={pred - target:+.1f}N, resid={r:+.4f}'
            )

    # If nothing is enabled, just return x0 (still compute cost once)
    if enabled_idx.size == 0:
        p = {all_keys[i]: float(x0_phys_clipped[i]) for i in range(n_all)}
        model = apply_params(base_model, p)
        res = np.zeros(len(cases), dtype=float)
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
                    model, t_settle, a_settle, y0, v0, s0, peak_element_index=None
                )
                y0, v0, s0 = settle_out.y_final, settle_out.v_final, settle_out.s_final

            out = newmark_peak_element_force(
                model, case.time_s, case.accel_g, y0, v0, s0, peak_element_index=t12_element_index
            )
            res[j] = (float(out.peak_force_n) - targets[j]) / scales[j]

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
    # Shared caches (ALL starts)
    # -------------------------
    model_cache: OrderedDict[bytes, SpineModel] = OrderedDict()
    settle_cache: OrderedDict[
        tuple[bytes, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = OrderedDict()
    peak_cache: OrderedDict[tuple[bytes, str], float] = OrderedDict()
    settle_time_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

    # Debug stats
    stats = {
        'model_hit': 0,
        'model_miss': 0,
        'peak_hit': 0,
        'peak_miss': 0,
        'settle_hit': 0,
        'settle_miss': 0,
    }
    seen_model_keys: set[bytes] = set()

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

    def _snap_int_full(x_int_full: np.ndarray) -> np.ndarray:
        if snap_norm_step <= 0.0:
            return x_int_full
        z = _norm01(x_int_full)
        zq = np.rint(z / snap_norm_step) * snap_norm_step
        zq = np.clip(zq, 0.0, 1.0)
        denom = hi_int_full - lo_int_full
        denom = np.where(denom == 0.0, 1.0, denom)
        return lo_int_full + zq * denom

    def _cache_key(x_int_full: np.ndarray) -> bytes:
        z = _norm01(x_int_full)
        step = cache_norm_step if cache_norm_step > 0.0 else 0.0
        if step > 0.0:
            q = np.rint(z / step).astype(np.int16)
            return q.tobytes()
        return z.astype(np.float32).tobytes()

    def _evaluate_int_full(
        x_int_full_raw: np.ndarray,
        *,
        case_indices: list[int] | None,
        want_details: bool,
    ) -> tuple[np.ndarray, float, tuple[list, dict | None, bytes]]:
        """
        Evaluate residuals/cost for either:
          - case_indices=None -> all cases
          - case_indices=[...] -> subset (used during exploration)
        """
        x_int_full = _snap_int_full(x_int_full_raw)
        key = _cache_key(x_int_full)

        model = _lru_get(model_cache, key)
        if model is None:
            stats['model_miss'] += 1
            seen_model_keys.add(key)
            p_model = _params_dict_from_int_full(x_int_full)
            model = apply_params(base_model, p_model)
            _lru_put(model_cache, key, model, max_model_cache)
        else:
            stats['model_hit'] += 1
            p_model = _params_dict_from_int_full(x_int_full) if want_details else None

        if case_indices is None:
            idxs = list(range(len(cases)))
        else:
            idxs = list(case_indices)

        res = np.zeros(len(idxs), dtype=float)
        details = []

        for out_i, j in enumerate(idxs):
            case = cases[j]
            pk_key = (key, case.name)
            pred_peak = _lru_get(peak_cache, pk_key)

            if pred_peak is None:
                stats['peak_miss'] += 1

                y0 = np.zeros(model.size(), dtype=float)
                v0 = np.zeros(model.size(), dtype=float)
                s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

                if case.settle_ms > 0.0:
                    dt = case_dt[j]
                    s_key = (key, dt, float(case.settle_ms))
                    settled = _lru_get(settle_cache, s_key)

                    if settled is None:
                        stats['settle_miss'] += 1
                        t_key = (dt, float(case.settle_ms))
                        if t_key not in settle_time_cache:
                            n_settle = int(round((case.settle_ms / 1000.0) / dt)) + 1
                            t_settle = dt * np.arange(n_settle)
                            a_settle = np.zeros_like(t_settle)
                            settle_time_cache[t_key] = (t_settle, a_settle)
                        t_settle, a_settle = settle_time_cache[t_key]

                        settle_out = newmark_peak_element_force(
                            model, t_settle, a_settle, y0, v0, s0, peak_element_index=None
                        )
                        y0, v0, s0 = settle_out.y_final, settle_out.v_final, settle_out.s_final
                        _lru_put(settle_cache, s_key, (y0, v0, s0), max_settle_cache)
                    else:
                        stats['settle_hit'] += 1
                        y0, v0, s0 = settled

                out = newmark_peak_element_force(
                    model,
                    case.time_s,
                    case.accel_g,
                    y0,
                    v0,
                    s0,
                    peak_element_index=t12_element_index,
                )
                pred_peak = float(out.peak_force_n)
                _lru_put(peak_cache, pk_key, pred_peak, max_peak_cache)
            else:
                stats['peak_hit'] += 1

            r = (pred_peak - targets[j]) / scales[j]
            res[out_i] = r
            if want_details:
                details.append((case.name, pred_peak, targets[j], r))

        cost = float(np.sum(res**2))
        return res, cost, (details, p_model, key)

    # -------------------------
    # Exploration stage (shared)
    # -------------------------
    rng = np.random.default_rng(42)

    # Choose which cases to use for exploration scoring
    # Default for 5 cases: [0,2,4] -> 50ms,100ms,200ms.
    if explore_case_count <= 0 or explore_case_count >= len(cases):
        explore_case_indices = None  # use all
    else:
        if len(cases) >= 5 and explore_case_count == 3:
            explore_case_indices = [0, 2, len(cases) - 1]
        else:
            picks = {0, len(cases) - 1}
            while len(picks) < explore_case_count:
                picks.add(int(rng.integers(0, len(cases))))
            explore_case_indices = sorted(picks)

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
        if explore_case_indices is None:
            print(f'  exploration cases: ALL ({len(cases)})')
        else:
            names = [cases[i].name for i in explore_case_indices]
            print(f'  exploration cases: {names} (count={len(explore_case_indices)})')
        print(f'  snap_norm_step={snap_norm_step}, cache_norm_step={cache_norm_step}')
        print(
            f'  explore_samples={explore_samples}, explore_keep={explore_keep}, n_starts={n_starts}'
        )

    def _random_int_full() -> np.ndarray:
        x = x0_int_full.copy()
        r = lb + rng.random(lb.size) * (ub - lb)
        x[enabled_idx] = r
        return x

    explored_subset: list[tuple[float, np.ndarray]] = []

    # Track best subset candidate
    best_subset_cost = None
    best_subset_x = None

    # Always include baseline
    _, cost0, extra0 = _evaluate_int_full(
        x0_int_full, case_indices=explore_case_indices, want_details=True
    )
    details0, p0, _key0 = extra0
    explored_subset.append((cost0, x0_int_full.copy()))
    best_subset_cost = cost0
    best_subset_x = x0_int_full.copy()

    if verbose:
        print(
            f'[explore] 0/{explore_samples} cost={cost0:.6f} best={best_subset_cost:.6f} '
            f'seen_model_keys={len(seen_model_keys)} model_cache={len(model_cache)} peak_cache={len(peak_cache)} settle_cache={len(settle_cache)}'
        )
        if p0 is None:
            p0 = _params_dict_from_int_full(x0_int_full)
        print(f'  best params: {_format_param_line(p0, keys=base_keys)}')
        print('  best subset residuals:')
        _print_case_details(details0)

    try:
        for i in range(1, explore_samples + 1):
            x = _random_int_full()
            want_details = True  # you asked for more debug; you said printing isn't slow for you
            _, cst, extra = _evaluate_int_full(
                x, case_indices=explore_case_indices, want_details=want_details
            )
            details, p, _key = extra

            explored_subset.append((cst, x))

            improved = False
            if best_subset_cost is None or cst < best_subset_cost:
                best_subset_cost = cst
                best_subset_x = x.copy()
                improved = True

            if p is None:
                p = _params_dict_from_int_full(_snap_int_full(x))

            print(
                f'[explore] {i}/{explore_samples} cost={cst:.6f} best={best_subset_cost:.6f} '
                f'{"IMPROVED" if improved else ""}'.rstrip()
            )
            print(
                f'  caches: seen_model_keys={len(seen_model_keys)} model_cache={len(model_cache)} '
                f'peak_cache={len(peak_cache)} settle_cache={len(settle_cache)}'
            )
            print(
                '  stats: '
                f'model_hit={stats["model_hit"]} model_miss={stats["model_miss"]} '
                f'peak_hit={stats["peak_hit"]} peak_miss={stats["peak_miss"]} '
                f'settle_hit={stats["settle_hit"]} settle_miss={stats["settle_miss"]}'
            )
            print(f'  sample params: {_format_param_line(p, keys=base_keys)}')
            print('  sample subset residuals:')
            _print_case_details(details)

            if improved and best_subset_x is not None:
                pbest = _params_dict_from_int_full(_snap_int_full(best_subset_x))
                _, _, extra_best = _evaluate_int_full(
                    best_subset_x, case_indices=explore_case_indices, want_details=True
                )
                details_best, _pbest2, _ = extra_best
                print('  ----')
                print(f'  NEW BEST subset params: {_format_param_line(pbest, keys=base_keys)}')
                print('  NEW BEST subset residuals:')
                _print_case_details(details_best)

    except KeyboardInterrupt:
        if verbose:
            print(
                '\n[explore] KeyboardInterrupt: stopping exploration early and proceeding to rescoring/refinement...'
            )

    # Keep best subset-scored points
    explored_subset.sort(key=lambda t: t[0])
    keep_n = max(explore_keep * 3, n_starts)
    explored_subset = explored_subset[:keep_n]

    if verbose:
        print(
            f'[explore] keeping top {len(explored_subset)} candidates (subset-scored) for full rescoring...'
        )

    # Rescore on ALL cases
    rescored_full: list[tuple[float, np.ndarray]] = []
    best_full_cost = None
    best_full_x = None

    try:
        for i, (_c_sub, x) in enumerate(explored_subset, start=1):
            _, cst, extra = _evaluate_int_full(x, case_indices=None, want_details=True)
            details, p, _key = extra
            rescored_full.append((cst, x))

            improved = False
            if best_full_cost is None or cst < best_full_cost:
                best_full_cost = cst
                best_full_x = x.copy()
                improved = True

            if p is None:
                p = _params_dict_from_int_full(_snap_int_full(x))

            print(
                f'[rescore] {i}/{len(explored_subset)} full_cost={cst:.6f} best_full={best_full_cost:.6f} '
                f'{"IMPROVED" if improved else ""}'.rstrip()
            )
            print(f'  params: {_format_param_line(p, keys=base_keys)}')
            print('  full residuals:')
            _print_case_details(details)

    except KeyboardInterrupt:
        if verbose:
            print('\n[rescore] KeyboardInterrupt: stopping rescoring early...')

    if not rescored_full:
        raise RuntimeError('Exploration produced no candidates to refine (unexpected).')

    rescored_full.sort(key=lambda t: t[0])
    rescored_full = rescored_full[: max(explore_keep, n_starts)]

    # Select diverse seeds among the best
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        za = _norm01(a)[enabled_idx]
        zb = _norm01(b)[enabled_idx]
        return float(np.linalg.norm(za - zb))

    seeds_full: list[np.ndarray] = []
    seeds_full.append(rescored_full[0][1].copy())

    for _, x in rescored_full[1:]:
        if len(seeds_full) >= n_starts:
            break
        if all(_dist(x, s) >= diversity_min_dist for s in seeds_full):
            seeds_full.append(x.copy())

    for _, x in rescored_full[1:]:
        if len(seeds_full) >= n_starts:
            break
        if not any(np.allclose(x, s, rtol=0.0, atol=1e-14) for s in seeds_full):
            seeds_full.append(x.copy())

    if verbose:
        print(
            f'[seeds] using {len(seeds_full)} diverse seeds for refinement (requested n_starts={n_starts})'
        )
        if best_full_x is not None:
            pbest = _params_dict_from_int_full(_snap_int_full(best_full_x))
            print(
                f'[seeds] best full params before refinement: {_format_param_line(pbest, keys=base_keys)}'
            )

    # -------------------------
    # Local optimization per seed
    # -------------------------
    class EarlyStopException(Exception):
        pass

    best_result: CalibrationResult | None = None

    for start_i, x_seed_full in enumerate(seeds_full):
        x0 = x_seed_full[enabled_idx].copy()

        iteration_count = 0
        prev_cost = None
        stall_count = 0
        best_state: tuple[dict, float, np.ndarray] | None = None  # (params, cost, res)

        # Always print immediately so you see refinement begin
        if verbose:
            p_seed = _params_dict_from_int_full(_snap_int_full(x_seed_full))
            print(f'\n=== Refinement start {start_i + 1}/{len(seeds_full)} ===')
            print(f'  seed params: {_format_param_line(p_seed, keys=base_keys)}')

        last_print_t = 0.0

        def residuals(x_var: np.ndarray) -> np.ndarray:
            nonlocal iteration_count, prev_cost, stall_count, best_state, last_print_t

            x_full = x0_int_full.copy()
            x_full[enabled_idx] = x_var

            # you wanted per-second debug; keep it (and include params/residuals)
            want_details = verbose and ((time.monotonic() - last_print_t) >= 1.0)
            res, cost, extra = _evaluate_int_full(
                x_full, case_indices=None, want_details=want_details
            )
            details, p, _key = extra

            # Note: SciPy uses 0.5*sum(r^2), but we track sum(r^2) as "cost" for consistency with your logs.
            if best_state is None or cost < best_state[1]:
                if p is None:
                    p = _params_dict_from_int_full(_snap_int_full(x_full))
                best_state = (p.copy(), cost, res.copy())

            iteration_count += 1
            delta = 0.0
            if prev_cost is not None:
                delta = cost - prev_cost
                if abs(delta) < cost_tol:
                    stall_count += 1
                else:
                    stall_count = 0
            prev_cost = cost

            if want_details:
                last_print_t = time.monotonic()
                if p is None:
                    p = _params_dict_from_int_full(_snap_int_full(x_full))

                print(f'\n--- Start {start_i + 1}, Iter {iteration_count} ---')
                print(f'  Params: {_format_param_line(p, keys=base_keys)}')
                # Also print enabled non-base params (if any are enabled)
                extras_printed = False
                for idx in enabled_idx:
                    k = all_keys[int(idx)]
                    if k in base_keys:
                        continue
                    lo, hi = bounds[k]
                    print(f'    {k} = {p[k]:.6g} (bounds=[{lo}, {hi}])')
                    extras_printed = True
                if extras_printed:
                    pass
                print('  Residuals per case:')
                _print_case_details(details)
                print(f'  Cost: {cost:.6f}  delta={delta:+.6f}  stall={stall_count}/{stall_iters}')
                print(
                    f'  Cache: model_cache={len(model_cache)} peak_cache={len(peak_cache)} settle_cache={len(settle_cache)} '
                    f'seen_model_keys={len(seen_model_keys)}'
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
            x_full = _snap_int_full(x_full)

            p_final = _params_dict_from_int_full(x_full)
            res_full, cost_full, _ = _evaluate_int_full(
                x_full, case_indices=None, want_details=False
            )

            result = CalibrationResult(
                params=p_final,
                success=bool(out.success),
                cost=float(np.sum(res_full**2)),
                residual_norm=float(np.linalg.norm(res_full)),
            )

        except EarlyStopException:
            if verbose:
                print(f'  -> Early stop: cost stalled for {stall_iters} iterations')

            if best_state is None:
                x_full = x0_int_full.copy()
                x_full[enabled_idx] = x0
                x_full = _snap_int_full(x_full)
                p0 = _params_dict_from_int_full(x_full)
                res0, cost0, _ = _evaluate_int_full(x_full, case_indices=None, want_details=False)
                best_state = (p0, cost0, res0)

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
            print(f'  best params so far: {_format_param_line(best_result.params, keys=base_keys)}')

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
