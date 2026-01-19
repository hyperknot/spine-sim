from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from collections import OrderedDict

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
    # Snap + cache quantization in normalized (0..1) internal coordinate space
    snap_norm_step: float = 0.02,
    cache_norm_step: float = 0.02,
    # Cache sizes (shared across starts)
    max_model_cache: int = 128,
    max_settle_cache: int = 4096,
    max_peak_cache: int = 20000,
) -> CalibrationResult:
    """
    Joint peak-based calibration for:
      - spine scales: s_k_spine, s_c_spine
      - buttocks absolute params: buttocks_k_n_per_m, buttocks_c_ns_per_m, buttocks_limit_mm
      - optional model params: c_base_ns_per_m, disc_poly_k2/k3, maxwell ratios/tau ...

    Disabling rule:
      If bounds[key] has identical endpoints (lo == hi), that variable is fixed and not optimized.

    Strategy:
      1) Exploration (random sampling + diversity) to find very different seeds.
      2) least_squares refinement from each seed.
      3) Snap in normalized space to avoid micro-optimizing floating-point dust.
      4) Cache models, settling results, and per-case predicted peaks across ALL starts.
    """
    if not cases:
        raise ValueError("No calibration cases provided.")

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

    # Precompute case targets/scales and dt for settling
    targets = np.asarray([float(c.target_peak_force_n) for c in cases], dtype=float)
    scales = np.asarray([max(abs(t), 1.0) for t in targets], dtype=float)
    case_dt = np.asarray([float(np.median(np.diff(c.time_s))) for c in cases], dtype=float)

    # Build internal parameterization:
    # - if lo>0 and hi>0: optimize in log-space (internal), physical = exp(internal)
    # - otherwise: linear internal = physical
    n_all = len(all_keys)
    lo_phys = np.zeros(n_all, dtype=float)
    hi_phys = np.zeros(n_all, dtype=float)
    x0_phys = np.zeros(n_all, dtype=float)
    use_log = np.zeros(n_all, dtype=bool)

    for i, k in enumerate(all_keys):
        lo, hi = float(bounds[k][0]), float(bounds[k][1])
        if hi < lo:
            raise ValueError(f"Invalid bounds for {k}: [{lo}, {hi}]")
        lo_phys[i] = lo
        hi_phys[i] = hi
        x0_phys[i] = float(init_params[k])

        # log-space only if strictly positive interval
        if lo > 0.0 and hi > 0.0:
            use_log[i] = True

    # Identify enabled (optimizable) variables
    eps = 0.0  # treat exactly-equal as disabled; config uses identical endpoints
    enabled_mask = (hi_phys - lo_phys) > eps
    enabled_idx = np.nonzero(enabled_mask)[0]
    fixed_idx = np.nonzero(~enabled_mask)[0]

    def _phys_from_internal(x_int_full: np.ndarray) -> np.ndarray:
        x_phys = x_int_full.copy()
        x_phys[use_log] = np.exp(x_phys[use_log])
        return x_phys

    def _internal_from_phys(x_phys_full: np.ndarray) -> np.ndarray:
        x_int = x_phys_full.copy()
        x_int[use_log] = np.log(np.clip(x_int[use_log], 1e-300, None))
        return x_int

    # Initial internal full vector
    x0_phys_clipped = np.clip(x0_phys, lo_phys, hi_phys)
    x0_int_full = _internal_from_phys(x0_phys_clipped)

    # Bounds in internal space (full)
    lo_int_full = lo_phys.copy()
    hi_int_full = hi_phys.copy()
    lo_int_full[use_log] = np.log(np.clip(lo_int_full[use_log], 1e-300, None))
    hi_int_full[use_log] = np.log(np.clip(hi_int_full[use_log], 1e-300, None))

    # If nothing is enabled, just evaluate once and return
    if enabled_idx.size == 0:
        p = {all_keys[i]: float(x0_phys_clipped[i]) for i in range(n_all)}
        model = apply_params(base_model, p)
        res = []
        for j, case in enumerate(cases):
            # settle if needed (no caching needed here)
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
            r = (float(out.peak_force_n) - targets[j]) / scales[j]
            res.append(r)

        res_arr = np.asarray(res, dtype=float)
        return CalibrationResult(
            params=p,
            success=True,
            cost=float(np.sum(res_arr**2)),
            residual_norm=float(np.linalg.norm(res_arr)),
        )

    # Convert enabled-only vectors for least_squares
    x0_int = x0_int_full[enabled_idx]
    lb = lo_int_full[enabled_idx]
    ub = hi_int_full[enabled_idx]

    # -------------------------
    # Shared caches (ALL starts)
    # -------------------------
    model_cache: OrderedDict[bytes, SpineModel] = OrderedDict()
    settle_cache: OrderedDict[tuple[bytes, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
    peak_cache: OrderedDict[tuple[bytes, str], float] = OrderedDict()
    settle_time_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

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
        denom = (hi_int_full - lo_int_full)
        denom = np.where(denom == 0.0, 1.0, denom)
        z = (x_int_full - lo_int_full) / denom
        return np.clip(z, 0.0, 1.0)

    def _snap_int_full(x_int_full: np.ndarray) -> np.ndarray:
        if snap_norm_step <= 0.0:
            return x_int_full
        z = _norm01(x_int_full)
        zq = np.rint(z / snap_norm_step) * snap_norm_step
        zq = np.clip(zq, 0.0, 1.0)
        denom = (hi_int_full - lo_int_full)
        denom = np.where(denom == 0.0, 1.0, denom)
        return lo_int_full + zq * denom

    def _cache_key(x_int_full: np.ndarray) -> bytes:
        z = _norm01(x_int_full)
        step = cache_norm_step if cache_norm_step > 0.0 else 0.0
        if step > 0.0:
            q = np.rint(z / step).astype(np.int16)
            return q.tobytes()
        return z.astype(np.float32).tobytes()

    def _params_dict_from_int_full(x_int_full: np.ndarray) -> dict:
        x_phys_full = _phys_from_internal(x_int_full)
        return {all_keys[i]: float(x_phys_full[i]) for i in range(n_all)}

    def _evaluate_int_full(x_int_full_raw: np.ndarray, *, want_details: bool) -> tuple[np.ndarray, float, list]:
        # Snap to kill micro differences
        x_int_full = _snap_int_full(x_int_full_raw)
        key = _cache_key(x_int_full)

        model = _lru_get(model_cache, key)
        if model is None:
            p = _params_dict_from_int_full(x_int_full)
            model = apply_params(base_model, p)
            _lru_put(model_cache, key, model, max_model_cache)
        else:
            p = _params_dict_from_int_full(x_int_full) if want_details else None

        res = np.zeros(len(cases), dtype=float)
        details = []

        for j, case in enumerate(cases):
            pk_key = (key, case.name)
            pred_peak = _lru_get(peak_cache, pk_key)

            if pred_peak is None:
                y0 = np.zeros(model.size(), dtype=float)
                v0 = np.zeros(model.size(), dtype=float)
                s0 = np.zeros((model.n_elems(), model.n_maxwell()), dtype=float)

                if case.settle_ms > 0.0:
                    dt = case_dt[j]
                    s_key = (key, dt, float(case.settle_ms))
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
                            model, t_settle, a_settle, y0, v0, s0, peak_element_index=None
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
                )
                pred_peak = float(out.peak_force_n)
                _lru_put(peak_cache, pk_key, pred_peak, max_peak_cache)

            r = (pred_peak - targets[j]) / scales[j]
            res[j] = r
            if want_details:
                details.append((case.name, pred_peak, targets[j], r))

        cost = float(np.sum(res**2))
        return res, cost, (details, p)

    # -------------------------
    # Exploration stage (shared)
    # -------------------------
    rng = np.random.default_rng(42)

    def _random_int_full() -> np.ndarray:
        # sample enabled vars in internal space uniformly across [lb, ub], fixed stay at init
        x = x0_int_full.copy()
        r = lb + rng.random(lb.size) * (ub - lb)
        x[enabled_idx] = r
        return x

    explored: list[tuple[float, np.ndarray]] = []
    # always include the base point
    base_res, base_cost, _ = _evaluate_int_full(x0_int_full, want_details=False)
    explored.append((base_cost, x0_int_full.copy()))

    for _ in range(max(explore_samples, 0)):
        x = _random_int_full()
        _, cost, _ = _evaluate_int_full(x, want_details=False)
        explored.append((cost, x))

    explored.sort(key=lambda t: t[0])
    explored = explored[: max(explore_keep, n_starts)]

    # Select diverse seeds among the best
    seeds_full: list[np.ndarray] = []
    seeds_full.append(explored[0][1].copy())

    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        za = _norm01(a)[enabled_idx]
        zb = _norm01(b)[enabled_idx]
        return float(np.linalg.norm(za - zb))

    for _, x in explored[1:]:
        if len(seeds_full) >= n_starts:
            break
        if all(_dist(x, s) >= diversity_min_dist for s in seeds_full):
            seeds_full.append(x.copy())

    # If not enough, fill with next best regardless of distance
    for _, x in explored[1:]:
        if len(seeds_full) >= n_starts:
            break
        if not any(np.allclose(x, s, rtol=0.0, atol=1e-14) for s in seeds_full):
            seeds_full.append(x.copy())

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

        last_print_t = 0.0

        def residuals(x_var: np.ndarray) -> np.ndarray:
            nonlocal iteration_count, prev_cost, stall_count, best_state, last_print_t

            x_full = x0_int_full.copy()
            x_full[enabled_idx] = x_var

            want_details = verbose and (time.monotonic() - last_print_t) >= 1.0
            res, cost, extra = _evaluate_int_full(x_full, want_details=want_details)
            details, p = extra

            if best_state is None or cost < best_state[1]:
                best_state = (p.copy() if p is not None else _params_dict_from_int_full(x_full), cost, res.copy())

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
                print(f"\n=== Start {start_i + 1}/{len(seeds_full)} | Eval {iteration_count} ===")
                print("  Params:")
                for k in base_keys:
                    if k in p:
                        print(f"    {k} = {p[k]:.6g}")
                # print only enabled extras (to avoid spam)
                for k in all_keys:
                    if k in base_keys:
                        continue
                    lo, hi = bounds[k]
                    if abs(float(hi) - float(lo)) <= 0.0:
                        continue
                    if k in p:
                        print(f"    {k} = {p[k]:.6g}   (bounds=[{lo}, {hi}])")
                print("  Residuals per case:")
                for name, pred, target, r in details:
                    print(f"    {name}: pred={pred:.1f}N, target={target:.1f}N, err={pred-target:+.1f}N, resid={r:+.4f}")
                print(f"  Cost: {cost:.6f}  delta={delta:+.6f}  stall={stall_count}/{stall_iters}")

            if stall_count >= stall_iters:
                raise EarlyStopException()

            return res

        try:
            out = least_squares(
                residuals,
                x0,
                bounds=(lb, ub),
                max_nfev=max_nfev,
                verbose=0,
            )
            # Build final params from solver output
            x_full = x0_int_full.copy()
            x_full[enabled_idx] = out.x
            x_full = _snap_int_full(x_full)

            p = _params_dict_from_int_full(x_full)
            res, cost, _ = _evaluate_int_full(x_full, want_details=False)

            result = CalibrationResult(
                params=p,
                success=bool(out.success),
                cost=float(np.sum(res**2)),
                residual_norm=float(np.linalg.norm(res)),
            )

        except EarlyStopException:
            if best_state is None:
                x_full = x0_int_full.copy()
                x_full[enabled_idx] = x0
                p = _params_dict_from_int_full(x_full)
                res, cost, _ = _evaluate_int_full(x_full, want_details=False)
                best_state = (p, cost, res)

            p_best, cost_best, res_best = best_state
            result = CalibrationResult(
                params=p_best,
                success=True,
                cost=float(cost_best),
                residual_norm=float(np.linalg.norm(res_best)),
            )

        if best_result is None or result.cost < best_result.cost:
            best_result = result

    if best_result is None:
        raise RuntimeError("Calibration failed to produce any result.")

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
    Joint curve-based calibration (waveform residuals) with same parameter set as peaks.

    Disabled parameters (lo == hi) are removed from optimization automatically.
    """
    if not cases:
        raise ValueError("No calibration cases provided.")

    # Stable ordering like peaks
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
        x0[i] = float(init_params[k])
        if lo[i] > 0.0 and hi[i] > 0.0:
            use_log[i] = True

    enabled = (hi - lo) > 0.0
    enabled_idx = np.nonzero(enabled)[0]

    # internal
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
