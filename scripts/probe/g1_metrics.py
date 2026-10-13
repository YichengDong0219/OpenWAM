"""G1 metrics (K3), implementing G1_TASKS.md section 3. Pure numpy, CPU only.

Physical actions are the unnormalized 20-D RoboTwin EEF layout
``[l_xyz(3), l_rot6d(6), l_grip(1), r_xyz(3), r_rot6d(6), r_grip(1)]`` (trace
``final.actions_physical``, shape (32, 20)). Velocities are in the model's normalized
80-D space; only the active dims (``meta.active_action_mask``) are used.

Run ``python scripts/probe/g1_metrics.py selftest`` for the synthetic checks.
"""

from __future__ import annotations

import sys

import numpy as np

ARMS = {"left": slice(0, 10), "right": slice(10, 20)}
XYZ, ROT6D, GRIP = slice(0, 3), slice(3, 9), 9


def as_np(x) -> np.ndarray:
    """float64 numpy view of numpy arrays or torch tensors (incl. bfloat16, which numpy lacks)."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _arm(actions: np.ndarray, arm: str) -> np.ndarray:
    a = as_np(actions)
    if a.ndim != 2 or a.shape[1] != 20:
        raise ValueError(f"expected (T, 20) physical actions, got {a.shape}")
    return a[:, ARMS[arm]]


def d_xyz(a: np.ndarray, b: np.ndarray) -> dict:
    """Per-step Euclidean xyz distance of each arm (metres); max and mean over steps and arms."""
    per = np.stack([np.linalg.norm(_arm(a, arm)[:, XYZ] - _arm(b, arm)[:, XYZ], axis=1) for arm in ARMS])
    return {"max": float(per.max()), "mean": float(per.mean()),
            "left_max": float(per[0].max()), "right_max": float(per[1].max())}


def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3), Gram-Schmidt on the first two columns (matches action_conversion)."""
    r = np.asarray(r6d, dtype=np.float64)
    a1, a2 = r[..., 0:3], r[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def degenerate_rot6d(actions, eps: float = 1e-6) -> int:
    """Count rot6d entries whose first column or Gram-Schmidt residual is (near) zero."""
    n = 0
    for arm in ARMS:
        r = _arm(actions, arm)[:, ROT6D]
        a1, a2 = r[:, 0:3], r[:, 3:6]
        n1 = np.linalg.norm(a1, axis=1)
        b1 = a1 / np.maximum(n1[:, None], 1e-12)
        res = np.linalg.norm(a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1, axis=1)
        n += int(((n1 < eps) | (res < eps)).sum())
    return n


def rot_geodesic_deg(a: np.ndarray, b: np.ndarray) -> dict:
    """Rotation convention matches benchmarks/utils/action_conversion._rot6d_to_matrix
    (rot6d = first two COLUMNS; Gram-Schmidt; columns b1, b2, b1 x b2)."""
    out = {"degenerate_rot6d": degenerate_rot6d(a) + degenerate_rot6d(b)}
    for arm in ARMS:
        ra, rb = rot6d_to_matrix(_arm(a, arm)[:, ROT6D]), rot6d_to_matrix(_arm(b, arm)[:, ROT6D])
        rel = np.einsum("tji,tjk->tik", ra, rb)
        cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        out[f"{arm}_max_deg"] = float(ang.max())
        out[f"{arm}_mean_deg"] = float(ang.mean())
    out["max_deg"] = max(out["left_max_deg"], out["right_max_deg"])
    return out


def gripper_diff(a: np.ndarray, b: np.ndarray) -> dict:
    """Continuous gripper channel differences only (no 0.5 event threshold: this checkpoint has no
    binary_action_dims and event timing needs its own calibrated, pre-registered threshold)."""
    out = {}
    for arm in ARMS:
        d = np.abs(_arm(a, arm)[:, GRIP] - _arm(b, arm)[:, GRIP])
        out[f"{arm}_mean_abs"] = float(d.mean())
        out[f"{arm}_max_abs"] = float(d.max())
    return out


def chunk_diff(a: np.ndarray, b: np.ndarray) -> dict:
    return {"xyz": d_xyz(a, b), "rot": rot_geodesic_deg(a, b), "grip": gripper_diff(a, b)}


def active_vector(v, active_mask) -> np.ndarray:
    v = as_np(v)
    mask = as_np(active_mask).astype(bool)
    if v.shape[-1] != mask.shape[0]:
        raise ValueError(f"velocity last dim {v.shape[-1]} != mask {mask.shape[0]}")
    return v[..., mask]


def rel_velocity_response(v_swap, v_base, active_mask) -> dict:
    """Velocity difference on the active dims (normalized model space, not physical motion)."""
    s, b = active_vector(v_swap, active_mask), active_vector(v_base, active_mask)
    diff = s - b
    num, den = float(np.linalg.norm(diff)), float(np.linalg.norm(b))
    return {"l2": num, "rms": float(np.sqrt(np.mean(diff**2))), "max_abs": float(np.abs(diff).max()),
            "base_l2": den, "base_rms": float(np.sqrt(np.mean(b**2))), "rel_l2": (num / den) if den > 0 else None}


def grid_decomposition(actions) -> dict:
    """Two-way sum-of-squares split of an (n_video, n_action, T, 20) grid on both arms' xyz.

    Returns the fraction of the total sum of squares (about the grand mean, summed over
    the 32 x 6 xyz coordinates) attributable to the video seed, the action seed and their
    interaction (one observation per cell, so the interaction absorbs the residual).
    """
    g = as_np(actions)
    if g.ndim != 4 or g.shape[-1] != 20:
        raise ValueError(f"expected (V, A, T, 20) grid, got {g.shape}")
    y = np.concatenate([g[..., 0:3], g[..., 10:13]], axis=-1)  # (V, A, T, 6)
    grand = y.mean(axis=(0, 1), keepdims=True)
    mv = y.mean(axis=1, keepdims=True)
    ma = y.mean(axis=0, keepdims=True)
    n_v, n_a = y.shape[:2]
    ss_total = float(((y - grand) ** 2).sum())
    ss_video = float(n_a * ((mv - grand) ** 2).sum())
    ss_action = float(n_v * ((ma - grand) ** 2).sum())
    ss_inter = float(((y - mv - ma + grand) ** 2).sum())
    raw = {"ss_total": ss_total, "ss_video": ss_video, "ss_action": ss_action, "ss_interaction": ss_inter}
    if ss_total == 0:
        return {**raw, "video": None, "action": None, "interaction": None}
    return {**raw, "video": ss_video / ss_total, "action": ss_action / ss_total, "interaction": ss_inter / ss_total}


def donor_alignment(a_swap, a_r, a_d, min_norm: float) -> dict:
    """Projection of the swap-induced change onto the donor-minus-receiver direction (xyz, both arms)."""
    def xyz(x):
        x = np.asarray(x, dtype=np.float64)
        return np.concatenate([x[:, 0:3], x[:, 10:13]], axis=-1).ravel()
    delta, d = xyz(a_swap) - xyz(a_r), xyz(a_d) - xyz(a_r)
    dn = float(np.linalg.norm(d))
    if dn < min_norm:
        return {"reported": False, "d_norm": dn, "reason": "donor-receiver difference below min_norm"}
    proj = float(delta @ d) / dn**2
    resid = float(np.linalg.norm(delta - proj * d)) / dn
    return {"reported": True, "d_norm": dn, "projection": proj, "orthogonal_residual": resid}


def arm_motion(actions, current_state) -> dict:
    """Commanded motion per arm relative to the current EEF state (both in the 20-D layout).

    D_a = max_k |xyz_a[k] - xyz_a^current|; P_a = |xyz_a[0] - current| + sum_k |xyz_a[k+1] - xyz_a[k]|.
    These are commanded targets, not executed trajectories.
    """
    cur = as_np(current_state).reshape(-1)
    if cur.shape[0] != 20:
        raise ValueError(f"current_state must be 20-D, got {cur.shape}")
    out = {}
    for arm, sl in ARMS.items():
        xyz = _arm(actions, arm)[:, XYZ]
        c = cur[sl][XYZ]
        out[f"{arm}_D"] = float(np.linalg.norm(xyz - c, axis=1).max())
        out[f"{arm}_P"] = float(np.linalg.norm(xyz[0] - c) + np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    return out


def active_arm(motion: dict, tau_on: float, tau_off: float) -> str:
    if not (tau_on > tau_off >= 0):
        raise ValueError("require tau_on > tau_off >= 0")
    dl, dr = motion["left_D"], motion["right_D"]
    if dl > tau_on and dr > tau_on:
        return "both"
    if dl < tau_off and dr < tau_off:
        return "neither"
    if dl > tau_on and dr < tau_off:
        return "left"
    if dr > tau_on and dl < tau_off:
        return "right"
    return "uncertain"


def _selftest() -> None:
    rng = np.random.default_rng(0)
    T = 32
    base = np.zeros((T, 20))
    base[:, 3:9] = base[:, 13:19] = [1, 0, 0, 0, 1, 0]  # identity rot6d
    # d_xyz: shift left xyz by 0.03 m at one step
    b = base.copy(); b[5, 0:3] += [0.03, 0, 0]
    r = d_xyz(base, b); assert abs(r["max"] - 0.03) < 1e-12 and r["right_max"] == 0
    # rotation: 90 deg about z on the right arm
    c = base.copy(); c[:, 13:19] = [0, 1, 0, -1, 0, 0]
    assert abs(rot_geodesic_deg(base, c)["right_max_deg"] - 90) < 1e-9
    # gripper crossing shift
    g1, g2 = base.copy(), base.copy(); g1[:, 9] = 1.0; g2[:, 9] = 1.0
    g1[10:, 9] = 0.0; g2[14:, 9] = 0.0
    gd = gripper_diff(g1, g2); assert abs(gd["left_mean_abs"] - 4 / T) < 1e-12 and gd["left_max_abs"] == 1.0
    assert rot_geodesic_deg(base, base)["degenerate_rot6d"] == 0
    z = base.copy(); z[3, 3:9] = 0; assert degenerate_rot6d(z) == 1
    # velocity response only on active dims
    mask = np.zeros(80, bool); mask[0:10] = mask[34:44] = True
    v = rng.normal(size=(T, 80)); w = v.copy(); w[:, 50] += 100  # inactive dim change ignored
    rv = rel_velocity_response(w, v, mask); assert rv["l2"] == 0 and rv["max_abs"] == 0 and rv["rel_l2"] == 0
    # grid decomposition: pure video effect
    grid = np.zeros((4, 4, T, 20)); grid[..., 3:9] = grid[..., 13:19] = [1, 0, 0, 0, 1, 0]
    for i in range(4):
        grid[i, :, :, 0] = i
    dec = grid_decomposition(grid); assert abs(dec["video"] - 1) < 1e-12 and dec["action"] == 0
    assert abs(dec["ss_video"] - dec["ss_total"]) < 1e-9
    # donor alignment: swap equals donor -> projection 1, residual 0
    d = base.copy(); d[:, 10:13] += 0.05
    al = donor_alignment(d, base, d, min_norm=1e-6); assert abs(al["projection"] - 1) < 1e-12 and al["orthogonal_residual"] < 1e-12
    assert donor_alignment(d, base, base, min_norm=1e-6)["reported"] is False
    # arm motion / active arm: left moves 0.1 m, right still
    mv = base.copy(); mv[:, 0] = np.linspace(0, 0.1, T)
    m = arm_motion(mv, np.concatenate([base[0, :10], base[0, 10:]]))
    assert abs(m["left_D"] - 0.1) < 1e-12 and m["right_D"] == 0 and abs(m["left_P"] - 0.1) < 1e-12
    assert active_arm(m, tau_on=0.02, tau_off=0.005) == "left"
    try:
        import torch
        tv = torch.tensor(v, dtype=torch.bfloat16)
        assert rel_velocity_response(tv, tv, torch.tensor(mask))["l2"] == 0
    except ImportError:
        pass
    print("g1_metrics selftest: all checks passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        _selftest()
    else:
        raise SystemExit(__doc__)
