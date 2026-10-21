"""Six-joint position-tracking benchmark with constrained MPC. SI units."""

from pathlib import Path
import json, time, argparse
from dataclasses import dataclass
from functools import partial

ROOT = Path(__file__).resolve().parent
import numpy as np
import scipy.sparse as sp
import osqp
from numba import njit

DT = 0.01
H = 10
DURATION = 8.0
LIMIT = 8.0
RATE = 1.0
QMAX = 2.6
VMAX = 2.0
HOME = np.array([0.2, -0.55, 0.8, 0.2, 0.7, -0.15])
MASS = np.array([4.0, 4.0, 3.0, 2.0, 1.5, 1.0])
INERTIA = np.array(
    [
        [0.035, 0.035, 0.02],
        [0.025, 0.10, 0.10],
        [0.03, 0.03, 0.02],
        [0.06, 0.06, 0.015],
        [0.012, 0.012, 0.008],
        [0.004, 0.004, 0.003],
    ]
)
COM = np.array(
    [[0, 0, -0.25], [-0.25, 0, 0], [0, 0, 0], [0, 0, -0.25], [0, 0, 0], [0, 0, -0.05]]
)
DH = np.array(
    [
        [0, -np.pi / 2, 0.5],
        [0.5, 0, 0],
        [0, np.pi / 2, 0],
        [0, -np.pi / 2, 0.5],
        [0, np.pi / 2, 0],
        [0, 0, 0.1],
    ]
)


@njit(cache=True)
def rot(axis, a):
    c = np.cos(a)
    s = np.sin(a)
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@njit(cache=True)
def geometry(q):
    T = np.eye(4)
    orig = np.zeros((6, 3))
    axes = np.zeros((6, 3))
    Ts = np.zeros((6, 4, 4))
    for i in range(6):
        orig[i] = T[:3, 3]
        axes[i] = T[:3, 2]
        a, al, d = DH[i]
        c = np.cos(q[i])
        s = np.sin(q[i])
        ca = np.cos(al)
        sa = np.sin(al)
        A = np.array(
            [
                [c, -s * ca, s * sa, a * c],
                [s, c * ca, -c * sa, a * s],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T = T @ A
        Ts[i] = T
    J = np.zeros((3, 6))
    Jw = axes.T.copy()
    for j in range(6):
        J[:, j] = np.cross(axes[j], T[:3, 3] - orig[j])
    return T[:3, 3].copy(), T[:3, :3].copy(), J, Jw, orig, axes, Ts


@njit(cache=True)
def mass_matrix(x, scale):
    # Euler base attitude (roll,pitch,yaw); translation fixed. 9 generalized coordinates.
    q = x[3:]
    p, R, J, Jw, o, z, Ts = geometry(q)
    Rb = rot(2, x[2]) @ rot(1, x[1]) @ rot(0, x[0])
    E = np.zeros((3, 3))
    E[:, 0] = (rot(2, x[2]) @ rot(1, x[1]))[:, 0]
    E[:, 1] = rot(2, x[2])[:, 1]
    E[2, 2] = 1.0
    M = np.zeros((9, 9))
    M[:3, :3] = E.T @ Rb @ np.diag(np.array([60.0, 70.0, 80.0])) @ Rb.T @ E
    for i in range(6):
        pc = Ts[i, :3, 3] + Ts[i, :3, :3] @ COM[i]
        rw = Rb @ pc
        JV = np.zeros((3, 9))
        JW = np.zeros((3, 9))
        JW[:, :3] = E
        for j in range(3):
            JV[:, j] = np.cross(E[:, j], rw)
        for j in range(i + 1):
            JV[:, j + 3] = Rb @ np.cross(z[j], pc - o[j])
            JW[:, j + 3] = Rb @ z[j]
        RI = Rb @ Ts[i, :3, :3]
        M += scale[i] * (
            MASS[i] * (JV.T @ JV) + JW.T @ RI @ np.diag(INERTIA[i]) @ RI.T @ JW
        )
    for j in range(6):
        M[j + 3, j + 3] += 0.02
    return M


@njit(cache=True)
def dynamics(x, v, scale):
    M = mass_matrix(x, scale)
    der = np.zeros((9, 9, 9))
    eps = 1e-5
    for k in range(9):
        xp = x.copy()
        xm = x.copy()
        xp[k] += eps
        xm[k] -= eps
        der[k] = (mass_matrix(xp, scale) - mass_matrix(xm, scale)) / (2 * eps)
    c = np.zeros(9)
    for i in range(9):
        for j in range(9):
            for k in range(9):
                c[i] += 0.5 * (der[k, i, j] + der[j, i, k] - der[i, j, k]) * v[j] * v[k]
    return M, c


@dataclass(frozen=True)
class RunOptions:
    """Sampling and plant settings used by the supplementary experiments.

    The default reproduces the primary 10 ms benchmark. The initial-joint
    index is zero-based; reference_anchor changes the target, not HOME.
    """

    dt: float = DT
    horizon: int = H
    increment_limit: float = RATE
    stage_scale: float = 1.0
    plant_substeps: int = 1
    reference_anchor: object = None
    initial_joint: object = None
    log_effective_residual: bool = False

    def __post_init__(self):
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(self.plant_substeps, int) or self.plant_substeps < 1:
            raise ValueError("plant_substeps must be a positive integer")
        if not np.isfinite(self.increment_limit) or self.increment_limit <= 0:
            raise ValueError("increment_limit must be positive and finite")
        if not np.isfinite(self.stage_scale) or self.stage_scale <= 0:
            raise ValueError("stage_scale must be positive and finite")


def integrate_plant(x, v, t, tau, scale, phases, cfg, *, dt, substeps):
    """Advance the plant using a held torque and semi-implicit Euler steps."""
    ds = dt / substeps
    for j in range(substeps):
        dist = (
            cfg.get("disturbance", 0.08)
            * (
                np.sin(1.1 * (t + j * ds) + phases)
                + 0.4 * np.cos(0.6 * (t + j * ds) + 2 * phases)
            )
            - 0.03 * v[3:]
            - 0.015 * np.tanh(v[3:] / 0.02)
        )
        M, c = dynamics(x, v, scale)
        if cfg.get("base"):
            kb = cfg["base"]
            u = -kb * x[:3] - 2 * np.sqrt(kb * np.array([60.0, 70.0, 80.0])) * v[:3]
            acc = np.linalg.solve(M, np.r_[u, tau + dist] - c)
        else:
            acc = np.r_[np.zeros(3), np.linalg.solve(M[3:, 3:], tau + dist - c[3:])]
        v = v + ds * acc
        x = x + ds * v
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 50:
            break
    return x, v


def target(t, omega, fast=False, *, anchor=None):
    p0, R0, *_ = geometry(HOME if anchor is None else anchor)
    Rt = R0 @ rot(2, omega * t)
    g = p0 + 0.10 * R0[:, 2] + 0.015 * (Rt[:, 0] - R0[:, 0])
    T = 1.0 if fast else 6.0
    s = np.clip(t / T, 0.0, 1.0)
    smooth = 10 * s**3 - 15 * s**4 + 6 * s**5
    r = 0.10 - 0.08 * smooth
    return g - r * Rt[:, 2], Rt, g, r


def observation(t, omega, cfg, rng, state, *, dt=DT, horizon=H, target_fn=target):
    DT = dt
    H = horizon
    target = target_fn
    period = cfg.get("sensor_period", DT)
    if t + 1e-9 >= state.get("next", -1):
        stamp = max(0.0, t - cfg.get("delay", 0.0))
        p, R, g, r = target(stamp, omega, cfg.get("fast", False))
        noise = rng.normal(0, cfg.get("noise", 0), 3)
        angle = rng.normal(0, cfg.get("angle_noise", 0), 3)
        state.update(next=t + period, stamp=stamp, noise=noise, angle=angle)
    # A zero-order-held target pose, with only the scheduled stand-off evolving.
    stamp = state["stamp"]
    _, Ro, go, _ = target(stamp, omega, cfg.get("fast", False))
    Ro = (
        Ro
        @ rot(0, state["angle"][0])
        @ rot(1, state["angle"][1])
        @ rot(2, state["angle"][2])
    )
    out = []
    for k in range(H + 1):
        # Ideal 100 Hz case uses analytic preview; sampled cases deliberately use pose hold.
        if (
            period == DT
            and cfg.get("delay", 0) == 0
            and cfg.get("noise", 0) == 0
            and cfg.get("angle_noise", 0) == 0
        ):
            pr, R, _, _ = target(t + k * DT, omega, cfg.get("fast", False))
        else:
            _, _, _, r = target(t + k * DT, omega, cfg.get("fast", False))
            pr = go + state["noise"] - r * Ro[:, 2]
            R = Ro
        out.append(pr)
    return np.array(out), Ro


class Predictor:
    def __init__(
        self,
        effort=2e-3,
        secondary=False,
        residual=False,
        gain=1.0,
        *,
        dt=DT,
        horizon=H,
        rate=RATE,
        stage_scale=1.0,
    ):
        self.dt = dt
        self.horizon = horizon
        self.rate = rate
        self.stage_scale = stage_scale
        DT = dt
        H = horizon
        self.effort = effort
        self.secondary = secondary
        self.residual = residual
        self.gain = gain
        self.Bq = np.zeros((6 * H, 6 * H))
        self.Bv = np.zeros_like(self.Bq)
        for i in range(H):
            for j in range(i + 1):
                self.Bq[i * 6 : i * 6 + 6, j * 6 : j * 6 + 6] = (
                    np.eye(6) * DT**2 * (i - j + 1)
                )
                self.Bv[i * 6 : i * 6 + 6, j * 6 : j * 6 + 6] = np.eye(6) * DT
        self.D = sp.diags([np.ones(H), -np.ones(H - 1)], [0, -1]).toarray()
        self.solver = None
        self.prev = np.zeros(6)
        self.W = np.zeros((9, 6))
        self.prevphi = None
        self.prevv = None
        self.prevm = None
        self.prevh = None
        self.last_dhat = np.zeros(6)

    def estimate(self, q, v):
        DT = self.dt
        # Nine normalized Gaussian rules in joint position/velocity features, all scales explicit.
        z = np.r_[q / QMAX, v / VMAX]
        centers = np.linspace(-1, 1, 9)[:, None] * np.ones((9, 12))
        logits = -np.sum((z[None, :] - centers) ** 2, axis=1) / 8.0
        raw = np.exp(logits)
        # Preserve ordinary arithmetic, but avoid 0/0 when extreme failed states underflow every rule.
        if raw.sum() < np.finfo(float).tiny:
            raw = np.exp(logits - np.max(logits))
        phi = raw / raw.sum()
        if self.residual and self.prevv is not None:
            y = self.prevm @ ((v - self.prevv) / DT) + self.prevh - self.prev
            err = y - self.W.T @ self.prevphi
            self.W = (1 - 1e-4) * self.W + 0.25 * np.outer(
                self.prevphi, np.clip(err, -2, 2)
            ) / (0.1 + self.prevphi @ self.prevphi)
            self.W = np.clip(self.W, -3.0, 3.0)
        self.prevphi = phi
        self.prevv = v.copy()
        return self.W.T @ phi if self.residual else np.zeros(6)

    def step(self, q, v, M, h, p, J, refs):
        DT = self.dt
        H = self.horizon
        RATE = self.rate
        stage = self.stage_scale
        start = time.perf_counter()
        dhat = self.estimate(q, v)
        self.last_dhat = dhat.copy()
        # Acceleration variables are condensed; every torque includes -dhat before constraining.
        T = sp.block_diag([M] * H, format="csc")
        b = np.tile(h - dhat, H)
        JH = sp.block_diag([J] * H, format="csc")
        G = JH @ self.Bq
        times = np.arange(1, H + 1) * DT
        free = np.concatenate(
            [p + tt * (J @ v) - refs[i + 1] for i, tt in enumerate(times)]
        )
        velerr = J @ v - (refs[1] - refs[0]) / DT
        # Terminal velocity regularization avoids short-horizon proportional-only behavior.
        GV = J @ self.Bv[-6:]
        vf = (refs[-1] - refs[-2]) / DT
        Wp = 4e4 * self.gain * stage
        Wv = 35.0 * self.gain
        P = (
            Wp * (G.T @ G)
            + Wv * (GV.T @ GV)
            + self.effort * stage * (T.T @ T).toarray()
            + 0.003 * stage * np.eye(6 * H)
        )
        f = (
            Wp * G.T @ free
            + Wv * GV.T @ (J @ v - vf)
            + self.effort * stage * np.asarray(T.T @ b)
        )
        if self.secondary:
            # Damping makes this a soft posture preference, not an exact null-space task.
            N = np.eye(6) - J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), J)
            asec = N @ (-2.0 * (q - HOME) - 1.5 * v)
            P += 0.025 * stage * np.eye(6 * H)
            f -= 0.025 * stage * np.tile(asec, H)
        Q = np.zeros((6 * H + 1, 6 * H + 1))
        Q[:-1, :-1] = 2 * P
        Q[-1, -1] = 2e8
        # Explicit full upper pattern allows a varying Jacobian without sparse-pattern changes.
        ii, jj = np.triu_indices(6 * H + 1)
        PP = sp.csc_matrix((Q[ii, jj], (ii, jj)), shape=Q.shape)
        Delta = sp.kron(self.D, np.eye(6), format="csc")
        DTau = Delta @ T
        db = np.asarray(Delta @ b)
        db[:6] -= self.prev
        # Include every entry of M blocks, even structural zeros.
        AA = sp.vstack(
            [
                T,
                DTau,
                sp.csc_matrix(self.Bq),
                sp.csc_matrix(self.Bq),
                sp.csc_matrix(self.Bv),
                sp.csc_matrix(self.Bv),
            ],
            format="csc",
        )
        slack = np.r_[
            np.zeros(12 * H),
            -np.ones(6 * H) * QMAX,
            np.ones(6 * H) * QMAX,
            -np.ones(6 * H) * VMAX,
            np.ones(6 * H) * VMAX,
        ]
        A = sp.hstack([AA, slack[:, None]], format="csc")
        A = sp.vstack(
            [A, sp.csc_matrix(([1.0], ([0], [6 * H])), shape=(1, 6 * H + 1))],
            format="csc",
        )
        qfree = np.concatenate([q + tt * v for tt in times])
        vfree = np.tile(v, H)
        inf = np.full(6 * H, np.inf)
        lo = np.r_[
            -LIMIT - b, -RATE - db, -inf, -QMAX - qfree, -inf, -VMAX - vfree, 0.0
        ]
        hi = np.r_[LIMIT - b, RATE - db, QMAX - qfree, inf, VMAX - vfree, inf, np.inf]
        ff = np.r_[2 * f, 0.0]
        if self.solver is None:
            self.solver = osqp.OSQP()
            self.solver.setup(
                P=PP,
                q=ff,
                A=A,
                l=lo,
                u=hi,
                verbose=False,
                eps_abs=1e-5,
                eps_rel=1e-5,
                max_iter=4000,
                polishing=False,
                adaptive_rho=True,
            )
            self.pattern = (A.indices.copy(), A.indptr.copy())
        else:
            if not np.array_equal(A.indices, self.pattern[0]) or not np.array_equal(
                A.indptr, self.pattern[1]
            ):
                raise RuntimeError("QP sparsity changed")
            self.solver.update(Px=PP.data, Ax=A.data, q=ff, l=lo, u=hi)
        result = self.solver.solve(raise_error=False)
        good = (
            result.info.status_val in (1, 2)
            and result.x is not None
            and np.all(np.isfinite(result.x))
        )
        raw = M @ result.x[:6] + h - dhat if good else self.prev.copy()
        # Clip the candidate to the amplitude and per-step increment bounds.
        low = np.maximum(-LIMIT, self.prev - RATE)
        high = np.minimum(LIMIT, self.prev + RATE)
        tau = np.clip(raw, low, high)
        guard = np.max(np.abs(tau - raw))
        self.prevm = M.copy()
        self.prevh = h.copy()
        self.prev = tau.copy()
        return tau, dict(
            ms=(time.perf_counter() - start) * 1000,
            solver_ms=result.info.solve_time * 1000,
            iterations=result.info.iter,
            status=result.info.status_val,
            fallback=int(not good),
            guard=guard,
            slack=float(result.x[-1]) if good else np.nan,
            primal=result.info.prim_res,
            dual=result.info.dual_res,
        )


def run(
    cfg,
    method="full",
    seed=0,
    effort=2e-3,
    gain=1.0,
    save=None,
    *,
    options=None,
    controller_factory=Predictor,
):
    """Run one trajectory; save all recorded samples, including an abort prefix."""
    options = options or RunOptions()
    DT = options.dt
    H = options.horizon
    RATE = options.increment_limit
    if method not in ("pid", "smc", "mpc", "mpc_posture", "mpc_residual", "full"):
        raise ValueError(f"Unknown controller: {method}")
    duration = cfg.get("duration", DURATION)
    if not np.isfinite(duration) or duration < DT:
        raise ValueError("duration must be finite and at least one control interval")
    target_fn = partial(target, anchor=options.reference_anchor)
    observation_fn = partial(observation, dt=DT, horizon=H, target_fn=target_fn)

    rng = np.random.default_rng(seed)
    steps = int(cfg.get("duration", DURATION) / DT)
    x = np.r_[np.zeros(3), HOME + rng.normal(0, cfg.get("initial", 0.003), 6)]
    v = np.zeros(9)
    if cfg.get("stress"):
        x[3:] += np.array([0.06, -0.05, 0.03, 0, 0, 0])
    if cfg.get("name") == "near_limit":
        x[-1] = 2.5
    if options.initial_joint is not None:
        joint, angle = options.initial_joint
        if not isinstance(joint, int) or not 0 <= joint < 6 or not np.isfinite(angle):
            raise ValueError(
                "initial_joint must contain an index 0..5 and finite angle"
            )
        x[3 + joint] = angle
    scale = 1 + rng.uniform(-cfg.get("inertia", 0.1), cfg.get("inertia", 0.1), 6)
    phases = rng.uniform(-np.pi, np.pi, 6)
    omega = np.deg2rad(cfg.get("tumble", 4.0))
    ctrl = controller_factory(
        effort,
        method in ("full", "mpc_posture"),
        method in ("full", "mpc_residual"),
        gain,
        dt=DT,
        horizon=H,
        rate=RATE,
        stage_scale=options.stage_scale,
    )
    obs = {}
    integ = np.zeros(3)
    prev = np.zeros(6)
    keys = [
        "time",
        "q",
        "v",
        "tau",
        "delta",
        "position",
        "reference",
        "observed",
        "rotation_error",
        "range",
        "closing_speed",
        "base",
        "dhat",
        "jac_sigma",
    ]
    if options.log_effective_residual:
        keys.append("effective_residual")
    log = {k: [] for k in keys}
    sol = []
    last_range = None
    aborted = False
    for k in range(steps):
        t = k * DT
        q = x[3:]
        qd = v[3:]
        p, R, J, Jw, *_ = geometry(q)
        Rb = rot(2, x[2]) @ rot(1, x[1]) @ rot(0, x[0])
        pw = Rb @ p
        Rw = Rb @ R
        refs, Ro = observation_fn(t, omega, cfg, rng, obs)
        true, Rt, g, r = target_fn(t, omega, cfg.get("fast", False))
        # Controller assumes stabilized base; actual world pose is logged separately in coupled trials.
        Mn, cn = dynamics(np.r_[np.zeros(3), q], np.r_[np.zeros(3), qd], np.ones(6))
        M = Mn[3:, 3:]
        h = cn[3:]
        if method in ("pid", "smc"):
            start = time.perf_counter()
            err = refs[0] - p
            ed = (refs[1] - refs[0]) / DT - J @ qd
            integ = np.clip(integ + err * DT, -0.02, 0.02)
            ac = 60 * gain * err + 14 * np.sqrt(gain) * ed + 12 * integ
            if method == "smc":
                ac = (
                    50 * gain * err
                    + 12 * np.sqrt(gain) * ed
                    + 0.2 * gain * np.tanh((ed + 5 * err) / 0.02)
                )
            a = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), ac) + (
                np.eye(6) - J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), J)
            ) @ (-2 * (q - HOME) - 1.5 * qd)
            raw = M @ a + h
            tau = np.clip(
                raw, np.maximum(-LIMIT, prev - RATE), np.minimum(LIMIT, prev + RATE)
            )
            info = dict(
                ms=(time.perf_counter() - start) * 1000,
                solver_ms=0.0,
                iterations=0,
                status=0,
                fallback=0,
                guard=np.max(np.abs(tau - raw)),
                slack=0.0,
                primal=0.0,
                dual=0.0,
            )
        else:
            tau, info = ctrl.step(q, qd, M, h, p, J, refs)
        dist = (
            cfg.get("disturbance", 0.08)
            * (np.sin(1.1 * t + phases) + 0.4 * np.cos(0.6 * t + 2 * phases))
            - 0.03 * qd
            - 0.015 * np.tanh(qd / 0.02)
        )
        Mp, cp = dynamics(x, v, scale)
        if cfg.get("base"):
            kb = cfg["base"]
            baseu = -kb * x[:3] - 2 * np.sqrt(kb * np.array([60.0, 70.0, 80.0])) * v[:3]
            acc = np.linalg.solve(Mp, np.r_[baseu, tau + dist] - cp)
        else:
            acc = np.r_[np.zeros(3), np.linalg.solve(Mp[3:, 3:], tau + dist - cp[3:])]
        rr = np.linalg.norm(g - pw)
        closing = 0.0 if last_range is None else -(rr - last_range) / DT
        last_range = rr
        ori = np.rad2deg(np.arccos(np.clip((np.trace(Rt.T @ Rw) - 1) / 2, -1, 1)))
        vals = [
            t,
            q.copy(),
            qd.copy(),
            tau.copy(),
            tau - prev,
            pw,
            true,
            refs[0],
            ori,
            rr,
            closing,
            x[:3].copy(),
            ctrl.last_dhat.copy(),
            np.linalg.svd(J, compute_uv=False)[-1],
        ]
        if options.log_effective_residual:
            vals.append(M @ acc[3:] + h - tau)
        for key, val in zip(keys, vals):
            log[key].append(val)
        sol.append(info)
        prev = tau.copy()
        if options.plant_substeps > 1:
            x, v = integrate_plant(
                x, v, t, tau, scale, phases, cfg, dt=DT, substeps=options.plant_substeps
            )
        else:
            v += DT * acc
            x += DT * v
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 50:
            aborted = True
            break
    log = {key: np.array(val) for key, val in log.items()}
    S = {key: np.array([s[key] for s in sol]) for key in sol[0]}
    e = np.linalg.norm(log["position"] - log["reference"], axis=1)
    eo = np.linalg.norm(log["position"] - log["observed"], axis=1)
    u = log["tau"]
    dq = log["delta"]
    q = log["q"]
    vel = log["v"]
    window = max(1, int(1 / DT))
    input_bad = np.any((np.abs(u) > LIMIT + 1e-8) | (np.abs(dq) > RATE + 1e-8), axis=1)
    state_bad = np.any((np.abs(q) > QMAX + 1e-8) | (np.abs(vel) > VMAX + 1e-8), axis=1)
    util = np.max(np.abs(q) / QMAX, axis=1)
    m = dict(
        case=cfg["name"],
        method=method,
        seed=seed,
        effort=effort,
        gain=gain,
        rms_mm=1000 * np.sqrt(np.mean(e**2)),
        observed_rms_mm=1000 * np.sqrt(np.mean(eo**2)),
        terminal_mm=1000 * np.mean(e[-window:]),
        peak_mm=1000 * np.max(e),
        post1_peak_mm=(
            1000 * np.max(e[int(1 / DT) :]) if len(e) > int(1 / DT) else float("nan")
        ),
        orientation_deg=float(np.mean(log["rotation_error"][-window:])),
        effort_Nm2s=DT * np.sum(u * u),
        peak_torque=np.max(np.abs(u)),
        peak_increment=np.max(np.abs(dq)),
        input_violation_pct=100 * np.mean(input_bad),
        state_violation_pct=100 * np.mean(state_bad),
        joint_margin_rad=QMAX - np.max(np.abs(q)),
        near_limit_pct=100 * np.mean(util > 0.75),
        near_limit_index=np.mean(np.maximum(util - 0.75, 0) ** 2),
        base_peak_deg=np.rad2deg(np.max(np.linalg.norm(log["base"], axis=1))),
        max_speed=np.max(
            np.linalg.norm(np.diff(log["position"], axis=0) / DT, axis=1), initial=0.0
        ),
        max_closing=np.max(log["closing_speed"]),
        terminal_range_mm=1000 * np.mean(log["range"][-window:]),
        fallbacks=int(np.sum(S["fallback"])),
        guard_count=int(np.sum(S["guard"] > 1e-6)),
        max_guard=float(np.max(S["guard"])),
        slack_max=float(np.nanmax(S["slack"])),
        time_mean_ms=np.mean(S["ms"]),
        time_p95_ms=np.quantile(S["ms"], 0.95),
        time_p99_ms=np.quantile(S["ms"], 0.99),
        time_max_ms=np.max(S["ms"]),
        over10ms_pct=100 * np.mean(S["ms"] > 10),
        solver_mean_ms=np.mean(S["solver_ms"]),
        iter_max=int(np.max(S["iterations"])),
        primal_max=float(np.max(S["primal"])),
        dual_max=float(np.max(S["dual"])),
        sigma_min=float(np.min(log["jac_sigma"])),
    )
    # Position criterion for this benchmark; orientation is not controlled.
    m["completed"] = not aborted
    m["record_duration_s"] = len(e) * DT
    m["failure_reason"] = (
        "Numerical abort: nonfinite state or coordinate magnitude above 50 rad; prefix preserved"
        if aborted
        else ""
    )
    m["position_criterion"] = bool(
        not aborted
        and np.mean(e[-window:]) < 0.01
        and np.max(e[-window:]) < 0.02
        and not np.any(input_bad | state_bad)
    )
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save, **log, **{"qp_" + k: v for k, v in S.items()})
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in m.items()}


CASES = [
    dict(name="nominal", inertia=0, disturbance=0, initial=0),
    dict(name="uncertain", inertia=0.2, disturbance=0.12),
    dict(
        name="active_constraints", inertia=0.2, disturbance=0.12, stress=True, fast=True
    ),
    dict(name="near_limit", inertia=0.2, disturbance=0.12),
    dict(name="base_stiff", inertia=0.2, disturbance=0.12, base=400.0),
    dict(name="base_compliant", inertia=0.2, disturbance=0.12, base=40.0),
    dict(
        name="noise_1mm_20Hz",
        inertia=0.2,
        disturbance=0.12,
        noise=0.001,
        angle_noise=np.deg2rad(0.1),
        sensor_period=0.05,
        delay=0.05,
    ),
    dict(
        name="noise_5mm_10Hz",
        inertia=0.2,
        disturbance=0.12,
        noise=0.005,
        angle_noise=np.deg2rad(0.5),
        sensor_period=0.1,
        delay=0.1,
    ),
    dict(name="tumble_8deg", inertia=0.2, disturbance=0.12, tumble=8.0),
    dict(name="strong_uncertainty", inertia=0.4, disturbance=0.3),
]


def validate():
    x = np.r_[[0.02, -0.03, 0.01], HOME]
    v = np.linspace(-0.1, 0.2, 9)
    s = np.ones(6)
    M, c = dynamics(x, v, s)
    eps = 1e-5
    md = (mass_matrix(x + eps * v, s) - mass_matrix(x - eps * v, s)) / (2 * eps)
    p, R, J, *_ = geometry(HOME)
    fd = np.column_stack(
        [
            (
                geometry(HOME + eps * np.eye(6)[i])[0]
                - geometry(HOME - eps * np.eye(6)[i])[0]
            )
            / (2 * eps)
            for i in range(6)
        ]
    )
    checks = dict(
        min_mass_eigenvalue=float(np.linalg.eigvalsh(M)[0]),
        mass_symmetry=float(np.max(np.abs(M - M.T))),
        kinetic_identity_error=float(abs(v @ c - 0.5 * v @ md @ v)),
        jacobian_fd_error=float(np.max(np.abs(J - fd))),
    )
    assert (
        checks["min_mass_eigenvalue"] > 0
        and checks["kinetic_identity_error"] < 1e-7
        and checks["jacobian_fd_error"] < 1e-7
    )
    return checks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--pilot", action="store_true", help="Run three exploratory trajectories"
    )
    mode.add_argument(
        "--tune", action="store_true", help="Run the archived four-gain training grid"
    )
    mode.add_argument("--batch", type=int, help="Primary experiment partition (0..3)")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()
    if args.batches != 4:
        ap.error("The archived protocol has four partitions; --batches must be 4")
    if args.batch is not None and not 0 <= args.batch < args.batches:
        ap.error("--batch must be between 0 and 3")
    (ROOT / "results").mkdir(exist_ok=True)
    checks = validate()
    print(json.dumps(checks), flush=True)
    (ROOT / "results" / "validation.json").write_text(json.dumps(checks, indent=2))
    if args.pilot:
        for name in ["active_constraints", "base_compliant", "near_limit"]:
            cfg = next(c for c in CASES if c["name"] == name)
            r = run(cfg, save=ROOT / "results" / f"pilot_{name}.npz")
            print(json.dumps(r), flush=True)
        (ROOT / "results" / "validation.json").write_text(json.dumps(checks, indent=2))
    elif args.tune:
        rows = []
        choice = {}
        cfg = next(c for c in CASES if c["name"] == "uncertain")
        for method in ["pid", "smc", "mpc"]:
            scores = []
            for gain in [0.5, 1.0, 2.0, 4.0]:
                rr = [run(cfg, method, seed, gain=gain) for seed in [10, 11, 12]]
                rows += rr
                score = np.mean(
                    [
                        r["rms_mm"]
                        + 0.05 * r["effort_Nm2s"]
                        + 0.2 * r["state_violation_pct"]
                        for r in rr
                    ]
                )
                scores.append((score, gain))
                print(method, gain, score, flush=True)
            choice[method] = min(scores)[1]
        (ROOT / "results" / "tuning.json").write_text(
            json.dumps(
                dict(
                    protocol="Minimize RMS in mm + 0.05 times torque-square integral in N^2 m^2 s + 0.2 times state violation percentage; seeds 10-12; four gain scales for every family. This is a finite-grid tuning protocol, not a global optimum or a claim of equal effort.",
                    choices=choice,
                    trials=rows,
                ),
                indent=2,
            )
        )
    elif args.batch is not None:
        tuning = json.loads((ROOT / "results" / "tuning.json").read_text())["choices"]
        jobs = []
        for ci, cfg in enumerate(CASES):
            methods = (
                ["pid", "smc", "mpc", "mpc_posture", "mpc_residual", "full"]
                if cfg["name"] == "uncertain"
                else (
                    ["mpc", "mpc_posture", "mpc_residual", "full"]
                    if cfg["name"] == "near_limit"
                    else ["full"]
                )
            )
            for method in methods:
                for seed in range(20):
                    jobs.append((cfg, method, seed + 100))
        outpath = ROOT / "results" / f"batch_{args.batch}.json"
        rows = (
            json.loads(outpath.read_text())
            if outpath.exists() and not args.rerun
            else []
        )
        done = {(r["case"], r["method"], r["seed"]) for r in rows}
        for index, (cfg, method, seed) in enumerate(jobs):
            if index % args.batches != args.batch:
                continue
            if (cfg["name"], method, seed) in done:
                continue
            dest = ROOT / "results" / f"{cfg['name']}_{method}_{seed}.npz"
            r = run(
                cfg,
                method,
                seed,
                gain=tuning[method] if method in ("pid", "smc") else 4.0,
                save=dest,
            )
            rows.append(r)
            (ROOT / "results" / f"batch_{args.batch}.json").write_text(
                json.dumps(rows, indent=2)
            )
            print(
                f"{index}/{len(jobs)} {cfg['name']} {method} {seed} RMS={r['rms_mm']:.3f}",
                flush=True,
            )
