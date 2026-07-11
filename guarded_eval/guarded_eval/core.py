"""
Pure-numpy core: no ROS imports here at all, so this can be unit-tested and
run headlessly (see the smoke test in this package's README) independent of
whether ROS 2 is even installed. Both selection_node.py and sim_node.py
import from this module rather than duplicating any of the task/policy/
evaluator logic.

The task: a 2-link planar arm has to reach a target position and *confirm*
arrival by closing its gripper and holding it closed there for a short dwell
window, not just touching it for one lucky tick. This mirrors the abstract
"reach and confirm" task from the roboworld-goodhart synthetic experiment,
now with real forward kinematics instead of direct end-effector control, so
the arm has to actually coordinate two joint angles instead of moving in a
straight line to the target.

Goodhart's-law structure: proxy_score only ever measures how close the
trajectory's closest approach got to the target, a single deterministic
rollout. truth_score requires the full confirm event (gripper closed, held
for DWELL_TICKS consecutive ticks, within CONFIRM_RADIUS) across several
noisy rollouts. A policy can get arbitrarily good at proxy_score by hovering
near the target without ever actually confirming, which truth_score would
catch and proxy_score would not, that gap is the entire point of the demo.
"""

from __future__ import annotations

import numpy as np

# -- arm geometry / task constants --------------------------------------
L1 = 1.0
L2 = 0.8
REACH = L1 + L2
MAX_DTHETA = 0.18       # rad/step, per-joint velocity limit
CONFIRM_RADIUS = 0.22
DWELL_TICKS = 5
EPISODE_LEN = 60

OBS_DIM = 8    # theta1/pi, theta2/pi, ee_x, ee_y, target_dx, target_dy, gripper, dwell/DWELL
ACTION_DIM = 8  # echoes DROID's 7 joint + 1 gripper; only [0],[1] (joint vel) and [7] (gripper) used
HIDDEN_DIM = 16
GENOME_DIM = OBS_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * ACTION_DIM + ACTION_DIM


def fk(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics for the 2-link planar arm. theta2 is relative to
    link1's own direction (the standard 2-link convention), not absolute.
    Returns (joint1_xy, ee_xy)."""
    t1, t2 = theta
    joint1 = np.array([L1 * np.cos(t1), L1 * np.sin(t1)])
    ee = joint1 + np.array([L2 * np.cos(t1 + t2), L2 * np.sin(t1 + t2)])
    return joint1, ee


def sample_task(rng: np.random.Generator) -> dict:
    """A random target placement plus a random starting pose, the synthetic
    analog of a DROID scene setup. Target is restricted to the arm's upper
    workspace (angle in [0, pi]) at a radius comfortably inside full reach,
    so every sampled task is geometrically reachable in principle -- whether
    a given policy actually *finds* the confirming behavior is a separate
    question, which is exactly what this demo is about."""
    angle = rng.uniform(0.0, np.pi)
    radius = rng.uniform(0.35, 0.9) * REACH
    target = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    start_theta = rng.uniform(-np.pi / 2, np.pi / 2, size=2)
    return {"target": target, "start_theta": start_theta}


def make_obs(theta, ee, target, gripper_closed, dwell_count) -> np.ndarray:
    return np.array([
        theta[0] / np.pi, theta[1] / np.pi,
        ee[0], ee[1],
        target[0] - ee[0], target[1] - ee[1],
        1.0 if gripper_closed else -1.0,
        dwell_count / DWELL_TICKS,
    ], dtype=np.float32)


def random_genome(rng: np.random.Generator, scale: float = 0.5) -> np.ndarray:
    return rng.normal(0.0, scale, size=GENOME_DIM).astype(np.float32)


def mutate(genome: np.ndarray, rng: np.random.Generator, sigma: float = 0.14) -> np.ndarray:
    return genome + rng.normal(0.0, sigma, size=genome.shape).astype(np.float32)


class MLP:
    """A 2-layer tanh MLP over a flat genome vector. The policy family that
    mutation/selection operate on -- small and fast on purpose."""

    def __init__(self, genome: np.ndarray):
        i, h, o = OBS_DIM, HIDDEN_DIM, ACTION_DIM
        idx = 0
        self.w1 = genome[idx: idx + i * h].reshape(i, h); idx += i * h
        self.b1 = genome[idx: idx + h]; idx += h
        self.w2 = genome[idx: idx + h * o].reshape(h, o); idx += h * o
        self.b2 = genome[idx: idx + o]; idx += o
        assert idx == genome.shape[0] == GENOME_DIM

    def act(self, obs: np.ndarray) -> np.ndarray:
        hidden = np.tanh(obs @ self.w1 + self.b1)
        return np.tanh(hidden @ self.w2 + self.b2)


def rollout(genome: np.ndarray, task: dict, action_noise: float = 0.0, obs_noise: float = 0.0,
            rng: np.random.Generator | None = None, record: bool = False) -> dict:
    """Runs one episode. With action_noise=0 and obs_noise=0 this is a fully
    deterministic rollout (the cheap-proxy regime); with noise > 0, repeated
    calls give a distribution of outcomes (the ground-truth regime), since a
    real arm and a real evaluator are never perfectly noise-free.

    If record=True, also returns a "frames" list of per-tick state dicts
    (theta, ee, gripper_closed, confirmed-so-far), for sim_node.py to stream
    live rather than just reporting a final summary."""
    if rng is None:
        rng = np.random.default_rng()

    policy = MLP(genome)
    theta = task["start_theta"].copy()
    target = task["target"]
    gripper_closed = False
    dwell_count = 0
    confirmed = False
    min_dist = np.linalg.norm(fk(theta)[1] - target)
    frames = [] if record else None

    for _ in range(EPISODE_LEN):
        _, ee = fk(theta)
        obs = make_obs(theta, ee, target, gripper_closed, dwell_count)
        if obs_noise > 0.0:
            obs = obs + rng.normal(0.0, obs_noise, size=obs.shape).astype(np.float32)
        action = policy.act(obs)

        dtheta = np.clip(action[:2], -1.0, 1.0) * MAX_DTHETA
        if action_noise > 0.0:
            dtheta = dtheta + rng.normal(0.0, action_noise, size=2)
        theta = theta + dtheta
        gripper_closed = action[7] > 0.0

        _, ee = fk(theta)
        d = np.linalg.norm(ee - target)
        min_dist = min(min_dist, d)

        if gripper_closed and d < CONFIRM_RADIUS:
            dwell_count += 1
        else:
            dwell_count = 0
        if dwell_count >= DWELL_TICKS:
            confirmed = True

        if record:
            frames.append({"theta": theta.tolist(), "ee": ee.tolist(),
                            "gripper_closed": bool(gripper_closed), "confirmed": bool(confirmed)})

    out = {"success": bool(confirmed), "min_dist": float(min_dist)}
    if record:
        out["frames"] = frames
    return out


def proxy_score(genome: np.ndarray, tasks: list[dict]) -> float:
    """The cheap neural-sim analog: one deterministic, noise-free rollout
    per task, scored purely by closest approach. Never checks confirmation."""
    scores = [-rollout(genome, task, action_noise=0.0, obs_noise=0.0)["min_dist"] for task in tasks]
    return float(np.mean(scores))


def truth_score(genome: np.ndarray, tasks: list[dict], n_rollouts: int = 6,
                 noise: float = 0.05, rng: np.random.Generator | None = None) -> float:
    """The honest real-world analog: strict, event-based confirmation rate
    across n_rollouts noisy rollouts per task, averaged over tasks. In
    [0, 1], higher is better."""
    if rng is None:
        rng = np.random.default_rng()
    task_rates = []
    for task in tasks:
        successes = sum(
            rollout(genome, task, action_noise=noise, obs_noise=noise, rng=rng)["success"]
            for _ in range(n_rollouts)
        )
        task_rates.append(successes / n_rollouts)
    return float(np.mean(task_rates))


def cheap_gate(genome: np.ndarray, gate_tasks: list[dict], rng: np.random.Generator | None = None) -> float:
    """A veto, not a score: truth_score with far fewer rollouts than the
    full ground truth. Used only by guarded selection, never by naive."""
    return truth_score(genome, gate_tasks, n_rollouts=2, noise=0.05, rng=rng)


def evolve_naive(pop: list[np.ndarray], tasks: list[dict], n_elites: int,
                  rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray]:
    """Ranks the whole population by proxy_score alone, keeps the top
    n_elites, mutates to refill. This is the RSI-style loop: retrain/
    reselect on your own top-ranked-by-proxy population, generation after
    generation, with no ground-truth check anywhere in the process."""
    scores = [proxy_score(g, tasks) for g in pop]
    order = np.argsort(scores)[::-1]
    elites = [pop[i] for i in order[:n_elites]]
    champion = elites[0]
    offspring = [mutate(elites[rng.integers(len(elites))], rng) for _ in range(len(pop) - n_elites)]
    return elites + offspring, champion


def evolve_guarded(pop: list[np.ndarray], selection_tasks: list[dict], gate_tasks: list[dict],
                    n_elites: int, rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Gates every candidate on cheap_gate (>0 passes) before ranking by
    proxy_score. Only genomes that clear the gate are eligible to become
    elites. If fewer than n_elites genomes pass, the remaining elite slots
    are backfilled with FRESH RANDOM genomes, not the best proxy-only
    performers -- that would just reintroduce naive selection through the
    back door for those slots. Returns (new_pop, champion, n_passers):
    n_passers is logged so a caller can see when the gate itself starves."""
    gate_scores = [cheap_gate(g, gate_tasks, rng=rng) for g in pop]
    passers = [(g, s) for g, s in zip(pop, gate_scores) if s > 0]
    n_passers = len(passers)

    if passers:
        proxy_scores = [proxy_score(g, selection_tasks) for g, _ in passers]
        order = np.argsort(proxy_scores)[::-1]
        elites = [passers[i][0] for i in order[:n_elites]]
    else:
        elites = []
    elites += [random_genome(rng) for _ in range(n_elites - len(elites))]

    champion = elites[0]  # best real passer if any passed, otherwise an honest fresh-random genome
    offspring = [mutate(elites[rng.integers(len(elites))], rng) for _ in range(len(pop) - n_elites)]
    return elites + offspring, champion, n_passers
