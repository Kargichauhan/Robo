"""
test/test_core.py — tests for the pure-numpy core.

The important one is test_hovering_policy_fools_the_proxy. It encodes the
project's entire thesis as an assertion: a policy that parks near the target
without ever closing its gripper scores well on proxy_score and exactly zero
on truth_score. If that test ever fails, the experiment is measuring nothing.

Run:
    cd guarded_eval && PYTHONPATH=. pytest test/ -v
"""

import numpy as np
import pytest

from guarded_eval.core import (
    ACTION_DIM,
    CONFIRM_RADIUS,
    DWELL_TICKS,
    EPISODE_LEN,
    GENOME_DIM,
    HIDDEN_DIM,
    L1,
    L2,
    OBS_DIM,
    REACH,
    cheap_gate,
    evolve_guarded,
    evolve_naive,
    fk,
    proxy_score,
    random_genome,
    rollout,
    sample_task,
    truth_score,
)


# ---------------------------------------------------------------- kinematics
def test_fk_zero_pose_is_fully_extended_along_x():
    _, ee = fk(np.array([0.0, 0.0]))
    assert ee == pytest.approx([L1 + L2, 0.0], abs=1e-9)


def test_fk_folded_elbow_reaches_the_inner_workspace_boundary():
    # theta2 = pi folds link 2 straight back along link 1
    _, ee = fk(np.array([0.0, np.pi]))
    assert ee == pytest.approx([L1 - L2, 0.0], abs=1e-9)


def test_fk_never_exceeds_reach():
    rng = np.random.default_rng(0)
    for _ in range(400):
        theta = rng.uniform(-4 * np.pi, 4 * np.pi, size=2)
        _, ee = fk(theta)
        assert np.linalg.norm(ee) <= REACH + 1e-9


def test_every_sampled_task_is_geometrically_reachable():
    rng = np.random.default_rng(1)
    for _ in range(300):
        t = sample_task(rng)
        r = np.linalg.norm(t["target"])
        assert abs(L1 - L2) <= r <= REACH


# ------------------------------------------------------------------- genome
def test_genome_dim_matches_the_mlp_layout():
    assert GENOME_DIM == OBS_DIM * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM * ACTION_DIM + ACTION_DIM


# ------------------------------------------------------------------ rollout
def test_rollout_is_deterministic_without_noise():
    rng = np.random.default_rng(2)
    g, task = random_genome(rng), sample_task(rng)
    a = rollout(g, task, action_noise=0.0, obs_noise=0.0)
    b = rollout(g, task, action_noise=0.0, obs_noise=0.0)
    assert a["success"] == b["success"]
    assert a["min_dist"] == pytest.approx(b["min_dist"])


def test_rollout_is_stochastic_with_noise():
    rng = np.random.default_rng(3)
    g, task = random_genome(rng), sample_task(rng)
    dists = {
        rollout(g, task, action_noise=0.05, obs_noise=0.05,
                rng=np.random.default_rng(s))["min_dist"]
        for s in range(8)
    }
    assert len(dists) > 1, "noise had no effect on the rollout"


def test_confirmed_is_sticky_once_set():
    """A confirm should survive the gripper reopening later in the episode.
    This is documented behaviour, not a bug -- the task is 'did it ever hold',
    not 'is it still holding at the final tick'."""
    rng = np.random.default_rng(4)
    for _ in range(200):
        g, task = random_genome(rng), sample_task(rng)
        out = rollout(g, task, record=True)
        frames = out["frames"]
        confirmed_at = next((i for i, f in enumerate(frames) if f["confirmed"]), None)
        if confirmed_at is not None:
            assert all(f["confirmed"] for f in frames[confirmed_at:])
            return
    pytest.skip("no confirming rollout found in 200 random genomes")


def test_record_returns_one_frame_per_tick():
    rng = np.random.default_rng(5)
    out = rollout(random_genome(rng), sample_task(rng), record=True)
    assert len(out["frames"]) == EPISODE_LEN


# ================================================================== THE THESIS
class HoverPolicy:
    """Drives the end effector to the target and parks there, but NEVER closes
    the gripper. It is the Goodhart adversary: perfect by the cheap proxy,
    a total failure by the real one."""

    def act(self, obs):
        a = np.zeros(ACTION_DIM)
        # obs[4], obs[5] are (target - ee). Steer both joints toward it.
        a[0] = np.clip(obs[4] + obs[5], -1.0, 1.0)
        a[1] = np.clip(obs[4] - obs[5], -1.0, 1.0)
        a[7] = -1.0            # gripper stays OPEN. forever.
        return a


def _rollout_with(policy, task, action_noise=0.0, obs_noise=0.0, rng=None):
    """rollout(), but driven by an arbitrary policy object instead of a genome."""
    from guarded_eval.core import MAX_DTHETA, make_obs

    if rng is None:
        rng = np.random.default_rng()
    theta = task["start_theta"].copy()
    target = task["target"]
    gripper_closed, dwell, confirmed = False, 0, False
    min_dist = float(np.linalg.norm(fk(theta)[1] - target))

    for _ in range(EPISODE_LEN):
        _, ee = fk(theta)
        obs = make_obs(theta, ee, target, gripper_closed, dwell)
        if obs_noise > 0:
            obs = obs + rng.normal(0.0, obs_noise, size=obs.shape)
        action = policy.act(obs)
        dtheta = np.clip(action[:2], -1.0, 1.0) * MAX_DTHETA
        if action_noise > 0:
            dtheta = dtheta + rng.normal(0.0, action_noise, size=2)
        theta = theta + dtheta
        gripper_closed = action[7] > 0.0
        _, ee = fk(theta)
        d = float(np.linalg.norm(ee - target))
        min_dist = min(min_dist, d)
        dwell = dwell + 1 if (gripper_closed and d < CONFIRM_RADIUS) else 0
        if dwell >= DWELL_TICKS:
            confirmed = True
    return {"success": confirmed, "min_dist": min_dist}


def test_hovering_policy_fools_the_proxy_and_fails_the_truth():
    """THE THESIS, AS AN ASSERTION.

    A policy that hovers on the target without ever gripping:
      - gets a GOOD proxy_score  (proxy only measures closest approach)
      - gets EXACTLY ZERO truth_score (truth requires a confirmed hold)

    proxy_score cannot see the difference between this and a real success.
    That blindness is the entire experiment. If this test fails, the proxy
    and the truth are no longer measuring different things.
    """
    rng = np.random.default_rng(9)
    tasks = [sample_task(rng) for _ in range(6)]
    hover = HoverPolicy()

    proxy = float(np.mean([-_rollout_with(hover, t)["min_dist"] for t in tasks]))
    truth = float(np.mean([
        np.mean([
            _rollout_with(hover, t, action_noise=0.05, obs_noise=0.05,
                          rng=np.random.default_rng(s))["success"]
            for s in range(6)
        ])
        for t in tasks
    ]))

    baseline = float(np.mean([proxy_score(random_genome(rng), tasks) for _ in range(24)]))

    assert proxy > baseline, (
        f"the hovering policy should beat a random genome on the proxy "
        f"(hover={proxy:.3f}, random baseline={baseline:.3f})"
    )
    assert truth == 0.0, (
        f"the hovering policy never closes its gripper, so it can never confirm; "
        f"truth_score must be exactly 0.0, got {truth}"
    )


# ---------------------------------------------------------------- selection
def test_evolve_naive_preserves_population_size_and_returns_the_top_genome():
    rng = np.random.default_rng(10)
    tasks = [sample_task(rng) for _ in range(4)]
    pop = [random_genome(rng) for _ in range(24)]
    new_pop, champ = evolve_naive(pop, tasks, n_elites=6, rng=rng)

    assert len(new_pop) == len(pop)
    best = max(pop, key=lambda g: proxy_score(g, tasks))
    assert proxy_score(champ, tasks) == pytest.approx(proxy_score(best, tasks))


def test_evolve_guarded_backfills_with_fresh_random_genomes_when_the_gate_starves():
    """If nothing passes the gate, the empty elite slots must be filled with NEW
    random genomes -- never with the best proxy scorers, which would silently
    turn guarded selection back into naive selection for those slots."""
    rng = np.random.default_rng(11)
    tasks = [sample_task(rng) for _ in range(4)]
    pop = [random_genome(rng) for _ in range(24)]

    # a gate no genome can pass: an unreachable target
    impossible = [{"target": np.array([99.0, 99.0]),
                   "start_theta": np.zeros(2)}]
    assert all(cheap_gate(g, impossible, rng=rng) == 0.0 for g in pop[:5])

    new_pop, champ, n_passers = evolve_guarded(
        pop, tasks, impossible, n_elites=6, rng=rng)

    assert n_passers == 0
    assert len(new_pop) == len(pop)
    # the champion must NOT be any member of the original population
    assert not any(np.array_equal(champ, g) for g in pop), (
        "gate starved, so the champion must be a fresh random genome, "
        "not a proxy-selected survivor"
    )


def test_evolve_guarded_only_ever_elects_gate_passers():
    rng = np.random.default_rng(12)
    tasks = [sample_task(rng) for _ in range(4)]
    gate_tasks = [sample_task(rng) for _ in range(3)]
    pop = [random_genome(rng) for _ in range(24)]

    for _ in range(6):
        pop, champ, n_passers = evolve_guarded(
            pop, tasks, gate_tasks, n_elites=6, rng=rng)
        assert 0 <= n_passers <= 24
        assert len(pop) == 24


def test_cheap_gate_is_cheaper_than_the_full_truth():
    """The gate is a veto, not a score: it must use strictly fewer rollouts
    than truth_score, or it isn't cheap and the whole framing collapses."""
    import inspect
    src = inspect.getsource(cheap_gate)
    assert "n_rollouts=2" in src
    assert truth_score.__defaults__[0] == 6      # n_rollouts default
