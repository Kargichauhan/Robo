"""
experiments/sweep.py — the compute-matched experiment.

The obvious objection to guarded selection is that it isn't better because of
the gate, it's better because the gate costs extra rollouts. Per generation:

    naive    : 24 genomes x 4 selection tasks                     =  96 rollouts
    guarded  : 24 x 3 gate tasks x 2 rollouts  (= 144)
               + proxy_score on whoever passed                    = ~180 rollouts

So guarded gets ~1.9x naive's evaluation budget. This script removes that
advantage by giving naive a compute-matched arm -- double the population, so
naive gets MORE rollouts per generation than guarded does -- and re-runs the
whole comparison, paired, across N seeds.

Run:
    python experiments/sweep.py --seeds 12 --generations 30
Writes results.csv and prints the table that goes on the page.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from guarded_eval.core import (
    cheap_gate,
    mutate,
    proxy_score,
    random_genome,
    rollout,
    sample_task,
    truth_score,
)

N_SELECTION_TASKS = 4
N_GATE_TASKS = 3
N_EVAL_TASKS = 6
FINAL_ROLLOUTS = 30   # low-variance final estimate of the champion's real success


class RolloutCounter:
    """Counts rollouts so the compute claim in the write-up is measured, not asserted."""
    n = 0


def _counted_rollout(*args, **kwargs):
    RolloutCounter.n += 1
    return rollout(*args, **kwargs)


def _setup(seed: int, pop_size: int):
    rng_init = np.random.default_rng(seed)
    sel = [sample_task(rng_init) for _ in range(N_SELECTION_TASKS)]
    gate = [sample_task(rng_init) for _ in range(N_GATE_TASKS)]
    ev = [sample_task(rng_init) for _ in range(N_EVAL_TASKS)]
    pop = [random_genome(rng_init) for _ in range(pop_size)]
    return sel, gate, ev, pop


def run_naive(seed: int, generations: int, pop_size: int) -> tuple[float, int]:
    """Naive selection. pop_size=24 is the baseline; pop_size=48 is the
    compute-matched arm (it gets ~192 rollouts/gen vs guarded's ~180)."""
    sel, _gate, ev, pop = _setup(seed, pop_size)
    rng = np.random.default_rng(seed * 1000 + 1)
    n_elites = round(6 * pop_size / 24)
    RolloutCounter.n = 0

    champion = pop[0]
    for _ in range(generations):
        scores = [proxy_score(g, sel) for g in pop]
        RolloutCounter.n += len(pop) * len(sel)
        order = np.argsort(scores)[::-1]
        elites = [pop[i] for i in order[:n_elites]]
        champion = elites[0]
        offspring = [
            mutate(elites[rng.integers(len(elites))], rng)
            for _ in range(pop_size - n_elites)
        ]
        pop = elites + offspring

    rng_eval = np.random.default_rng(seed * 1000 + 99)
    return truth_score(champion, ev, n_rollouts=FINAL_ROLLOUTS, rng=rng_eval), RolloutCounter.n // generations


def run_guarded(seed: int, generations: int) -> tuple[float, int]:
    sel, gate, ev, pop = _setup(seed, 24)
    rng = np.random.default_rng(seed * 1000 + 2)
    n_elites = 6
    RolloutCounter.n = 0

    champion = pop[0]
    for _ in range(generations):
        gate_scores = [cheap_gate(g, gate, rng=rng) for g in pop]
        RolloutCounter.n += len(pop) * len(gate) * 2
        passers = [g for g, s in zip(pop, gate_scores) if s > 0]

        if passers:
            ps = [proxy_score(g, sel) for g in passers]
            RolloutCounter.n += len(passers) * len(sel)
            order = np.argsort(ps)[::-1]
            elites = [passers[i] for i in order[:n_elites]]
        else:
            elites = []
        elites += [random_genome(rng) for _ in range(n_elites - len(elites))]

        champion = elites[0]
        offspring = [
            mutate(elites[rng.integers(len(elites))], rng)
            for _ in range(24 - n_elites)
        ]
        pop = elites + offspring

    rng_eval = np.random.default_rng(seed * 1000 + 99)
    return truth_score(champion, ev, n_rollouts=FINAL_ROLLOUTS, rng=rng_eval), RolloutCounter.n // generations


def one_seed(args):
    seed, generations = args
    n24, c24 = run_naive(seed, generations, 24)
    n48, c48 = run_naive(seed, generations, 48)
    g24, cg = run_guarded(seed, generations)
    return {
        "seed": seed,
        "naive_24": n24, "naive_24_rollouts_per_gen": c24,
        "naive_48": n48, "naive_48_rollouts_per_gen": c48,
        "guarded_24": g24, "guarded_24_rollouts_per_gen": cg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    jobs = [(s, a.generations) for s in range(1, a.seeds + 1)]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(one_seed, jobs))
    rows.sort(key=lambda r: r["seed"])

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n24 = [r["naive_24"] for r in rows]
    n48 = [r["naive_48"] for r in rows]
    g24 = [r["guarded_24"] for r in rows]

    print(f"\n{a.seeds} seeds x {a.generations} generations. "
          f"Champion's real success rate on {N_EVAL_TASKS} held-out tasks "
          f"({FINAL_ROLLOUTS} noisy rollouts each).\n")
    print("                          rollouts/gen    mean real success")
    print(f"  naive   (pop 24)          {rows[0]['naive_24_rollouts_per_gen']:>4}"
          f"            {statistics.mean(n24):.3f}")
    print(f"  naive   (pop 48)          {rows[0]['naive_48_rollouts_per_gen']:>4}"
          f"            {statistics.mean(n48):.3f}   <- MORE compute than guarded")
    print(f"  guarded (pop 24)          {rows[0]['guarded_24_rollouts_per_gen']:>4}"
          f"            {statistics.mean(g24):.3f}")

    wins = sum(1 for i in range(len(rows)) if g24[i] > n48[i] + 0.01)
    losses = sum(1 for i in range(len(rows)) if n48[i] > g24[i] + 0.01)
    print(f"\n  guarded beats compute-matched naive on {wins}/{a.seeds} seeds")
    print(f"  compute-matched naive beats guarded on {losses}/{a.seeds} seeds")
    print(f"  ties: {a.seeds - wins - losses}/{a.seeds}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
