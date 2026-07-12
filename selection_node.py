"""
selection_node: runs the naive and guarded evolutionary-selection loops side
by side, one generation per timer tick, both starting from an identical
random population and identical task sets so the comparison is paired
rather than two independently-seeded runs.

Topics published:
  /champion_naive    (std_msgs/Float64MultiArray) -- this generation's best
                     naive-selected genome (flat weight vector, length
                     core.GENOME_DIM)
  /champion_guarded  (std_msgs/Float64MultiArray) -- this generation's best
                     guarded-selected genome (or, in a starved generation,
                     an honestly-reported fresh random genome; see core.py's
                     evolve_guarded docstring)
  /metrics           (std_msgs/String, JSON) -- {gen, max_gen, n_passers,
                     history: {naive: [...], guarded: [...]}}. Each history
                     entry is {gen, proxy, truth, collapsed}, computed on a
                     held-out eval task set that neither selection process
                     ever sees or selects on -- this is what lets a viewer
                     watch proxy and truth diverge honestly.

Topics subscribed:
  /control           (std_msgs/String, JSON {"cmd": "reset"|"pause"|"run"})

Parameters: pop_size, n_elites, max_generations, gen_period_s, seed.
"""

from __future__ import annotations

import json

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

from guarded_eval.core import (
    evolve_guarded,
    evolve_naive,
    proxy_score,
    random_genome,
    sample_task,
    truth_score,
)

N_SELECTION_TASKS = 4
N_GATE_TASKS = 3
N_EVAL_TASKS = 6
TRUTH_ROLLOUTS = 20   # 6 was too few: the logged truth was mostly sampling noise


class SelectionNode(Node):
    def __init__(self):
        super().__init__("selection_node")

        self.declare_parameter("pop_size", 24)
        self.declare_parameter("n_elites", 6)
        self.declare_parameter("max_generations", 30)
        self.declare_parameter("gen_period_s", 1.0)
        self.declare_parameter("seed", 9)

        self.pop_size = self.get_parameter("pop_size").value
        self.n_elites = self.get_parameter("n_elites").value
        self.max_generations = self.get_parameter("max_generations").value
        self.seed = self.get_parameter("seed").value
        period = float(self.get_parameter("gen_period_s").value)

        self.pub_champion_naive = self.create_publisher(Float64MultiArray, "/champion_naive", 10)
        self.pub_champion_guarded = self.create_publisher(Float64MultiArray, "/champion_guarded", 10)
        self.pub_metrics = self.create_publisher(String, "/metrics", 10)
        self.create_subscription(String, "/control", self._on_control, 10)

        self.state = "running"
        self._reset()

        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"selection_node up: pop_size={self.pop_size} n_elites={self.n_elites} "
            f"max_generations={self.max_generations} seed={self.seed} period={period}s"
        )

    # -- lifecycle --------------------------------------------------------
    def _reset(self):
        rng_init = np.random.default_rng(self.seed)
        self.selection_tasks = [sample_task(rng_init) for _ in range(N_SELECTION_TASKS)]
        self.gate_tasks = [sample_task(rng_init) for _ in range(N_GATE_TASKS)]
        self.eval_tasks = [sample_task(rng_init) for _ in range(N_EVAL_TASKS)]

        initial_pop = [random_genome(rng_init) for _ in range(self.pop_size)]
        self.pop_naive = [g.copy() for g in initial_pop]
        self.pop_guarded = [g.copy() for g in initial_pop]

        # separate RNG streams per condition from here on, but identical
        # starting population and task sets -- a paired comparison.
        self.rng_naive = np.random.default_rng(self.seed * 1000 + 1)
        self.rng_guarded = np.random.default_rng(self.seed * 1000 + 2)
        self.rng_log = np.random.default_rng(self.seed * 1000 + 3)

        self.gen = 0
        self.history = {"naive": [], "guarded": []}
        self.n_passers = None

        scores = [proxy_score(g, self.selection_tasks) for g in self.pop_naive]
        champ0 = self.pop_naive[int(np.argmax(scores))]
        self._log_and_publish(champ0, champ0, n_passers=None)

    def _on_control(self, msg: String):
        try:
            cmd = json.loads(msg.data).get("cmd")
        except (json.JSONDecodeError, AttributeError):
            self.get_logger().warn(f"ignoring malformed /control message: {msg.data!r}")
            return
        if cmd == "reset":
            self.get_logger().info("control: reset")
            self._reset()
        elif cmd == "pause":
            self.state = "paused"
        elif cmd == "run":
            self.state = "running"
        else:
            self.get_logger().warn(f"unknown /control cmd: {cmd!r}")

    # -- per-generation step ----------------------------------------------
    def _tick(self):
        if self.state != "running" or self.gen >= self.max_generations:
            return

        self.gen += 1
        self.pop_naive, champ_naive = evolve_naive(self.pop_naive, self.selection_tasks,
                                                     self.n_elites, self.rng_naive)
        self.pop_guarded, champ_guarded, n_passers = evolve_guarded(
            self.pop_guarded, self.selection_tasks, self.gate_tasks, self.n_elites, self.rng_guarded)

        self._log_and_publish(champ_naive, champ_guarded, n_passers)

    def _log_and_publish(self, champ_naive: np.ndarray, champ_guarded: np.ndarray, n_passers: int | None):
        p_naive = proxy_score(champ_naive, self.eval_tasks)
        t_naive = truth_score(champ_naive, self.eval_tasks, n_rollouts=TRUTH_ROLLOUTS, rng=self.rng_log)
        p_guarded = proxy_score(champ_guarded, self.eval_tasks)
        t_guarded = truth_score(champ_guarded, self.eval_tasks, n_rollouts=TRUTH_ROLLOUTS, rng=self.rng_log)

        self.history["naive"].append({"gen": self.gen, "proxy": p_naive, "truth": t_naive,
                                       "collapsed": t_naive == 0.0})
        self.history["guarded"].append({"gen": self.gen, "proxy": p_guarded, "truth": t_guarded,
                                         "collapsed": t_guarded == 0.0})
        self.n_passers = n_passers

        self.pub_champion_naive.publish(Float64MultiArray(data=champ_naive.tolist()))
        self.pub_champion_guarded.publish(Float64MultiArray(data=champ_guarded.tolist()))

        payload = {
            "gen": self.gen,
            "max_gen": self.max_generations,
            "n_passers": n_passers,
            "n_elites": self.n_elites,
            "history": self.history,
        }
        self.pub_metrics.publish(String(data=json.dumps(payload)))


def main(args=None):
    rclpy.init(args=args)
    node = SelectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
