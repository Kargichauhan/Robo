"""
sim_node: two live 2-link arms stepped in lock-step on the same shared
target, each driven by whatever champion genome selection_node last
published for its condition, so a viewer can watch the *behavioral*
consequence of naive vs. guarded selection, not just the numbers.

Topics published:
  /naive/joint_states    (sensor_msgs/JointState, names [joint1, joint2])
  /guarded/joint_states  (sensor_msgs/JointState, names [joint1, joint2])
  /robot_state           (std_msgs/String, JSON): {l1, l2, reach,
                         confirm_radius, dwell_max, target, naive: {theta, ee,
                         gripper_closed, confirmed, dwell_count, t,
                         success_rate}, guarded: {...same shape...}}

Topics subscribed:
  /champion_naive    (std_msgs/Float64MultiArray) -- swaps the naive arm's
                     policy to whatever selection_node last picked
  /champion_guarded  (std_msgs/Float64MultiArray) -- same, for the guarded arm

Parameters: rate_hz, seed.

Each arm's own episode is stepped with small action/observation noise (this
is the "ground truth" regime from core.py, not the noise-free proxy regime),
since the whole point of the demo is to show what actually happens when you
run the champion for real, not what the proxy claimed. When both arms reach
core.EPISODE_LEN ticks, each arm's confirmed/not result is pushed into its
own rolling 20-episode success-rate window, and a fresh shared target (and
shared starting pose, for a fair comparison) is sampled for the next episode.
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from guarded_eval.core import (
    CONFIRM_RADIUS,
    DWELL_TICKS,
    EPISODE_LEN,
    L1,
    L2,
    MAX_DTHETA,
    MLP,
    REACH,
    fk,
    make_obs,
    random_genome,
    sample_task,
)

NOISE = 0.05
SUCCESS_WINDOW = 20
JOINT_NAMES = ["joint1", "joint2"]


class ArmState:
    def __init__(self, genome: np.ndarray):
        self.set_genome(genome)
        self.theta = np.zeros(2)
        self.gripper_closed = False
        self.dwell_count = 0
        self.confirmed = False
        self.success_window: deque[int] = deque(maxlen=SUCCESS_WINDOW)

    def set_genome(self, genome: np.ndarray):
        self.policy = MLP(genome)

    def start_episode(self, start_theta: np.ndarray):
        self.theta = start_theta.copy()
        self.gripper_closed = False
        self.dwell_count = 0
        self.confirmed = False

    def step(self, target: np.ndarray, rng: np.random.Generator):
        _, ee = fk(self.theta)
        obs = make_obs(self.theta, ee, target, self.gripper_closed, self.dwell_count)
        obs = obs + rng.normal(0.0, NOISE, size=obs.shape).astype(np.float32)
        action = self.policy.act(obs)

        dtheta = np.clip(action[:2], -1.0, 1.0) * MAX_DTHETA + rng.normal(0.0, NOISE, size=2)
        self.theta = self.theta + dtheta
        self.gripper_closed = bool(action[7] > 0.0)

        _, ee = fk(self.theta)
        d = float(np.linalg.norm(ee - target))
        if self.gripper_closed and d < CONFIRM_RADIUS:
            self.dwell_count += 1
        else:
            self.dwell_count = 0
        if self.dwell_count >= DWELL_TICKS:
            self.confirmed = True

    @property
    def success_rate(self) -> float:
        if not self.success_window:
            return 0.0
        return sum(self.success_window) / len(self.success_window)


class SimNode(Node):
    def __init__(self):
        super().__init__("sim_node")

        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("seed", 1)

        rate_hz = float(self.get_parameter("rate_hz").value)
        self.seed = self.get_parameter("seed").value
        self.rng = np.random.default_rng(self.seed)

        self.pub_naive_js = self.create_publisher(JointState, "/naive/joint_states", 10)
        self.pub_guarded_js = self.create_publisher(JointState, "/guarded/joint_states", 10)
        self.pub_state = self.create_publisher(String, "/robot_state", 10)

        self.create_subscription(Float64MultiArray, "/champion_naive",
                                  lambda msg: self.naive.set_genome(np.array(msg.data)), 10)
        self.create_subscription(Float64MultiArray, "/champion_guarded",
                                  lambda msg: self.guarded.set_genome(np.array(msg.data)), 10)

        self.naive = ArmState(random_genome(self.rng))
        self.guarded = ArmState(random_genome(self.rng))
        self._new_episode()
        self.t = 0

        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(f"sim_node up: rate_hz={rate_hz} seed={self.seed}")

    def _new_episode(self):
        task = sample_task(self.rng)
        self.target = task["target"]
        self.naive.start_episode(task["start_theta"])
        self.guarded.start_episode(task["start_theta"])

    def _tick(self):
        self.naive.step(self.target, self.rng)
        self.guarded.step(self.target, self.rng)
        self.t += 1

        if self.t >= EPISODE_LEN:
            self.naive.success_window.append(int(self.naive.confirmed))
            self.guarded.success_window.append(int(self.guarded.confirmed))
            self.t = 0
            self._new_episode()

        self._publish()

    def _publish(self):
        now = self.get_clock().now().to_msg()
        for arm, pub in ((self.naive, self.pub_naive_js), (self.guarded, self.pub_guarded_js)):
            js = JointState()
            js.header.stamp = now
            js.name = JOINT_NAMES
            js.position = [float(arm.theta[0]), float(arm.theta[1])]
            pub.publish(js)

        def arm_payload(arm: ArmState) -> dict:
            _, ee = fk(arm.theta)
            return {
                "theta": arm.theta.tolist(),
                "ee": ee.tolist(),
                "gripper_closed": arm.gripper_closed,
                "confirmed": arm.confirmed,
                "dwell_count": arm.dwell_count,
                "t": self.t,
                "success_rate": arm.success_rate,
            }

        payload = {
            "l1": L1, "l2": L2, "reach": REACH, "confirm_radius": CONFIRM_RADIUS,
            "dwell_max": DWELL_TICKS,
            "target": self.target.tolist(),
            "naive": arm_payload(self.naive),
            "guarded": arm_payload(self.guarded),
        }
        self.pub_state.publish(String(data=json.dumps(payload)))


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
