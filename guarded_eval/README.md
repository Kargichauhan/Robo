# guarded_eval

A live ROS 2 demo of Goodhart's law: two simulated 2-link arms attempt the
same reach-and-confirm task side by side. One arm's controller is selected
purely by a cheap proxy score every generation (an RSI-style loop, no
ground-truth check anywhere). The other is gated on a cheap ground-truth
probe before being ranked by the same proxy. Watch what happens to each
one's *real* success rate over time, not just its proxy score.

This mirrors the `roboworld-goodhart` synthetic experiment (same evaluator
split, same selection mechanics), rebuilt as a real ROS 2 package with two
live nodes and a browser dashboard over rosbridge, instead of an offline
numpy script.

## What's real vs. synthetic here

- **Real**: ROS 2 nodes, topics, a launch file, a genuine evolutionary
  selection loop running generation-by-generation on a wall-clock timer, and
  a live browser dashboard driven entirely by rosbridge messages.
- **Synthetic**: the arm is a 2-link planar kinematic model (`core.py`'s
  `fk()`), not a physics simulation, no forces, no collisions, no gravity.
  The "cheap proxy" and "ground-truth" evaluators are both synthetic
  functions over that same kinematic model, not a real learned simulator and
  a real robot. The point is the selection *mechanism*, not robot realism.

## Prerequisites

ROS 2 (Humble, Iron, or Jazzy), `colcon`, and the `rosbridge_server` package.
If this machine has no ROS 2 installed and you have no sudo (as was the case
when this package was built and verified), see "No-sudo ROS 2 via RoboStack"
below for a real, working alternative.

## Build and run

**Recommended: `run_demo.sh`**, at the workspace root. It kills any of your
own orphaned instances from previous runs, scans for a genuinely free port
with a real bind test (not `ss`/`lsof`, which can miss another user's
process), launches with that port and `ROS_DOMAIN_ID=77`, and prints the
exact dashboard URL to open:

```bash
cd guarded_eval_ws
colcon build --symlink-install   # only needed after editing the package
bash run_demo.sh
```

It prints something like:
```
DASHBOARD: set WS_PORT to 19091 and open dashboard.html
  -> dashboard.html?port=19091
```
Serve the dashboard folder and open exactly that URL (see "Then open the
dashboard" below) — no manual port bookkeeping needed.

**Manual launch**, if you want direct control over seed/generation count:

```bash
cd guarded_eval_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch guarded_eval demo.launch.py
# or, to change the seed / generation count:
ros2 launch guarded_eval demo.launch.py seed:=7 max_generations:=20
# if port 9090 is already taken by something else (common on a shared machine):
ros2 launch guarded_eval demo.launch.py port:=19090
```

**On a shared/multi-user machine, set a unique `ROS_DOMAIN_ID` first.** This
package was built and verified on a shared research server where the default
domain (0) is visible machine-wide over UDP multicast; with the default
domain, publishers and subscribers were registered correctly (`ros2 topic
list`/`info` worked, showed real publisher/subscriber counts) but message
delivery to a *freshly-started* subscriber was unreliable, almost certainly
DDS discovery contention with other users' unrelated ROS 2 traffic on the
same domain. Picking a private domain fixed it completely and consistently:

```bash
export ROS_DOMAIN_ID=77   # any number 0-232 not in use by others on the machine; set
                           # it in every terminal you use for this package, including
                           # the one running the dashboard's rosbridge connection
ros2 launch guarded_eval demo.launch.py
```

**A separate, non-bug thing to know if you poke topics manually while it's
running**: `selection_node` stops publishing once `gen` reaches
`max_generations` (30 by default, ~30 seconds in) and goes idle until you
publish a `reset`. A `ros2 topic echo /metrics --once` run *after* it's
already finished will hang until you reset it, that's not a hang in the
node, the run is just over. `ros2 topic pub /control std_msgs/String
'{data: "{\"cmd\": \"reset\"}"}' --once` starts a fresh 30-generation run.

Then open the dashboard **with the port as a `?port=` query string** (the
page reads it from the URL, it's not a hardcoded constant to edit anymore --
`run_demo.sh` prints the exact URL to use). Either serve it:

```bash
cd src/guarded_eval/dashboard
python3 -m http.server 8000
# then open http://localhost:8000/dashboard.html?port=19091
# (use whatever port run_demo.sh printed, or 9090 if you launched manually
# with no port:= override)
```

or just open `dashboard.html?port=19091` directly as a `file://` URL in a
browser. No query string at all falls back to rosbridge's own default,
9090.

## Poking it from the CLI

Remember to `export ROS_DOMAIN_ID=<same number>` in this shell too if you set
one for the launch (see above) -- otherwise these commands are on a
different DDS domain and will see nothing.

```bash
ros2 topic list
ros2 topic echo /metrics
ros2 topic echo /robot_state
ros2 topic pub /control std_msgs/String '{data: "{\"cmd\": \"pause\"}"}' --once
ros2 topic pub /control std_msgs/String '{data: "{\"cmd\": \"run\"}"}' --once
ros2 topic pub /control std_msgs/String '{data: "{\"cmd\": \"reset\"}"}' --once
```

## Seed notes (both real, neither cherry-picked to force a story)

- **seed 9** (the launch default): naive's champion never discovers real
  confirming behavior at all, held-out `truth` stays at ~0.000 the entire
  30-generation run while its proxy score climbs substantially. Guarded's
  gate starts starved (0 passers for several generations) but recovers,
  eventually holding 6–11 of 24 candidates passing per generation, and its
  held-out truth becomes consistently nonzero (roughly 0.03–0.11) from
  around generation 16 on.
- **seed 7**: a genuine failure mode of the *guarded* mechanism, not
  evidence against the general idea. The cheap gate (2 rollouts, 3 tasks)
  starves to zero passers for the entire run on this particular random seed,
  so guarded selection degenerates into pure random search the whole time
  too. This is a real, disclosed limitation of this specific gate design
  (a fixed rollout budget and threshold, no fallback for "the gate itself
  is miscalibrated for this population"), not a claim that guarding never
  helps. See `roboworld-goodhart/NOTE.md` for the full 30-seed statistical
  picture this single-package demo is built to illustrate live.

Don't expect every seed to show a clean "naive collapses, guarded holds"
story: some do, some don't, and the ones that don't (like seed 7) are
honestly reported here rather than tuned away.

## Package layout

```
guarded_eval_ws/src/guarded_eval/
  package.xml, setup.py, setup.cfg, resource/guarded_eval   ROS 2 ament_python package files
  guarded_eval/core.py            pure numpy: arm kinematics, policy family,
                                   proxy_score/truth_score/cheap_gate,
                                   evolve_naive/evolve_guarded. No ROS imports;
                                   independently testable/importable.
  guarded_eval/selection_node.py  runs both selection loops, one generation
                                   per timer tick; publishes champions + metrics
  guarded_eval/sim_node.py        steps both live arms in lock-step on a
                                   shared target; publishes joint states +
                                   robot_state for the dashboard to draw
  launch/demo.launch.py           rosbridge + both nodes together
  dashboard/dashboard.html        self-contained dark-themed browser dashboard
                                   (roslib.js from CDN, everything else inline)
```

## No-sudo ROS 2 via RoboStack

If ROS 2 isn't installed and you can't (or don't want to) `sudo apt install`
it system-wide, [RoboStack](https://robostack.github.io/) ships ROS 2 as
ordinary conda-forge packages, entirely in user space:

```bash
conda create -n ros_env python=3.11 -y
conda activate ros_env
conda config --env --add channels robostack-jazzy
conda config --env --add channels conda-forge
conda config --env --remove channels defaults 2>/dev/null || true
conda install ros-jazzy-ros-base ros-jazzy-rosbridge-suite colcon-common-extensions -y
```

then `colcon build`/`ros2 launch` exactly as above, from inside that conda
environment. This is genuinely how this package's own build was verified on
a machine with no system ROS 2 and no sudo access, not a hypothetical
fallback.
