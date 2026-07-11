"""Launches rosbridge (port 9090 by default, for the browser dashboard),
selection_node, and sim_node together. Launch args: seed:=9,
max_generations:=30, port:=9090.

  ros2 launch guarded_eval demo.launch.py
  ros2 launch guarded_eval demo.launch.py seed:=7 max_generations:=20
  ros2 launch guarded_eval demo.launch.py port:=9091   # if 9090 is already taken

(port is a real launch arg, not hardcoded, because on a shared machine 9090
may already be bound by someone else's process -- dashboard.html's WS_PORT
constant needs to match whatever you pass here.)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    seed_arg = DeclareLaunchArgument("seed", default_value="9",
                                      description="RNG seed for selection_node's populations/tasks")
    max_gen_arg = DeclareLaunchArgument("max_generations", default_value="30",
                                         description="Number of generations selection_node runs before stopping")
    port_arg = DeclareLaunchArgument("port", default_value="9090",
                                      description="rosbridge websocket port")

    seed = LaunchConfiguration("seed")
    max_generations = LaunchConfiguration("max_generations")
    port = LaunchConfiguration("port")

    rosbridge = Node(
        package="rosbridge_server",
        executable="rosbridge_websocket",
        name="rosbridge_websocket",
        parameters=[{"port": port}],
        output="screen",
    )

    selection = Node(
        package="guarded_eval",
        executable="selection_node",
        name="selection_node",
        parameters=[{
            "pop_size": 24,
            "n_elites": 6,
            "max_generations": max_generations,
            "gen_period_s": 1.0,
            "seed": seed,
        }],
        output="screen",
    )

    sim = Node(
        package="guarded_eval",
        executable="sim_node",
        name="sim_node",
        parameters=[{
            "rate_hz": 20.0,
            "seed": 1,
        }],
        output="screen",
    )

    return LaunchDescription([seed_arg, max_gen_arg, port_arg, rosbridge, selection, sim])
