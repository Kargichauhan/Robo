import os
from glob import glob

from setuptools import find_packages, setup

package_name = "guarded_eval"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "dashboard"), glob("dashboard/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="kchauha3",
    maintainer_email="kchauha3@example.com",
    description=(
        "Live side-by-side demo of naive vs. guarded selection under "
        "optimization pressure against a cheap proxy evaluator."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "selection_node = guarded_eval.selection_node:main",
            "sim_node = guarded_eval.sim_node:main",
        ],
    },
)
