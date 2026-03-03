# setup.py
from setuptools import setup, find_packages

setup(
    name="yolo-iterative-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.3.0",
        "torch>=2.0.0",
        "watchdog>=3.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "yolo-pipeline-init=pipeline.cli:init_project",
            "yolo-pipeline-watch=pipeline.watcher:main",
            "yolo-pipeline-train=pipeline.train:main",
            "yolo-pipeline-monitor=pipeline.monitor:main",
            "yolo-pipeline-score=pipeline.active_learning:main",
            "yolo-pipeline-export=pipeline.export:main",
            "yolo-pipeline-doctor=pipeline.doctor:main",
        ],
    },
    python_requires=">=3.8",
)
