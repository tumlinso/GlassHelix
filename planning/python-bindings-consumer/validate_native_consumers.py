"""Run current synthetic installed consumers under the CUDA controller lease."""
import os
import subprocess
import sys
assert os.environ.get("TODO_GPU_LEASE_RECEIPT"), "CUDA controller lease required"
subprocess.run([sys.executable, "-B", "tests/learning/installed_consumer.py", "--gpu", "--gh-prefix", "/tmp/gh-bindings-installed-sdk", "--output", "/tmp/gh-native-owner.json"], check=True)
subprocess.run([sys.executable, "-B", "-m", "experiments.celltag_reprogramming.pilot", "--fixture", "--gpu", "--gh-prefix", "/tmp/gh-bindings-installed-sdk", "--output", "/tmp/gh-shared-support-native.json"], check=True)
