"""Make `import splat_hitl` work from a fresh clone, with no install step.

Every test used to open with the same three lines of `sys.path` surgery. This
is the one place pytest looks for it, so those thirteen copies are gone and a
student can run the suite the moment they clone -- before reading a word about
packaging.

Installing properly still works and is what the flight machine does:

    pip install -e .
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
