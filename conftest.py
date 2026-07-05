# conftest.py (repo root) — ensures src is on sys.path for bare pytest invocation
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
