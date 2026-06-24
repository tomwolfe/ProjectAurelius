"""Pytest configuration — adds engine/server/ to sys.path for API tests."""

from __future__ import annotations

import os
import sys

_server_dir = os.path.join(os.path.dirname(__file__), "..", "server")
if os.path.isdir(_server_dir):
    sys.path.insert(0, _server_dir)
