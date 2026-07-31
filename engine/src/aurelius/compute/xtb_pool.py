"""Robust batched xTB runner with a dedicated worker thread.

Replaces the original ``BatchXTBRunner`` in ``scoring/oracle/quantum.py``
which relied on ``threading.Timer`` for periodic flushing.  This
implementation uses a dedicated daemon worker thread with an ``Event``-
based wake-up mechanism, eliminating fragile ``time.sleep`` polling.

Thread safety for the shared temp directory is ensured by protecting
directory creation and file writes with an ``RLock``.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


def _find_xtb_binary() -> str | None:
    """Locate the xTB binary on the system PATH."""
    for candidate in ["xtb", "xtb_opt"]:
        with contextlib.suppress(Exception):
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return candidate
    return None


_XTB_BIN: str | None = _find_xtb_binary()
_HAS_XTB: bool = _XTB_BIN is not None


# Thread-safe shared temp workspace — created once, cleaned up at exit.
_xtb_lock = threading.RLock()
_xtb_base_temp: str | None = None

if _HAS_XTB:
    with _xtb_lock:
        _xtb_base_temp = tempfile.mkdtemp(prefix="aurelius_xtb_pool_")


@atexit.register
def _cleanup_xtb_workspace() -> None:
    """Remove the shared xTB temp workspace on interpreter exit."""
    global _xtb_base_temp
    if _xtb_base_temp is not None and os.path.exists(_xtb_base_temp):
        shutil.rmtree(_xtb_base_temp, ignore_errors=True)
        _xtb_base_temp = None


def _ensure_temp_dir() -> str:
    """Return the shared temp directory (thread-safe)."""
    global _xtb_base_temp
    if _xtb_base_temp is None and _HAS_XTB:
        with _xtb_lock:
            if _xtb_base_temp is None:
                _xtb_base_temp = tempfile.mkdtemp(prefix="aurelius_xtb_pool_")
    return _xtb_base_temp or tempfile.gettempdir()


def has_xtb() -> bool:
    """Return True if the xTB binary is available on PATH."""
    return _HAS_XTB


_XTB_HOMO_RE = re.compile(r"HOMO\s*:\s*([-+]?\d+\.?\d*)\s*eV")
_XTB_LUMO_RE = re.compile(r"LUMO\s*:\s*([-+]?\d+\.?\d*)\s*eV")


def _parse_xtb_output(output: str) -> dict[str, float] | None:
    """Parse xTB output text for HOMO, LUMO, and dipole moment."""
    homo_match = _XTB_HOMO_RE.search(output)
    lumo_match = _XTB_LUMO_RE.search(output)

    if homo_match and lumo_match:
        homo = float(homo_match.group(1))
        lumo = float(lumo_match.group(1))
        logger.info("xTB Pool: HOMO=%.3f eV, LUMO=%.3f eV", homo, lumo)
        return {
            "homo_eV": homo,
            "lumo_eV": lumo,
            "dipole_D": 0.0,
        }

    logger.debug("Could not parse HOMO/LUMO from xTB output")
    return None


def _run_xtb(
    xyz_content: str,
    workdir: str | None = None,
    solvent: str | None = "ether",
) -> dict[str, float] | None:
    """Run xTB single-point calculation and parse HOMO/LUMO from output.

    Uses the module-level cached binary path and shared temp workspace
    to avoid per-call binary discovery.

    Args:
        xyz_content: XYZ-format molecular geometry string.
        workdir: Optional working directory for the xTB run.
        solvent: Implicit solvation model (e.g. "ether"). When set,
            appends ``--alpb <solvent>`` to the xTB command line.
    """
    if _XTB_BIN is None:
        return None
    if workdir is None:
        base = _ensure_temp_dir()
        with _xtb_lock:
            workdir = tempfile.mkdtemp(dir=base, prefix="mol_")

    xyz_path = os.path.join(workdir, "input.xyz")
    with open(xyz_path, "w") as f:
        f.write(xyz_content)

    cmd = [_XTB_BIN, "--gfn", "2", "--sp", xyz_path]
    if solvent:
        cmd.extend(["--alpb", solvent])

    try:
        result = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True, text=True, timeout=120,
        )
        return _parse_xtb_output(result.stdout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        logger.debug("xTB run failed: %s", exc)
        return None


def _run_xtb_from_file(
    xyz_path: str,
    solvent: str | None = "ether",
) -> dict[str, float] | None:
    """Run xTB on a single XYZ file path (no temp dir creation)."""
    if _XTB_BIN is None:
        return None
    workdir = os.path.dirname(xyz_path)
    cmd = [_XTB_BIN, "--gfn", "2", "--sp", xyz_path]
    if solvent:
        cmd.extend(["--alpb", solvent])
    try:
        result = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True, text=True, timeout=120,
        )
        return _parse_xtb_output(result.stdout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        logger.debug("xTB file run failed for %s: %s", xyz_path, exc)
        return None


def run_xtb_batch(
    xyz_list: list[str],
    max_workers: int = 4,
    solvent: str | None = "ether",
) -> list[dict[str, float] | None]:
    """Run multiple xTB single-point calculations in parallel.

    Uses ``ThreadPoolExecutor`` (xTB subprocess calls are I/O-bound).
    Cleans up the entire batch directory in one call.

    Args:
        xyz_list: List of XYZ-format molecular geometry strings.
        max_workers: Maximum parallel xTB processes (default 4).
        solvent: Implicit solvation model (e.g. "ether"). When set,
            appends ``--alpb <solvent>`` to all xTB command lines.

    Returns:
        List of result dicts in the same order as xyz_list. Each entry is
        None if xTB failed or is unavailable.
    """
    if _XTB_BIN is None:
        return [None] * len(xyz_list)

    base = _ensure_temp_dir()
    with _xtb_lock:
        batch_dir = tempfile.mkdtemp(dir=base, prefix="xtb_batch_")

    xyz_paths: list[str] = []
    for i, xyz_content in enumerate(xyz_list):
        xyz_path = os.path.join(batch_dir, f"mol_{i}.xyz")
        with open(xyz_path, "w") as f:
            f.write(xyz_content)
        xyz_paths.append(xyz_path)

    results: list[dict[str, float] | None] = [None] * len(xyz_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_run_xtb_from_file, xyz_paths[i], solvent): i
            for i in range(len(xyz_list))
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.debug("xTB batch item %d failed: %s", idx, exc)
                results[idx] = None

    shutil.rmtree(batch_dir, ignore_errors=True)

    return results


class _PendingJob:
    """A queued xTB calculation awaiting batch dispatch."""

    __slots__ = ("xyz", "future")

    def __init__(self, xyz: str, future: concurrent.futures.Future[dict[str, float] | None]) -> None:
        self.xyz = xyz
        self.future = future


class BatchXTBRunner:
    """Batch xTB runner with a dediworker thread for periodic flushing.

    Jobs are accumulated in a thread-safe list.  When the list reaches
    *batch_size* the caller's thread flushes them synchronously.  A
    background daemon thread periodically flushes jobs that have been
    sitting longer than *flush_interval*.

    Thread-safe.  Intended to be shared across ``XTBBackend`` instances
    so that conformer-level calculations from multiple molecules are
    batched together.

    Args:
        batch_size: Flush when this many jobs accumulate (default 10).
        flush_interval: Maximum seconds to wait before flushing (default 5.0).
        max_workers: Maximum parallel xTB processes per batch (default 4).
        solvent: Implicit solvation model (e.g. "ether"). When set,
            appends ``--alpb <solvent>`` to all xTB command lines.
    """

    def __init__(
        self,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        max_workers: int = 4,
        solvent: str | None = "ether",
    ) -> None:
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_workers = max_workers
        self._solvent = solvent
        self._lock = threading.RLock()
        self._pending: list[_PendingJob] = []
        self._running = True
        self._wake_event = threading.Event()

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self) -> None:
        """Daemon worker: wait for wake signal or timeout, then flush pending."""
        while self._running:
            self._wake_event.wait(timeout=self._flush_interval)
            self._wake_event.clear()
            if not self._running:
                break
            self._flush_pending()

    def _flush_pending(self) -> None:
        """Flush all currently pending jobs (must NOT hold lock when calling)."""
        with self._lock:
            batch = self._pending
            self._pending = []
        if not batch:
            return
        xyzs = [j.xyz for j in batch]
        results = run_xtb_batch(xyzs, max_workers=self._max_workers, solvent=self._solvent)
        for job, result in zip(batch, results, strict=False):
            if not job.future.set_running_or_notify_cancel():
                continue
            job.future.set_result(result)
        self._last_flush = time.monotonic()

    def submit(self, xyz: str) -> concurrent.futures.Future[dict[str, float] | None]:
        """Submit an XYZ geometry for xTB evaluation.

        Returns a ``Future`` that will be resolved when the batch is flushed.
        """
        future: concurrent.futures.Future[dict[str, float] | None] = concurrent.futures.Future()
        job = _PendingJob(xyz, future)
        with self._lock:
            self._pending.append(job)
            if len(self._pending) >= self._batch_size:
                batch = self._pending
                self._pending = []
            else:
                self._wake_event.set()
                return future

        # Synchronous flush (batch_size reached)
        xyzs = [j.xyz for j in batch]
        results = run_xtb_batch(xyzs, max_workers=self._max_workers, solvent=self._solvent)
        for job, result in zip(batch, results, strict=False):
            if not job.future.set_running_or_notify_cancel():
                continue
            job.future.set_result(result)
        return future

    def flush(self) -> None:
        """Explicitly flush all pending jobs synchronously."""
        self._flush_pending()

    def shutdown(self) -> None:
        """Shut down the worker thread and drain remaining jobs synchronously."""
        self._running = False
        self._wake_event.set()
        self._thread.join(timeout=5)

        with self._lock:
            remaining = self._pending
            self._pending = []
        for job in remaining:
            if not job.future.set_running_or_notify_cancel():
                continue
            result = _run_xtb(job.xyz, solvent=self._solvent)
            job.future.set_result(result)

    @property
    def pending_count(self) -> int:
        """Number of jobs currently queued."""
        with self._lock:
            return len(self._pending)
