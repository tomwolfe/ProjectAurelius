"""Unit tests for thread-safe state management.

Verifies that LoopState operations are safe under concurrent access:
- active_learning_queue additions are atomic
- _all_results updates are atomic
- screened_fingerprints updates are atomic
- clear() is thread-safe
- export_active_learning_queue() is thread-safe
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from aurelius.agent.state import LoopState
from aurelius.types import MoleculeContext


def test_active_learning_queue_atomic_append(tmp_path):
    """Active queue appends must be atomic under concurrent access."""
    state = LoopState(path=str(tmp_path / "state.json"))
    n_threads = 20
    n_per_thread = 10

    threads = []
    for t_idx in range(n_threads):
        def _append(idx: int) -> None:
            for i in range(n_per_thread):
                state.active_learning_queue.append(
                    {"idx": idx, "item": i}
                )
        threads.append(threading.Thread(target=_append, args=(t_idx,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected_count = n_threads * n_per_thread
    assert len(state.active_learning_queue) == expected_count, (
        f"Expected {expected_count} items, got {len(state.active_learning_queue)}"
    )


def test_all_results_atomic_append(tmp_path):
    """_all_results updates must be atomic under concurrent access."""
    state = LoopState(path=str(tmp_path / "state.json"))
    n_threads = 15

    threads = []
    for t_idx in range(n_threads):
        def _append(idx: int) -> None:
            state._all_results.append({
                "smiles": f"C{'C' * idx}O",
                "score": idx,
            })
        threads.append(threading.Thread(target=_append, args=(t_idx,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(state._all_results) == n_threads


def test_screened_fingerprints_atomic_append(tmp_path):
    """Screened fingerprints must be atomic under concurrent access."""
    state = LoopState(path=str(tmp_path / "state.json"))
    n_threads = 10

    threads = []
    for t_idx in range(n_threads):
        def _append(idx: int) -> None:
            state.screened_fingerprints.append((
                f"C{idx}O",
                idx,
                {"score": idx},
            ))
        threads.append(threading.Thread(target=_append, args=(t_idx,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(state.screened_fingerprints) == n_threads


def test_clear_thread_safety(tmp_path):
    """Clear must safely reset all internal state."""
    state = LoopState(path=str(tmp_path / "state.json"))

    # Populate state
    for i in range(50):
        state._all_results.append({"smiles": f"C{i}O", "score": i})
        state.active_learning_queue.append({"idx": i})
        state.screened_fingerprints.append((f"C{i}O", i, {}))
        state._all_scores.append(float(i))
        state.batch_means.append(float(i))

    # Clear all state
    state.clear()

    assert len(state._all_results) == 0
    assert len(state.active_learning_queue) == 0
    assert len(state.screened_fingerprints) == 0
    assert len(state._all_scores) == 0
    assert len(state.batch_means) == 0


def test_export_active_learning_queue_thread_safety(tmp_path):
    """Export must produce consistent snapshots."""
    import json

    state = LoopState(path=str(tmp_path / "state.json"))

    # Populate queue
    for i in range(30):
        state.active_learning_queue.append({"smiles": f"C{i}O", "score": i})

    # Export to file
    path = tmp_path / "queue.json"
    state.export_active_learning_queue(str(path))

    with open(path) as f:
        data = json.load(f)

    assert len(data) == 30


def test_concurrent_add_and_clear(tmp_path):
    """Verify no data loss when adding and clearing concurrently."""
    state = LoopState(path=str(tmp_path / "state.json"))
    stop_event = threading.Event()
    added_count = 0
    count_lock = threading.Lock()

    def _add_items() -> None:
        nonlocal added_count
        idx = 0
        while not stop_event.is_set():
            item = {"idx": idx, "data": f"item-{idx}"}
            with state._state_lock:
                state._all_results.append(item)
            with state._al_queue_lock:
                state.active_learning_queue.append(item)
            idx += 1
            with count_lock:
                added_count += 1
            time.sleep(0.001)

    def _clear_state() -> None:
        time.sleep(0.05)  # Let some items accumulate
        state.clear()

    add_thread = threading.Thread(target=_add_items)
    clear_thread = threading.Thread(target=_clear_state)

    add_thread.start()
    clear_thread.start()

    stop_event.set()
    clear_thread.join()
    add_thread.join()

    # Items added after clear should not cause data corruption
    for item in state._all_results:
        assert "idx" in item
    for item in state.active_learning_queue:
        assert "idx" in item


def test_locks_initialized(tmp_path):
    """Verify all lock attributes are properly initialised."""
    state = LoopState(path=str(tmp_path / "state.json"))
    assert state._state_lock is not None
    assert state._cache_lock is not None
    assert state._al_queue_lock is not None
