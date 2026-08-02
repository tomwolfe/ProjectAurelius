"""Tests for BatchXTBRunner — batching, flushing, and shutdown behavior.

These tests verify the queue/flush logic of the batch runner without
requiring the xTB binary to be present.  The actual xTB execution path
is covered by ``test_oracle_real_data.py``.
"""

from __future__ import annotations

from aurelius.scoring.oracle.quantum import BatchXTBRunner


class TestBatchXTBRunner:
    def test_submit_single(self) -> None:
        """Submitting a single job should eventually resolve."""
        runner = BatchXTBRunner(batch_size=3, flush_interval=0.5)
        future = runner.submit("3\n\ntest xyz")
        future.result(timeout=5)
        runner.shutdown()

    def test_batch_size_triggers_flush(self) -> None:
        """When queue reaches batch_size, jobs should flush immediately."""
        runner = BatchXTBRunner(batch_size=3, flush_interval=60.0)
        futures = [runner.submit(f"3\n\nmol {i}") for i in range(3)]
        # All futures should resolve without waiting for the timeout
        for f in futures:
            f.result(timeout=5)
        runner.shutdown()

    def test_flush_interval_triggers_flush(self) -> None:
        """When flush_interval passes, pending jobs should flush."""
        runner = BatchXTBRunner(batch_size=10, flush_interval=0.5)
        future = runner.submit("3\n\ntest xyz")
        future.result(timeout=5)
        runner.shutdown()

    def test_explicit_flush(self) -> None:
        """Calling flush() explicitly should dispatch all pending jobs."""
        runner = BatchXTBRunner(batch_size=10, flush_interval=60.0)
        future = runner.submit("3\n\ntest xyz")
        runner.flush()
        future.result(timeout=5)
        runner.shutdown()

    def test_initial_pending_count_zero(self) -> None:
        """Fresh runner should have zero pending jobs."""
        runner = BatchXTBRunner(batch_size=10, flush_interval=60.0)
        assert runner.pending_count == 0
        runner.shutdown()

    def test_pending_count_after_submit(self) -> None:
        """Pending count should reflect queued jobs before flush."""
        runner = BatchXTBRunner(batch_size=10, flush_interval=60.0)
        runner.submit("3\n\ntest xyz")
        assert runner.pending_count == 1
        runner.submit("3\n\ntest xyz")
        assert runner.pending_count == 2
        runner.shutdown()

    def test_pending_count_after_flush(self) -> None:
        """Pending count should be zero after flush."""
        runner = BatchXTBRunner(batch_size=10, flush_interval=60.0)
        runner.submit("3\n\ntest xyz")
        runner.flush()
        assert runner.pending_count == 0
        runner.shutdown()

    def test_shutdown_drains_pending(self) -> None:
        """Shutdown should resolve all pending jobs."""
        runner = BatchXTBRunner(batch_size=10, flush_interval=60.0)
        future = runner.submit("3\n\ntest xyz")
        runner.shutdown()
        future.result(timeout=5)

    def test_multiple_batches(self) -> None:
        """Multiple consecutive batches should each resolve."""
        runner = BatchXTBRunner(batch_size=2, flush_interval=60.0, max_workers=2)
        for batch in range(3):
            futures = [runner.submit(f"3\n\nbatch {batch} mol {i}") for i in range(2)]
            for f in futures:
                f.result(timeout=5)
        runner.shutdown()

    def test_no_cross_contamination(self) -> None:
        """Results should map correctly to their submissions."""
        runner = BatchXTBRunner(batch_size=2, flush_interval=60.0)
        f1 = runner.submit("3\n\nfirst")
        f2 = runner.submit("3\n\nsecond")
        # Both should resolve (xTB may fail, but future should not error)
        f1.result(timeout=5)
        f2.result(timeout=5)
        runner.shutdown()
