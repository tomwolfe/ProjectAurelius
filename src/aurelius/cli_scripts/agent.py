#!/usr/bin/env python3
"""Autonomous Screening Agent — Project Aurelius.

CLI entry point that delegates to the consolidated run_screening in loop.py.
"""

from __future__ import annotations

import logging
import sys

from aurelius.agent.loop import AgentConfig, run_screening

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("aurelius_agent")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aurelius Autonomous Screening Agent")
    parser.add_argument("--max-generations", type=int, default=50, help="Maximum generations to run")
    parser.add_argument("--batch-size", type=int, default=50, help="Candidates per batch")
    args = parser.parse_args()

    try:
        cfg = AgentConfig(max_generations=args.max_generations, batch_size=args.batch_size)
        run_screening(cfg)
    except KeyboardInterrupt:
        print("\n[AGENT] Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        log.error("Fatal error: %s", e, exc_info=True)
        print(f"\n[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
