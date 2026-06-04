#!/usr/bin/env bash
#
# record_demo.sh — Record an asciinema terminal demo of Aurelius.
#
# Prerequisites (install locally — NOT in project dependencies):
#   brew install asciinema          # macOS
#   pipx install asciinema          # alternative via pipx
#
#   brew install agg                # asciinema GIF generator (asciinema/agg)
#   pipx install asciinema-agg      # alternative via pipx
#
# Usage:
#   ./scripts/record_demo.sh
#
# This will produce:
#   demo.cast  — raw asciinema recording (can be embedded in Markdown)
#
# To convert to GIF for README:
#   agg demo.cast demo.gif          # defaults to 120px font, 80x24 terminal
#   agg --font-size 16 demo.cast demo.gif   # smaller font for README embedding
#
# Notes:
#   - The --idle-time-limit=5 flag collapses idle gaps >5s into a
#     single frame, avoiding dead air during RDKit import overhead.
#   - This script is for LOCAL execution only. asciinema requires an
#     interactive TTY and will fail in CI.

asciinema rec \
  -c "aurelius doctor && aurelius agent --max-generations 2 --batch-size 5" \
  --idle-time-limit=5 \
  demo.cast
